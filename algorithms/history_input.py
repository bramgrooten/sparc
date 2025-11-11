"""
HistoryInput baseline.

This baseline trains an AdapterPolicy (history‑based policy) without
privileged context.  The policy receives the current observation and a
history of past observation–action pairs, but the critics still have
access to the ground‑truth context. The update logic is
essentially standard SAC/QR‑SAC: the critics take (obs, action, context),
the actor takes (obs, history).

Run this baseline by setting ``args.alg`` to ``history_input``.
"""
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import wandb

from utils.env_utils import make_env, make_eval_envs
from utils.log_utils import eval_agent, log_videos
from utils.modules import AdapterPolicy, CriticContext
from utils.replay_buffer_dict import ReplayBufferDict
from algorithms.qr_sac import _quantile_huber_loss, min_critic_quantiles

# Importing the custom Wind environments to register them in gym
from environments import wind_halfcheetah, wind_hopper, wind_walker2d, wind_ant  # noqa: F401


def main(args, run_name: str, device: torch.device):
    args.context_in_obs = True
    args.history_in_obs = True

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env, args.seed + i, i, run_name, args) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # evaluation envs
    eval_envs, eval_envs_names = make_eval_envs(args.env, args.seed, args.capture_video, run_name, args)
    if args.capture_video:
        video_dirs = [f"videos/{run_name}/eval_{name}" for name in eval_envs_names]
        video_dirs.append(f"videos/{run_name}/train")
        uploaded_videos = set()

    # initial observation to get shapes
    obs_dict, _ = envs.reset(seed=args.seed)
    obs = obs_dict["obs"]
    context = obs_dict["context"]
    history = obs_dict["history"]

    # shapes
    action_shape = envs.single_action_space.shape
    obs_shape = obs.shape[1:]
    context_shape = context.shape[1:]
    history_shape = history.shape[1:]

    # instantiate actor (AdapterPolicy) and critics
    actor = AdapterPolicy(envs, args.history_length, obs_shape, action_shape).to(device)
    num_quantiles = args.num_quantiles if args.sparc_expert_alg == "qr_sac" else 1
    q_kwargs = {
        "obs_shape": obs_shape,
        "context_shape": context_shape,
        "action_shape": action_shape,
        "num_quantiles": num_quantiles,
    }
    q1 = CriticContext(**q_kwargs).to(device)
    q2 = CriticContext(**q_kwargs).to(device)
    q1_target = CriticContext(**q_kwargs).to(device)
    q2_target = CriticContext(**q_kwargs).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    q_optimizer = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # replay buffer storing obs, context, history
    rb = ReplayBufferDict(
        args.buffer_size,
        obs_shape=obs_shape,
        context_shape=context_shape,
        history_shape=history_shape,
        action_shape=action_shape,
        device=device,
    )

    if args.sparc_expert_alg == "qr_sac":
        tau_hat = (torch.arange(args.num_quantiles, dtype=torch.float32, device=device) + 0.5) / args.num_quantiles

    start_time = time.time()
    for global_step in range(args.total_timesteps):
        # periodic evaluation
        if global_step % args.eval_freq == 0:
            names = [f"history_{name}" for name in eval_envs_names]
            # actor requires agent_type="adapter" so eval_agent will pass history
            eval_agent(actor, eval_envs, names, global_step, device, agent_type="adapter")

        # sample actions
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(
                torch.tensor(obs, dtype=torch.float32, device=device),
                torch.tensor(history, dtype=torch.float32, device=device),
            )
            actions = actions.detach().cpu().numpy()

        # step environment
        next_obs_dict, rewards, terminations, truncations, infos = envs.step(actions)
        # handle final observations
        real_next_obs_dict = next_obs_dict.copy()
        if "final_observation" in infos:
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs_dict[idx] = infos["final_observation"][idx]

        # log training episodes
        if "episode" in infos:
            for i, finished in enumerate(infos["episode"]['_r']):
                if finished:
                    log_dict = {
                        "train/episode_return": infos["episode"]['r'][i],
                        "train/episode_length": infos["episode"]['l'][i],
                    }
                    wandb.log(log_dict, step=global_step)

        # store transitions
        for env_idx in range(args.num_envs):
            rb.add(
                obs[env_idx],
                context[env_idx],
                history[env_idx],
                real_next_obs_dict["obs"][env_idx],
                real_next_obs_dict["context"][env_idx],
                actions[env_idx],
                rewards[env_idx],
                terminations[env_idx],
            )

        # update obs, context, history
        obs_dict = next_obs_dict
        obs = obs_dict["obs"]
        context = obs_dict["context"]
        history = obs_dict["history"]

        # update networks
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            batch_obs = data["obs"]
            batch_context = data["context"]
            batch_history = data["history"]
            batch_next_obs = data["next_obs"]
            batch_next_context = data["next_context"]
            batch_actions = data["actions"]
            batch_rewards = data["rewards"]
            batch_dones = data["dones"]

            with torch.no_grad():
                next_actions, next_log_pi, _ = actor.get_action(batch_next_obs, batch_history)
                q1_next = q1_target(batch_next_obs, next_actions, batch_next_context)
                q2_next = q2_target(batch_next_obs, next_actions, batch_next_context)
                if args.sparc_expert_alg == "sac":
                    min_q_next = torch.min(q1_next, q2_next) - alpha * next_log_pi
                    target_q = batch_rewards + (1.0 - batch_dones) * args.gamma * min_q_next
                else:
                    min_next_quantiles = min_critic_quantiles(q1_next, q2_next) - alpha * next_log_pi
                    target_q = batch_rewards.unsqueeze(1) + (1.0 - batch_dones.unsqueeze(1)) * args.gamma * min_next_quantiles

            # current Q estimates
            q1_pred = q1(batch_obs, batch_actions, batch_context)
            q2_pred = q2(batch_obs, batch_actions, batch_context)

            if args.sparc_expert_alg == "sac":
                q1_loss = F.mse_loss(q1_pred, target_q)
                q2_loss = F.mse_loss(q2_pred, target_q)
            else:
                q1_loss = _quantile_huber_loss(q1_pred, target_q, tau_hat, args.quantile_kappa)
                q2_loss = _quantile_huber_loss(q2_pred, target_q, tau_hat, args.quantile_kappa)
            q_loss = q1_loss + q2_loss
            q_optimizer.zero_grad()
            q_loss.backward()
            if args.grad_clip_critic > 0:
                torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), max_norm=args.grad_clip_critic)
            q_optimizer.step()

            # actor update
            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(batch_obs, batch_history)
                    q1_pi = q1(batch_obs, pi, batch_context)
                    q2_pi = q2(batch_obs, pi, batch_context)
                    if args.sparc_expert_alg == "sac":
                        min_q_pi = torch.min(q1_pi, q2_pi)
                    else:
                        min_q_pi = min_critic_quantiles(q1_pi, q2_pi).mean(dim=1, keepdim=True)
                    actor_loss = ((alpha * log_pi) - min_q_pi).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi_detach, _ = actor.get_action(batch_obs, batch_history)
                        alpha_loss = (-log_alpha.exp() * (log_pi_detach + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(q1.parameters(), q1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(q2.parameters(), q2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # logging
            if global_step % args.log_freq == 0:
                log_dict = {
                    "losses/q1_loss": q1_loss.item(),
                    "losses/q2_loss": q2_loss.item(),
                    "losses/q_loss": q_loss.item() / 2.0,
                    "losses/actor_loss": actor_loss.item(),
                    "losses/alpha": alpha,
                    "throughput/SPS": int(global_step / (time.time() - start_time)),
                }
                if args.autotune:
                    log_dict["losses/alpha_loss"] = alpha_loss.item()
                wandb.log(log_dict, step=global_step)
                if args.capture_video:
                    uploaded_videos = log_videos(global_step, uploaded_videos, video_dirs)

    envs.close()
    eval_envs.close()
