"""
RMA Phase 1 - Train the expert policy only.

This phase trains an ExpertPolicy with context information for a fixed number
of timesteps and saves the trained expert model to disk.  It uses the same
CriticContext architecture as SPARC but omits the history adapter entirely.

Usage: run via train.py with args.alg set to 'rma_phase1'.
Phase‑2 (adapter training) is implemented in ``rma_phase2.py``.
"""
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import wandb
from utils.env_utils import make_env, make_eval_envs
from utils.log_utils import eval_agent, log_videos
from utils.modules import ExpertPolicy, CriticContext
from utils.replay_buffer_dict import ReplayBufferExpert
from algorithms.qr_sac import _quantile_huber_loss, min_critic_quantiles

# Importing the custom Wind environments to register them in gym
from environments import wind_halfcheetah, wind_hopper, wind_walker2d, wind_ant  # noqa: F401


def main(args, run_name: str, device: torch.device):
    """
    Phase‑1 of RMA: train an ExpertPolicy with access to context.

    The environment is configured to include context in the observation but not
    the history.  We train the expert for ``args.total_timesteps`` steps using
    either SAC or QR‑SAC (controlled by ``args.sparc_expert_alg``).  At the end
    of training, the expert's state dict is saved to disk so that phase‑2 can
    load it.
    """
    # ensure we have context in the observation and no history for expert training
    args.context_in_obs = True
    args.history_in_obs = False

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env, args.seed + i, i, run_name, args) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # eval env setup
    eval_envs, eval_envs_names = make_eval_envs(args.env, args.seed, args.capture_video, run_name, args)
    if args.capture_video:
        video_dirs = [f"videos/{run_name}/eval_{eval_name}" for eval_name in eval_envs_names]
        video_dirs.append(f"videos/{run_name}/train")
        uploaded_videos = set()

    obs_dict, _ = envs.reset(seed=args.seed)
    obs = obs_dict["obs"]
    context = obs_dict["context"]

    # determine shapes
    action_shape = envs.single_action_space.shape
    obs_shape = obs.shape[1:]
    context_shape = context.shape[1:]

    # build networks
    expert_policy = ExpertPolicy(envs, obs_shape, context_shape, action_shape).to(device)
    num_quantiles = args.num_quantiles if args.sparc_expert_alg == "qr_sac" else 1
    q_kwargs = {"obs_shape": obs_shape, "context_shape": context_shape, "action_shape": action_shape, "num_quantiles": num_quantiles}
    q1 = CriticContext(**q_kwargs).to(device)
    q2 = CriticContext(**q_kwargs).to(device)
    tq1 = CriticContext(**q_kwargs).to(device)
    tq2 = CriticContext(**q_kwargs).to(device)
    tq1.load_state_dict(q1.state_dict())
    tq2.load_state_dict(q2.state_dict())
    q_optimizer = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.q_lr)
    expert_optimizer = optim.Adam(list(expert_policy.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # replay buffer
    rb = ReplayBufferExpert(args.buffer_size, obs_shape, context_shape, action_shape, device)

    # Precompute tau_hat for QR‑SAC if needed
    if args.sparc_expert_alg == 'qr_sac':
        tau_hat = (torch.arange(args.num_quantiles, dtype=torch.float32, device=device) + 0.5) / args.num_quantiles

    start_time = time.time()
    for global_step in range(args.total_timesteps):
        # periodic evaluation
        if global_step % args.eval_freq == 0:
            names_expert = [f"expert_{name}" for name in eval_envs_names]
            eval_agent(expert_policy, eval_envs, names_expert, global_step, device, agent_type="expert")

        # sample actions
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = expert_policy.get_action(torch.tensor(obs, dtype=torch.float32, device=device),
                                                     torch.tensor(context, dtype=torch.float32, device=device))
            actions = actions.detach().cpu().numpy()

        next_obs_dict, rewards, terminations, truncations, infos = envs.step(actions)

        # logging of episode returns
        if "episode" in infos:
            for i, finished in enumerate(infos["episode"]['_r']):
                if finished:
                    log_dict = {
                        "train/episode_return": infos["episode"]['r'][i],
                        "train/episode_length": infos["episode"]['l'][i],
                    }
                    wandb.log(log_dict, step=global_step)

        real_next_obs_dict = next_obs_dict.copy()
        if "final_observation" in infos:
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs_dict[idx] = infos["final_observation"][idx]

        # store each environment separately
        for env_idx in range(args.num_envs):
            o   = obs_dict["obs"][env_idx]
            ctx = obs_dict["context"][env_idx]
            no  = real_next_obs_dict["obs"][env_idx]
            nctx = real_next_obs_dict["context"][env_idx]
            act = actions[env_idx]
            rew = rewards[env_idx]
            done = terminations[env_idx]
            rb.add(o, ctx, no, nctx, act, rew, done)

        # update current observation/context
        obs_dict = next_obs_dict
        obs = obs_dict["obs"]
        context = obs_dict["context"]

        # start gradient updates
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            batch_obs = data["obs"]
            batch_ctx = data["context"]
            batch_next_obs = data["next_obs"]
            batch_next_ctx = data["next_context"]
            batch_actions = data["actions"]
            batch_rewards = data["rewards"]
            batch_dones = data["dones"]

            with torch.no_grad():
                next_actions, next_log_pi, _ = expert_policy.get_action(batch_next_obs, batch_next_ctx)
                q1_next = tq1(batch_next_obs, next_actions, batch_next_ctx)
                q2_next = tq2(batch_next_obs, next_actions, batch_next_ctx)
                if args.sparc_expert_alg == "sac":
                    min_q_next = torch.min(q1_next, q2_next) - alpha * next_log_pi
                    target_q = batch_rewards + (1.0 - batch_dones) * args.gamma * min_q_next
                else: # QR‑SAC
                    min_next = min_critic_quantiles(q1_next, q2_next) - alpha * next_log_pi
                    target_q = batch_rewards.unsqueeze(1) + (1.0 - batch_dones.unsqueeze(1)) * args.gamma * min_next

            # current predictions
            q1_pred = q1(batch_obs, batch_actions, batch_ctx)
            q2_pred = q2(batch_obs, batch_actions, batch_ctx)

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

            # policy update
            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = expert_policy.get_action(batch_obs, batch_ctx)
                    q1_pi = q1(batch_obs, pi, batch_ctx)
                    q2_pi = q2(batch_obs, pi, batch_ctx)
                    if args.sparc_expert_alg == "sac":
                        min_q_pi = torch.min(q1_pi, q2_pi)
                    else: # QR‑SAC
                        min_q_pi = min_critic_quantiles(q1_pi, q2_pi).mean(dim=1, keepdim=True)
                    actor_loss = ((alpha * log_pi) - min_q_pi).mean()
                    expert_optimizer.zero_grad()
                    actor_loss.backward()
                    expert_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, new_log_pi, _ = expert_policy.get_action(batch_obs, batch_ctx)
                        alpha_loss = (-log_alpha.exp() * (new_log_pi + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(q1.parameters(), tq1.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(q2.parameters(), tq2.parameters()):
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

    # save the trained expert policy for phase‑2
    os.makedirs("models", exist_ok=True)
    expert_path = os.path.join("models", f"{run_name}_expert.pth")
    torch.save(expert_policy.state_dict(), expert_path)
    print(f"Saved expert policy to {expert_path}")
