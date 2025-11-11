"""
SPARC - Single-Phase Adaptation for Robust Control
Modules live in utils/modules.py (ExpertPolicy, AdapterPolicy, CriticContext).
"""
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import wandb
from utils.replay_buffer_dict import ReplayBufferDict
from utils.env_utils import make_env, make_eval_envs
from utils.log_utils import log_videos, eval_agent
from utils.modules import ExpertPolicy, AdapterPolicy, CriticContext
from utils.misc_utils import copy_modules
from algorithms.qr_sac import _quantile_huber_loss, min_critic_quantiles

# Import wind environments to register them in gym
from environments import wind_halfcheetah, wind_hopper, wind_walker2d, wind_ant  # noqa: F401



def main(args, run_name: str, device: torch.device):
    args.history_in_obs = True
    args.context_in_obs = True

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
    history = obs_dict["history"]

    # shapes
    action_shape = envs.single_action_space.shape  # (act_dim,)
    obs_shape = obs.shape[1:]  # (obs_dim,)
    context_shape = context.shape[1:]  # (2,) for wind envs
    history_shape = history.shape[1:]  # (H, obs_dim + act_dim)

    expert_policy = ExpertPolicy(envs, obs_shape, context_shape, action_shape).to(device)
    adapter_policy = AdapterPolicy(envs, args.history_length, obs_shape, action_shape).to(device)
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
    adapter_optimizer = optim.Adam(list(adapter_policy.adapter.parameters()), lr=args.adapter_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    rb = ReplayBufferDict(
        args.buffer_size,
        obs_shape=obs_shape,
        context_shape=context_shape,
        history_shape=history_shape,
        action_shape=action_shape,
        device=device,
    )

    if args.sparc_expert_alg == 'qr_sac':
        # Precompute quantile midpoints tau_hat = (i+0.5)/N
        tau_hat = (torch.arange(args.num_quantiles, dtype=torch.float32, device=device) + 0.5) / args.num_quantiles

    start_time = time.time()
    for global_step in range(args.total_timesteps):

        # periodical evaluation
        if global_step % args.eval_freq == 0:
            names_expert = []
            for name in eval_envs_names:
                names_expert.append(f"expert_{name}")
            eval_agent(expert_policy, eval_envs, names_expert, global_step, device, agent_type="expert")
            names_adapter = []
            for name in eval_envs_names:
                names_adapter.append(f"adapter_{name}")
            eval_agent(adapter_policy, eval_envs, names_adapter, global_step, device, agent_type="adapter")

        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            if args.rollout_policy == "expert":
                actions, _, _ = expert_policy.get_action(torch.Tensor(obs).to(device), torch.Tensor(context).to(device))
            elif args.rollout_policy == "adapter":
                actions, _, _ = adapter_policy.get_action(torch.Tensor(obs).to(device), torch.Tensor(history).to(device))
            else:
                raise ValueError(f"Unknown rollout policy: {args.rollout_policy}.")
            actions = actions.detach().cpu().numpy()

        next_obs_dict, rewards, terminations, truncations, infos = envs.step(actions)

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

        for env_idx in range(args.num_envs):
            obs = obs_dict["obs"][env_idx]
            context = obs_dict["context"][env_idx]
            history = obs_dict["history"][env_idx]
            next_obs = real_next_obs_dict["obs"][env_idx]
            next_context = real_next_obs_dict["context"][env_idx]
            action = actions[env_idx]
            reward = rewards[env_idx]
            done = terminations[env_idx]
            rb.add(obs, context, history, next_obs, next_context, action, reward, done)

        obs_dict = next_obs_dict
        obs = obs_dict["obs"]
        context = obs_dict["context"]
        history = obs_dict["history"]

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
                # next actions and log‑probs from the expert
                next_actions, next_log_pi, _ = expert_policy.get_action(batch_next_obs, batch_next_context)
                # critic targets
                q1_next = tq1(batch_next_obs, next_actions, batch_next_context)
                q2_next = tq2(batch_next_obs, next_actions, batch_next_context)

                if args.sparc_expert_alg == "sac":
                    min_q_next = torch.min(q1_next, q2_next) - alpha * next_log_pi
                    target_q = batch_rewards + (1.0 - batch_dones) * args.gamma * min_q_next.view(-1, 1)
                else:  # qr_sac
                    min_next_quantiles = min_critic_quantiles(q1_next, q2_next) - alpha * next_log_pi
                    target_q = batch_rewards.unsqueeze(1) + (1.0 - batch_dones.unsqueeze(1)) * args.gamma * min_next_quantiles

            # current Q estimates
            q1_pred = q1(batch_obs, batch_actions, batch_context)
            q2_pred = q2(batch_obs, batch_actions, batch_context)

            # critic losses
            if args.sparc_expert_alg == "sac":
                q1_loss = F.mse_loss(q1_pred, target_q)
                q2_loss = F.mse_loss(q2_pred, target_q)
            else:  # qr_sac
                q1_loss = _quantile_huber_loss(q1_pred, target_q, tau_hat, args.quantile_kappa)
                q2_loss = _quantile_huber_loss(q2_pred, target_q, tau_hat, args.quantile_kappa)

            q_loss = q1_loss + q2_loss
            q_optimizer.zero_grad()
            q_loss.backward()
            if args.grad_clip_critic > 0:
                torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), max_norm=args.grad_clip_critic)
            q_optimizer.step()

            # policy update on the expert at specified frequency
            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = expert_policy.get_action(batch_obs, batch_context)
                    q1_pi = q1(batch_obs, pi, batch_context)
                    q2_pi = q2(batch_obs, pi, batch_context)
                    if args.sparc_expert_alg == "sac":
                        min_q_pi = torch.min(q1_pi, q2_pi)
                    else:
                        min_q_pi = min_critic_quantiles(q1_pi, q2_pi).mean(dim=1, keepdim=True)
                    actor_loss = ((alpha * log_pi) - min_q_pi).mean()
                    expert_optimizer.zero_grad()
                    actor_loss.backward()
                    expert_optimizer.step()

                    if args.autotune:
                        # entropy coefficient update
                        with torch.no_grad():
                            _, new_log_pi, _ = expert_policy.get_action(batch_obs, batch_context)
                        alpha_loss = (-log_alpha.exp() * (new_log_pi + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

                # copy expert policy parameters to adapter policy
                copy_modules(expert_policy, adapter_policy, ["fc1", "fc2", "fc3", "fc_mean", "fc_logstd"])

            # soft update target critics
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(q1.parameters(), tq1.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(q2.parameters(), tq2.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # history adapter regression: align history embedding with context embedding
            with torch.no_grad():
                z = expert_policy.encoder(batch_context)   # latent context encoding from expert
            z_hat = adapter_policy.adapter(batch_history)  # latent history encoding from adapter
            adapter_loss = F.mse_loss(z_hat, z.detach())
            adapter_optimizer.zero_grad()
            adapter_loss.backward()
            adapter_optimizer.step()

            # logging
            if global_step % args.log_freq == 0:
                log_dict = {
                    "losses/q1_pred": q1_pred.mean().item(),
                    "losses/q2_pred": q1_pred.mean().item(),
                    "losses/q1_loss": q1_loss.item(),
                    "losses/q2_loss": q2_loss.item(),
                    "losses/q_loss": q_loss.item() / 2.0,
                    "losses/actor_loss": actor_loss.item(),
                    "losses/alpha": alpha,
                    "losses/history_adapter_loss": adapter_loss.item(),
                    "throughput/SPS": int(global_step / (time.time() - start_time)),
                }
                if args.autotune:
                    log_dict["losses/alpha_loss"] = alpha_loss.item()
                wandb.log(log_dict, step=global_step)
                if args.capture_video:
                    uploaded_videos = log_videos(global_step, uploaded_videos, video_dirs)

    envs.close()
    eval_envs.close()
