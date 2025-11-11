import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import wandb
from stable_baselines3.common.buffers import ReplayBuffer
from utils.modules import Actor, Critic
from utils.env_utils import make_env, make_eval_envs
from utils.log_utils import log_videos, eval_agent

# Importing the custom Wind environments to register them in gym
from environments import wind_halfcheetah, wind_hopper, wind_walker2d, wind_ant  # noqa: F401



def main(args, run_name, device):
    args.history_in_obs = False
    args.context_in_obs = False

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

    actor = Actor(envs).to(device)
    qf1 = Critic(envs).to(device)
    qf2 = Critic(envs).to(device)
    qf1_target = Critic(envs).to(device)
    qf2_target = Critic(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):

        # periodical evaluation
        if global_step % args.eval_freq == 0:
            eval_agent(actor, eval_envs, eval_envs_names, global_step, device)

        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "episode" in infos:
            for i, finished in enumerate(infos["episode"]['_r']):
                if finished:
                    log_dict = {
                        "train/episode_return": infos["episode"]['r'][i],
                        "train/episode_length": infos["episode"]['l'][i],
                    }
                    wandb.log(log_dict, step=global_step)

        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the critics
            q_optimizer.zero_grad()
            qf_loss.backward()
            if args.grad_clip_critic > 0:
                torch.nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), max_norm=args.grad_clip_critic)
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                # compensate for the delay by doing 'policy_frequency' updates instead of 1
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    # optimize the actor
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # logging
            if global_step % args.log_freq == 0:
                log_dict = {
                    "losses/qf1_values": qf1_a_values.mean().item(),
                    "losses/qf2_values": qf2_a_values.mean().item(),
                    "losses/qf1_loss": qf1_loss.item(),
                    "losses/qf2_loss": qf2_loss.item(),
                    "losses/qf_loss": qf_loss.item() / 2.0,
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
