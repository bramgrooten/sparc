"""
RMA Phase 2 - Train the adapter policy using a fixed expert.

This phase loads a previously trained expert policy from disk, freezes its
weights, and trains only the history adapter to match the expert's latent
context embedding.  During this phase, actions are still generated
using the adapter policy. The adapter learns via supervised regression
on the expert's context encoder outputs.

Usage: run via train.py with args.alg set to 'rma_phase2' and
``args.expert_model_path`` set to the path of the saved expert.
"""
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import wandb
from utils.env_utils import make_env, make_eval_envs
from utils.log_utils import eval_agent, log_videos
from utils.modules import ExpertPolicy, AdapterPolicy
from utils.misc_utils import copy_modules
from utils.replay_buffer_dict import ReplayBufferDict

# Importing the custom Wind environments to register them in gym
from environments import wind_halfcheetah, wind_hopper, wind_walker2d, wind_ant  # noqa: F401


def main(args, run_name: str, device: torch.device):
    """
    Phase‑2 of RMA: train the adapter (history) module with a fixed expert.

    The expert policy is loaded from ``args.expert_model_path``.  We freeze the
    expert weights and copy its observation encoder and decision layers
    into the adapter policy.  We then train only the history adapter using
    a mean‑squared error loss between the expert's latent context encoding and
    the adapter's latent history embedding.  No reinforcement learning is
    performed here.
    """
    # we need both context and history in the observation
    args.context_in_obs = True
    args.history_in_obs = True

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

    # shapes for replay buffer
    action_shape = envs.single_action_space.shape
    obs_shape = obs.shape[1:]
    context_shape = context.shape[1:]
    history_shape = history.shape[1:]

    # build expert and adapter
    expert_policy = ExpertPolicy(envs, obs_shape, context_shape, action_shape).to(device)
    # load the expert model
    if not os.path.isfile(args.expert_model_path):
        raise FileNotFoundError(f"Expert model not found at {args.expert_model_path}")
    expert_state = torch.load(args.expert_model_path, map_location=device)
    expert_policy.load_state_dict(expert_state)
    expert_policy.eval()
    # freeze expert parameters
    for param in expert_policy.parameters():
        param.requires_grad = False

    # initialize adapter policy and copy expert's encoders/decision layers
    adapter_policy = AdapterPolicy(envs, args.history_length, obs_shape, action_shape).to(device)
    copy_modules(expert_policy, adapter_policy, ["fc1", "fc2", "fc3", "fc_mean", "fc_logstd"])
    # freeze all adapter parameters except the history adapter
    for param in adapter_policy.parameters():
        param.requires_grad = False
    for param in adapter_policy.adapter.parameters():
        param.requires_grad = True

    adapter_optimizer = optim.Adam(list(adapter_policy.adapter.parameters()), lr=args.adapter_lr)

    # replay buffer storing obs, context, history
    rb = ReplayBufferDict(
        args.buffer_size,
        obs_shape=obs_shape,
        context_shape=context_shape,
        history_shape=history_shape,
        action_shape=action_shape,
        device=device,
    )

    start_time = time.time()
    for global_step in range(args.total_timesteps):
        # periodic evaluation of adapter policy
        if global_step % args.eval_freq == 0:
            names_adapter = [f"adapter_{name}" for name in eval_envs_names]
            eval_agent(adapter_policy, eval_envs, names_adapter, global_step, device, agent_type="adapter")

        # collect actions using the adapter policy
        actions, _, _ = adapter_policy.get_action(
            torch.tensor(obs, dtype=torch.float32, device=device),
            torch.tensor(history, dtype=torch.float32, device=device),
        )
        actions = actions.detach().cpu().numpy()

        next_obs_dict, rewards, terminations, truncations, infos = envs.step(actions)

        # store transitions in buffer
        real_next_obs_dict = next_obs_dict.copy()
        if "final_observation" in infos:
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs_dict[idx] = infos["final_observation"][idx]

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

        # update current observation, context, history
        obs_dict = next_obs_dict
        obs = obs_dict["obs"]
        context = obs_dict["context"]
        history = obs_dict["history"]

        # only update adapter after learning starts
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            batch_ctx = data["context"]
            batch_hist = data["history"]


            with torch.no_grad():
                z = expert_policy.encoder(batch_ctx)    # compute expert latent context encoding
            z_hat = adapter_policy.adapter(batch_hist)  # compute adapter latent history encoding
            adapter_loss = F.mse_loss(z_hat, z.detach())
            adapter_optimizer.zero_grad()
            adapter_loss.backward()
            adapter_optimizer.step()

            # logging
            if global_step % args.log_freq == 0:
                log_dict = {
                    "losses/adapter_loss": adapter_loss.item(),
                    "throughput/SPS": int(global_step / (time.time() - start_time)),
                }
                wandb.log(log_dict, step=global_step)
                if args.capture_video:
                    uploaded_videos = log_videos(global_step, uploaded_videos, video_dirs)

    envs.close()
    eval_envs.close()

    # save adapter policy
    os.makedirs("models", exist_ok=True)
    adapter_path = os.path.join("models", f"{run_name}_adapter.pth")
    torch.save(adapter_policy.state_dict(), adapter_path)
    print(f"Saved adapter policy to {adapter_path}")
