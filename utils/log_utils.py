import os
import wandb
import numpy as np
import torch


def eval_agent(actor, eval_envs, eval_envs_names, global_step, device, episodes=10, agent_type="default"):
    obs, _ = eval_envs.reset()
    num_envs = eval_envs.num_envs
    episodic_returns = [[] for _ in range(num_envs)]
    episodic_lengths = [[] for _ in range(num_envs)]
    episode_counts = np.zeros(num_envs, dtype=int)

    while (episode_counts < episodes).any():
        if isinstance(obs, dict):
            context = obs.get("context")
            history = obs.get("history")
            obs = obs["obs"]  # rename this last

        with torch.no_grad():
            if agent_type == "expert":
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device), torch.Tensor(context).to(device))
            elif agent_type == "adapter":
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device), torch.Tensor(history).to(device))
            else:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()

        obs, rewards, terminated, truncated, infos = eval_envs.step(actions)

        if "episode" in infos:
            for i, finished in enumerate(infos["episode"]['_r']):
                if finished:
                    episodic_returns[i].append(infos["episode"]['r'][i])
                    episodic_lengths[i].append(infos["episode"]['l'][i])
                    episode_counts[i] += 1

    # logging average returns for each contextual env (wind speed setting)
    for i in range(num_envs):
        env_name = eval_envs_names[i]
        avg_return = np.mean(episodic_returns[i][:episodes])  # only take the first `episodes` episodes
        avg_length = np.mean(episodic_lengths[i][:episodes])
        log_dict = {
            f"eval_return/{env_name}": avg_return,
            f"eval_length/{env_name}": avg_length,
        }
        wandb.log(log_dict, step=global_step)
        print(f"[Eval] Step{global_step} {env_name}: Avg Return = {avg_return:.2f}")
        # print(f"[Eval] Step{global_step} {env_name}: Avg Length = {avg_length:.2f}")


def log_videos(step, uploaded_videos, video_dirs):
    for video_dir in video_dirs:
        if os.path.exists(video_dir):
            video_files = os.listdir(video_dir)
            for vid in video_files:
                full_video_path = os.path.join(video_dir, vid)
                if full_video_path not in uploaded_videos:
                    wandb.log({
                        video_dir: wandb.Video(full_video_path, caption=vid)
                    }, step=step)
                    uploaded_videos.add(full_video_path)
    return uploaded_videos
