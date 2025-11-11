import gymnasium as gym


def get_train_wind_speeds(env_id, args):
    if args.wind:
        if args.wind_x_z is not None:
            if len(args.wind_x_z) != 4:
                raise ValueError("wind_x_z must contain four values: x_min, x_max, z_min, z_max")
            return {'x': (args.wind_x_z[0], args.wind_x_z[1]), 'z': (args.wind_x_z[2], args.wind_x_z[3])}
        else:
            # default wind speeds for training
            if "HalfCheetah" in env_id:
                return {'x': (-2.5, 2.5), 'z': (-5, 5)}
            elif "Hopper" in env_id:
                return {'x': (-10, 10), 'z': (-2.5, 2.5)}
            elif "Walker2d" in env_id:
                return {'x': (-10, 10), 'z': (-2.5, 2.5)}
            elif "Ant" in env_id:
                return {'x': (-0.1, 0.1), 'z': (-0.1, 0.1)}
            else:
                raise ValueError(f"Unknown env_id: {env_id}")
    else:
        return {'x': (0, 0), 'z': (0, 0)}


def get_eval_wind_speeds(env_id):
    if "Wind" in env_id:
        if "HalfCheetah" in env_id:
            return {
                "no_wind": {'x': 0.0, 'z': 0.0},
                "wind-1_25": {'x': -1.25, 'z': 2.5},
                "wind2_5": {'x': 2.5, 'z': 5.0},
                "wind-5": {'x': -5.0, 'z': -10.0},
            }
        elif "Hopper" in env_id or "Walker2d" in env_id:
            return {
                "no_wind": {'x': 0.0, 'z': 0.0},
                "wind-5": {'x': -5.0, 'z': 1.25},
                "wind10": {'x': 10.0, 'z': 2.5},
                "wind-20": {'x': -20.0, 'z': -5.0},
            }
        elif "Ant" in env_id:
            return {
                "no_wind": {'x': 0.0, 'z': 0.0},
                "wind-0_05": {'x': -0.05, 'z': 0.05},
                "wind0_1": {'x': 0.1, 'z': 0.1},
                "wind-0_2": {'x': -0.2, 'z': -0.2},
            }
        else:
            return None
    else:
        return None


def make_env(env_id, seed, idx, run_name, args):
    def thunk():
        env_kwargs = {}

        if 'Wind' in env_id:
            winds = get_train_wind_speeds(env_id, args)
            print(f"Setting wind speeds for {env_id}: {winds}")
            env_kwargs.update({
                "wind_speed_interval_x": winds['x'],
                "wind_speed_interval_z": winds['z'],
                "history_length": args.history_length,
                "context_in_obs": args.context_in_obs,
                "history_in_obs": args.history_in_obs,
            })
            if 'WindHopper' in env_id or 'WindAnt' in env_id:
                env_kwargs["terminate_when_unhealthy"] = False

        if args.capture_video and idx == 0:
            env_kwargs["render_mode"] = "rgb_array"

        env = gym.make(env_id, **env_kwargs)

        if args.capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}/train")

        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


def make_wind_eval_env(env_id, seed, capture_video, run_name, wind_name, wind_setting, args):
    def thunk():
        env_kwargs = {
            "wind_speed_interval_x": (wind_setting['x'], wind_setting['x']),
            "wind_speed_interval_z": (wind_setting['z'], wind_setting['z']),
            "history_length": args.history_length,
            "context_in_obs": args.context_in_obs,
            "history_in_obs": args.history_in_obs,
        }
        if capture_video:
            env_kwargs["render_mode"] = "rgb_array"
        if "WindHopper" in env_id or "WindAnt" in env_id:
            env_kwargs["terminate_when_unhealthy"] = False

        env = gym.make(env_id, **env_kwargs)

        if capture_video:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}/eval_{wind_name}")

        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def make_eval_envs(env_id, seed, capture_video, run_name, args):
    winds = get_eval_wind_speeds(env_id)
    if winds is None:
        # just make a single env
        eval_envs = gym.vector.SyncVectorEnv([make_env(env_id, seed, 0, run_name + "_eval", args)])
        eval_envs_names = ["eval"]
    else:
        # make multiple envs
        eval_envs_list = []
        eval_envs_names = []
        seed_i = seed
        for wind_name, wind_setting in winds.items():
            seed_i += 1
            eval_envs_list.append(make_wind_eval_env(env_id, seed_i, capture_video, run_name, wind_name, wind_setting, args))
            eval_envs_names.append(wind_name)
        eval_envs = gym.vector.SyncVectorEnv(eval_envs_list)
    return eval_envs, eval_envs_names
