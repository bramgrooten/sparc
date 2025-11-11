import gymnasium as gym
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv
from gymnasium import spaces
from typing import Tuple, Dict, Any
import mujoco
import numpy as np


class WindHalfCheetahEnv(HalfCheetahEnv):
    def __init__(
            self,
            wind_speed_interval_x: Tuple[float, float] = (0.0, 0.0),
            wind_speed_interval_z: Tuple[float, float] = (0.0, 0.0),
            resample_wind_speed_prob: float = 0.0,
            context_in_obs: bool = False,
            history_in_obs: bool = False,
            history_length: int = 50,
            **kwargs
    ):
        """
        A custom HalfCheetah environment with adjustable wind as context.
        HalfCheetah walks in the x-direction and the wind can be adjusted in the x and z directions.

        Args:
            wind_speed_interval_x (Tuple[float, float]): Interval from which to sample the x-component of the wind speed.
                Default is (0.0, 0.0), meaning no wind in the x-direction.
            wind_speed_interval_z (Tuple[float, float]): Interval from which to sample the z-component of the wind speed.
                Default is (0.0, 0.0), meaning no wind in the z-direction.
            resample_wind_speed_prob (float): Probability of resampling the wind speed at each step.
                Must be between 0.0 and 1.0. Default is 0.0, meaning the wind speed is constant throughout an episode.
            context_in_obs (bool): Whether to include the wind speed in the observation. Default is False.
            history_in_obs (bool): Whether to include the history in the observation. Default is False.
            history_length (int): Number of previous obs,action pairs to include in the history. Default is 50.
            **kwargs: Additional keyword arguments passed to the base HalfCheetahEnv.
        """
        super().__init__(**kwargs)
        assert wind_speed_interval_x[1] >= wind_speed_interval_x[0], "Interval should be (low, high)"
        assert wind_speed_interval_z[1] >= wind_speed_interval_z[0], "Interval should be (low, high)"
        self._wind_speed_interval_x = wind_speed_interval_x
        self._wind_speed_interval_z = wind_speed_interval_z
        self._resample_wind_speed_prob = resample_wind_speed_prob
        self.cur_wind_speed = self._sample_env_context()["wind_speed"]
        self._context_in_obs = context_in_obs
        self._history_in_obs = history_in_obs

        original_obs_size = self.observation_space.shape[0]
        original_obs_action_size = original_obs_size + self.action_space.shape[0]
        context_size = 2  # current wind speed in x and z directions
        history_shape = (history_length, original_obs_action_size)

        if context_in_obs and history_in_obs:  # Add context and history to observation
            self.observation_space = spaces.Dict({
                "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(original_obs_size,), dtype=np.float64),
                "context": spaces.Box(low=-np.inf, high=np.inf, shape=(context_size,), dtype=np.float64),
                "history": spaces.Box(low=-np.inf, high=np.inf, shape=history_shape, dtype=np.float64),
            })
            self.cur_history = np.zeros(history_shape, dtype=np.float64)
            self.history_shape = history_shape
        elif history_in_obs:  # Add history to observation
            self.observation_space = spaces.Dict({
                "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(original_obs_size,), dtype=np.float64),
                "history": spaces.Box(low=-np.inf, high=np.inf, shape=history_shape, dtype=np.float64),
            })
            self.cur_history = np.zeros(history_shape, dtype=np.float64)
            self.history_shape = history_shape
        elif context_in_obs:  # Add context to observation
            self.observation_space = spaces.Dict({
                "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(original_obs_size,), dtype=np.float64),
                "context": spaces.Box(low=-np.inf, high=np.inf, shape=(context_size,), dtype=np.float64),
            })

    def _sample_env_context(self):
        wind_speed_x = self.np_random.uniform(*self._wind_speed_interval_x)
        wind_speed_z = self.np_random.uniform(*self._wind_speed_interval_z)
        return {
            "wind_speed": (wind_speed_x, wind_speed_z)
        }

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        self.cur_wind_speed = self._sample_env_context()["wind_speed"]
        # Reset applied wind forces (will be applied in step function)
        self.data.xfrc_applied[:] = 0

        observation = self._get_obs()

        if not self._context_in_obs and not self._history_in_obs:
            return observation
        else:
            observation = {"obs": observation}
            if self._context_in_obs:
                observation["context"] = np.array(self.cur_wind_speed)
            if self._history_in_obs:
                self.cur_history = np.zeros(self.history_shape, dtype=np.float64)
                observation["history"] = self.cur_history
            return observation

    def _get_reset_info(self) -> Dict[str, Any]:
        """Function that generates the `info` that is returned during a `reset()`."""
        info = super()._get_reset_info()
        info["wind_speed"] = self.cur_wind_speed
        return info

    def _step_mujoco_simulation(self, ctrl, n_frames):
        # Override this method to apply wind forces during simulation
        self.data.ctrl[:] = ctrl

        wind_force_x, wind_force_z = self.cur_wind_speed
        wind_force = np.array([wind_force_x, 0, wind_force_z, 0, 0, 0])

        for _ in range(n_frames):
            # Apply wind force to all body parts of the robot
            self.data.xfrc_applied[:] = wind_force
            # Step the simulation
            mujoco.mj_step(self.model, self.data)
            # Reset applied forces to zero after each step, to avoid accumulating forces
            self.data.xfrc_applied[:] = 0

        # Compute force-related quantities
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def step(self, action):
        # Possibly resample wind speed
        if self.np_random.uniform(0, 1) < self._resample_wind_speed_prob:
            self.cur_wind_speed = self._sample_env_context()["wind_speed"]

        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "wind_speed": self.cur_wind_speed,
            **reward_info,
        }
        if self.render_mode == "human":
            self.render()

        if not self._context_in_obs and not self._history_in_obs:
            # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            return observation, reward, False, False, info
        else:
            observation = {"obs": observation}
            if self._context_in_obs:
                observation["context"] = np.array(self.cur_wind_speed)
            if self._history_in_obs:
                self.cur_history[:-1] = self.cur_history[1:]
                self.cur_history[-1] = np.concatenate((observation["obs"], action))
                observation["history"] = self.cur_history
            return observation, reward, False, False, info


gym.register(
    id="WindHalfCheetah-v5",
    entry_point="environments.wind_halfcheetah:WindHalfCheetahEnv",
)
