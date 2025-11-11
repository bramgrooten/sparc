import gymnasium as gym
import time

# Importing the custom environments to register them in gym
from environments import wind_halfcheetah, wind_hopper, wind_walker2d, wind_ant


if __name__ == '__main__':
    '''
    To visualize the custom wind environments, you can run this script with:
    >>> python visualize_envs.py
    Tip: press TAB to switch camera views.
    '''

    # Choose the environment you want to visualize:
    # env_name = 'WindHalfCheetah-v5'
    # env_name = 'WindHopper-v5'
    # env_name = 'WindWalker2d-v5'
    env_name = 'WindAnt-v5'

    # Set the wind speeds you want to visualize:
    # HalfCheetah
    # wind_x = -2.5
    # wind_z = 5

    # Ant
    wind_x = -0.1
    wind_z = 0.1

    env_kwargs = {
        'wind_speed_interval_x': (wind_x, wind_x),
        'wind_speed_interval_z': (wind_z, wind_z),
        'render_mode': 'human',
        'width': 1200,
        'height': 800,
    }
    if 'Hopper' in env_name or 'Walker2d' in env_name:
        # useful to view Hopper/Walker2d longer
        env_kwargs['terminate_when_unhealthy'] = False

    env = gym.make(env_name, **env_kwargs)

    obs, info = env.reset()
    print(f'Wind speed: {info["wind_speed"]}')

    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        time.sleep(0.04)

    env.close()
