from gym.envs.registration import register

from gym_cube.envs.cube_env import CubeEnv

register(
    id=CubeEnv.id,
    entry_point='gym_cube.envs:CubeEnv',
    max_episode_steps=100000,
    nondeterministic=False
)

