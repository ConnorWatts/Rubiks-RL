from typing import Dict, List, Tuple, Union
import random
import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class CubeEnv(gym.Env):

    #metadata = {'render.modes': ['human']}
    id = 'cube-v0'

    def __init__(self):

        self.dim = 3 # allow this to be variable
        self.cube = self._get_solved_cube(self.dim)
        self.num_moves_reset = 3
        self.num_unique_moves = 18

        # get this from input
        self.action_space = spaces.Discrete(self.num_unique_moves)

        # would this be true for impossible combinations
        self.observation_space = spaces.MultiDiscrete([[5,5,5] for _ in range(6)])

        ...
    def step(self, action)-> Tuple[np.ndarray, float, bool, dict]:
        ...

    def reset(self):

        self.cube = self._get_solved_cube(self.dim)
        for action in random.sample(range(0, self.num_unique_moves), self.num_moves_reset):
            self._move(action)

    def render(self, mode='human', close=False):
        ...

    

    def _is_solved(self) -> bool:

        for side in self.cube:
            if not np.all(side == side[0]):
                return False
        return True

    @staticmethod
    def _get_solved_cube(rubiks_dim:int) -> np.ndarray:

        # find smarter way to create with variable dim

        cube = np.array([[[1,1,1],[1,1,1],[1,1,1]],\
                [[2,2,2],[2,2,2],[2,2,2]],\
                [[3,3,3],[3,3,3],[3,3,3]],\
                [[4,4,4],[4,4,4],[4,4,4]],\
                [[5,5,5],[5,5,5],[5,5,5]],\
                [[6,6,6],[6,6,6],[6,6,6]]])

        return cube

    def _get_observation(self) -> np.ndarray:

        return self.cube

    def _get_reward(self,action:int) -> float:

        # basic reward system at first
        return 1 if self._is_solved() else 0
        

  

