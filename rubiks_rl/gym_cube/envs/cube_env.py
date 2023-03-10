from typing import Dict, List, Tuple, Union
import random
import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class CubeEnv(gym.Env):

    id = 'cube-v0'

    def __init__(self, dim: int, num_moves_reset: int):

        """
        Constructs the environment for the Rubiks Cube.
        
        :param dim: int denoting the dimension of the rubiks cube
        :param num_moves_reset: int denoting the number of moves to randomly scramble
            the rubik's cube
        
        """

        self.dim = dim
        self.cube = self._get_solved_cube(self.dim)
        self.num_moves_reset = num_moves_reset
        self.num_unique_moves = self.dim * 6
        # should total step counts be in here

        self.done = False
        self.reward = 0.0
        self.rewards = []
        self.cum_reward = 0.0

        ## core moves ##
        self.core_moves = {0:{"Side":1,"Orientation":"Vertical","Direction":"Up"},\
            1:{"Side":1,"Orientation":"Vertical","Direction":"Down"},\
                2:{"Side":1,"Orientation":"Horizontal","Direction":"Left"},\
                    3:{"Side":1,"Orientation":"Horizontal","Direction":"Right"},\
                        4:{"Side":2,"Orientation":"Vertical","Direction":"Up"},\
                            5:{"Side":2,"Orientation":"Vertical","Direction":"Down"}}

        # gym parameters #
        self.action_space = spaces.Discrete(self.num_unique_moves)
        self.observation_space = spaces.MultiDiscrete([[5,5,5] for _ in range(6)])

    def step(self, action:int)-> Tuple[np.ndarray, float, bool, bool, dict]:

        """"
        Given an action perform one step on the environment( One turn of cube)

        :param action: int index of move/turn 

        :return observation : np.ndarray denoting state of cube after move
        :return reward : float denoting the reward given from the action
        :return done : bool denoting whether the episode is done and the cube is solved

        """
        self._move(action)
        self.reward = self._get_reward()
        self.cum_reward += self.reward
        self.done = self._is_solved()

        if self.done:
            self.rewards.append(self.cum_reward)
            self.cum_reward = 0.0

        self.observation = self._get_observation()
        return self.observation, self.reward, self.done, self.done,{}

    def reset(self) -> Tuple[np.ndarray]:
        """
        Reset the current cube to a random cube. Additionally resets done/rewards
        flag/count. 

        :return cube: np.ndarray denoting the reset cube
        """
        self.reward = 0.0
        self.done = False
        self.cube = self._get_solved_cube(self.dim)
        for action in random.sample(range(0, self.num_unique_moves), self.num_moves_reset):
            self._move(action)
        return self.cube

    def render(self):
        ...

    ######## move methods ########

    def horizontal_left(self,row,side) -> None:

        cube = self.cube.copy()
        cube[0][row], cube[1][row], cube[2][row], cube[3][row] = \
             self.cube[1][row], self.cube[2][row],self.cube[3][row],self.cube[0][row]
        self.cube = cube  

        if row == 0:
            self.cube[4] = np.rot90(self.cube[4],axes=(1,0))
        elif row == self.dim - 1:
            self.cube[5] = np.rot90(self.cube[5])


    def horizontal_right(self,row,side) -> None:

        cube = self.cube.copy()
        cube[0][row], cube[1][row], cube[2][row], cube[3][row] = \
             self.cube[3][row], self.cube[0][row],self.cube[1][row],self.cube[2][row]
        self.cube = cube  

        if row == 0:
            self.cube[4] = np.rot90(self.cube[4])
        elif row == self.dim - 1:
            self.cube[5] = np.rot90(self.cube[5],axes=(1,0))

    def vertical_up(self,col,side) -> None:

        if side == 1:
            cube = self.cube.copy()
            cube[0][:,col], cube[4][:,col], cube[2][:,(self.dim - 1) - col][:], cube[5][:,col][:] = \
                self.cube[5][:,col], self.cube[0][:,col], self.cube[4][:,col][::-1], self.cube[2][:,(self.dim - 1) - col][::-1]
            self.cube = cube 
            if col == 0:
                self.cube[3] = np.rot90(self.cube[3])
            elif col == self.dim - 1:
                self.cube[1] = np.rot90(self.cube[1],axes=(1,0))
        elif side == 2:
            cube = self.cube.copy()
            cube[1][:,col][:], cube[4][self.dim -1 - col], cube[3][:,(self.dim - 1) - col], cube[5][col] = \
                self.cube[5][col][::-1], self.cube[1][:,col], self.cube[4][self.dim - 1 - col], self.cube[3][:,(self.dim - 1) - col]
            self.cube = cube 
            if col == 0:
                self.cube[0] = np.rot90(self.cube[0])
            elif col == self.dim - 1:
                self.cube[2] = np.rot90(self.cube[2],axes=(1,0))

    def vertical_down(self,col,side) -> None:

        cube = self.cube.copy()
            
        if side == 1:
            cube[0][:,col], cube[4][:,col][:], cube[2][:,(self.dim - 1) - col][:], cube[5][:,col] = \
                self.cube[4][:,col], self.cube[2][:,(self.dim - 1) - col][::-1], self.cube[5][:,col][::-1],self.cube[0][:,col]
            self.cube = cube 
            if col == 0:
                self.cube[3] = np.rot90(self.cube[3],axes=(1,0))
            elif col == self.dim - 1:
                self.cube[1] = np.rot90(self.cube[1])
        elif side == 2:
            cube[1][:,col], cube[4][self.dim -1 - col], cube[3][:,(self.dim - 1) - col], cube[5][col] = \
                self.cube[4][self.dim - 1 - col], self.cube[3][:,(self.dim - 1) - col],self.cube[5][col], self.cube[1][:,col]
            self.cube = cube 
            if col == 0:
                self.cube[0] = np.rot90(self.cube[0],axes=(1,0))
            elif col == self.dim - 1:
                self.cube[2] = np.rot90(self.cube[2])
                
    
    def _move(self, action:int) -> None:

        """
        Perform one move on the current cube.

        :param action: int denoting index of move to perform

        """

        row_col = action%self.dim
        core_move_idx = action//self.dim

        core_move = self.core_moves[core_move_idx]

        if core_move["Orientation"] == "Horizontal" and core_move["Direction"] == "Left":
            self.horizontal_left(row_col,core_move["Side"])
        elif core_move["Orientation"] == "Horizontal" and core_move["Direction"] == "Right":
            self.horizontal_right(row_col,core_move["Side"])
        elif core_move["Orientation"] == "Vertical" and core_move["Direction"] == "Up":
            self.vertical_up(row_col,core_move["Side"])
        elif core_move["Orientation"] == "Vertical" and core_move["Direction"] == "Down":
            self.vertical_down(row_col,core_move["Side"])
        else:
            print("Move not found")


    ######## general methods ########

    def _is_solved(self) -> bool:
       
        for side in self.cube:
            if not np.all(side == side[0][0]):
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

    def _get_reward(self) -> float:

        # basic reward system at first
        if self._is_solved():
            return 0
        else:
            return -1


if __name__ == "__main__":

    # for testing

    # to do CHECK ALL MOVES
    # check output of step
    cube = CubeEnv()
    cube._move(10)
    print(cube._get_observation())
    print(cube._is_solved())
        

  

