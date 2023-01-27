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
        # should total step counts be in here

        self.done = False
        self.reward = 0.0

        # check indexing is correct
        self.core_moves = {0:{"Side":1,"Orientation":"Vertical","Direction":"Up"},\
            1:{"Side":1,"Orientation":"Vertical","Direction":"Down"},\
                2:{"Side":1,"Orientation":"Horizontal","Direction":"Left"},\
                    3:{"Side":1,"Orientation":"Horizontal","Direction":"Right"},\
                        4:{"Side":2,"Orientation":"Vertical","Direction":"Up"},\
                            5:{"Side":2,"Orientation":"Vertical","Direction":"Down"}}

        # get this from input
        self.action_space = spaces.Discrete(self.num_unique_moves)

        # would this be true for impossible combinations
        self.observation_space = spaces.MultiDiscrete([[5,5,5] for _ in range(6)])

    def step(self, action:int)-> Tuple[np.ndarray, float, bool, dict]:

        #if self.done:
            #self.reset()
            #self.observation = self._get_observation()
            #return self.observation, self.reward, self.done, {}

        self._move(action)
        self.reward = self._get_reward()
        self.observation = self._get_observation()
        return self.observation, self.reward, self.done, {}

    def reset(self):

        self.reward = 0.0
        self.done = False
        self.cube = self._get_solved_cube(self.dim)
        for action in random.sample(range(0, self.num_unique_moves), self.num_moves_reset):
            self._move(action)
        return self.cube

    def render(self, mode='human', close=False):
        ...

    ### move methods ###

    def horizontal_left(self,row,side) -> None:

        # this doesnt feel like the best way to do it (try in place again)
        cube = self.cube.copy()
        cube[0][row], cube[1][row], cube[2][row], cube[3][row] = \
             self.cube[1][row], self.cube[2][row],self.cube[3][row],self.cube[0][row]
        self.cube = cube  

        if row == 0:
            self.cube[4] = np.rot90(self.cube[4],axes=(1,0))
        elif row == self.dim - 1:
            self.cube[5] = np.rot90(self.cube[5])


    def horizontal_right(self,row,side) -> None:

        # this doesnt feel like the best way to do it (try in place again)
        cube = self.cube.copy()
        cube[0][row], cube[1][row], cube[2][row], cube[3][row] = \
             self.cube[3][row], self.cube[0][row],self.cube[1][row],self.cube[2][row]
        self.cube = cube  

        if row == 0:
            self.cube[4] = np.rot90(self.cube[4])
        elif row == self.dim - 1:
            self.cube[5] = np.rot90(self.cube[5],axes=(1,0))

        pass

    def vertical_up(self,col,side) -> None:

        #side will prob be most complicated here

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
            cube[1][:,col], cube[4][self.dim -1 - col], cube[3][:,(self.dim - 1) - col], cube[5][col] = \
                self.cube[5][col].reverse(), self.cube[1][:,col], self.cube[4][self.dim - 1 - col], self.cube[3][:,(self.dim - 1) - col]
            self.cube = cube 
            if col == 0:
                self.cube[0] = np.rot90(self.cube[0])
            elif col == self.dim - 1:
                self.cube[2] = np.rot90(self.cube[2],axes=(1,0))

    def vertical_down(self,col,side) -> None:

        #side will prob be most complicated here
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

        # there are N x 6 moves
        # N for each row/col
        # 6 for the "core moves"
        # so use mod and div then

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


    ### general methods ###


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
            self.done = True
            return 1
        else:
            return 0


if __name__ == "__main__":

    # for testing

    # to do CHECK ALL MOVES
    # check output of step
    cube = CubeEnv()
    cube._move(10)
    print(cube._get_observation())
    print(cube._is_solved())
        

  

