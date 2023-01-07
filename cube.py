import numpy as np
from random import seed
from random import randint
import random

class Cube():
    #testy
    def __init__(self,create_cube) -> None:

        if create_cube:
            self.cube = self.get_solved_cube()

        # get from config
        self.size = 3

        self.core_moves = ["side1_vertical_up",\
                           "side1_vertical_down",\
                           "side1_horizontal_left",\
                           "side1_horizontal_right",\
                           "side2_vertical_up",\
                           "side1_vertical_down"]

        

        print(self.cube)
        self.horizontal_twist_test()

        self.colour_map = {"R":1,"Y":2,"O":3,"W":4,"B":5,"G":6} 

        pass

    def get_solved_cube(self) -> list[list[int]]:

        #look at smarter way
        #maybe numpy 

        cube = np.array([[[1,1,1],[1,1,1],[1,1,1]],\
                [[2,2,2],[2,2,2],[2,2,2]],\
                [[3,3,3],[3,3,3],[3,3,3]],\
                [[4,4,4],[4,4,4],[4,4,4]],\
                [[5,5,5],[5,5,5],[5,5,5]],\
                [[6,6,6],[6,6,6],[6,6,6]]])


        return cube

    def horizontal_twist_test(self) -> None:
        
        #pass this in
        row = 1

        # this doesnt feel like the best way to do it (try in place again)
        cube = self.cube.copy()
        cube[0][row], cube[1][row], cube[2][row], cube[3][row] = \
             self.cube[1][row], self.cube[2][row],self.cube[3][row],self.cube[0][row]
        self.cube = cube  

        if row == 0:
            self.cube[5] = np.rot90(self.cube[5])
        elif row == self.size - 1:
            self.cube[6] = np.rot90(self.cube[6])
        print(self.cube)
        pass

    def move(self,row_col,core_move) -> None:

        # rowcol
        # rotation
        # side

        pass

    def random_move(self) -> None:

        # there is 6xN different moves
        # N for the row/column

        row_col = randint(0,self.size)
        core_move = random.choice(self.core_moves)
        self.move(row_col,core_move)

    def is_valid_cube(self) -> bool:
        pass

    def get_sides(self) -> int:
        pass

    def is_solved(self) -> bool:
        pass

    def scramble(self) -> None:
        pass

    def draw(self) -> None:
        pass

