import numpy as np
from random import seed
from random import randint
import random

class Cube():
    #testy
    def __init__(self,config,create_cube) -> None:

        if create_cube:
            self.cube = self.get_solved_cube()

        # get from config
        self.rubiks_dim = config["rubiks_dim"]
        self.num_moves_scramble = config["num_moves_scramble"]

        self.core_moves = {1:{"Side":1,"Orientation":"Vertical","Direction":"Up"},\
            2:{"Side":1,"Orientation":"Vertical","Direction":"Down"},\
                3:{"Side":1,"Orientation":"Horizontal","Direction":"Left"},\
                    4:{"Side":1,"Orientation":"Horizontal","Direction":"Right"},\
                        5:{"Side":2,"Orientation":"Vertical","Direction":"Up"},\
                            6:{"Side":2,"Orientation":"Vertical","Direction":"Down"}}

        print(self.cube)
        self.scramble()


        self.colour_map = {"R":1,"Y":2,"O":3,"W":4,"B":5,"G":6} 
        online = {"W":1,"B":2,"Y":3,"G":4,"O":5,"R":6} 

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

    def horizontal_left(self,row,side) -> None:

        # this doesnt feel like the best way to do it (try in place again)
        cube = self.cube.copy()
        cube[0][row], cube[1][row], cube[2][row], cube[3][row] = \
             self.cube[1][row], self.cube[2][row],self.cube[3][row],self.cube[0][row]
        self.cube = cube  

        if row == 0:
            self.cube[4] = np.rot90(self.cube[4],axes=(1,0))
        elif row == self.rubiks_dim - 1:
            self.cube[5] = np.rot90(self.cube[5])

        pass

    def horizontal_right(self,row,side) -> None:

        # this doesnt feel like the best way to do it (try in place again)
        cube = self.cube.copy()
        cube[0][row], cube[1][row], cube[2][row], cube[3][row] = \
             self.cube[3][row], self.cube[0][row],self.cube[1][row],self.cube[2][row]
        self.cube = cube  

        if row == 0:
            self.cube[4] = np.rot90(self.cube[4])
        elif row == self.rubiks_dim - 1:
            self.cube[5] = np.rot90(self.cube[5],axes=(1,0))

        pass

    def vertical_up(self,col,side) -> None:

        #side will prob be most complicated here

        if side == 1:
            cube = self.cube.copy()
            cube[0][:,col], cube[4][:,col], cube[2][:,(self.rubiks_dim - 1) - col], cube[5][:,col] = \
                self.cube[5][:,col], self.cube[0][:,col], self.cube[4][:,col].reverse(), self.cube[2][:,(self.rubiks_dim - 1) - col].reverse()
            self.cube = cube 
            if col == 0:
                self.cube[3] = np.rot90(self.cube[3])
            elif col == self.rubiks_dim - 1:
                self.cube[1] = np.rot90(self.cube[1],axes=(1,0))
        elif side == 2:
            cube = self.cube.copy()
            cube[1][:,col], cube[4][self.rubiks_dim -1 - col], cube[3][:,(self.rubiks_dim - 1) - col], cube[5][col] = \
                self.cube[5][col].reverse(), self.cube[1][:,col], self.cube[4][self.rubiks_dim - 1 - col], self.cube[3][:,(self.rubiks_dim - 1) - col]
            self.cube = cube 
            if col == 0:
                self.cube[0] = np.rot90(self.cube[0])
            elif col == self.rubiks_dim - 1:
                self.cube[2] = np.rot90(self.cube[2],axes=(1,0))

    def vertical_down(self,col,side) -> None:

        #side will prob be most complicated here
        cube = self.cube.copy()
            
        if side == 1:
            cube[0][:,col], cube[4][:,col], cube[2][:,(self.rubiks_dim - 1) - col], cube[5][:,col] = \
                self.cube[4][:,col], self.cube[2][:,(self.rubiks_dim - 1) - col].reverse(), self.cube[5][:,col].reverse(),self.cube[0][:,col]
            self.cube = cube 
            if col == 0:
                self.cube[3] = np.rot90(self.cube[3],axes=(1,0))
            elif col == self.rubiks_dim - 1:
                self.cube[1] = np.rot90(self.cube[1])
        elif side == 2:
            cube[1][:,col], cube[4][self.rubiks_dim -1 - col], cube[3][:,(self.rubiks_dim - 1) - col], cube[5][col] = \
                self.cube[4][self.rubiks_dim - 1 - col], self.cube[3][:,(self.rubiks_dim - 1) - col],self.cube[5][col], self.cube[1][:,col]
            self.cube = cube 
            if col == 0:
                self.cube[0] = np.rot90(self.cube[0],axes=(1,0))
            elif col == self.rubiks_dim - 1:
                self.cube[2] = np.rot90(self.cube[2])



    def move(self,row_col,core_move) -> None:

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
        print(self.cube)

        pass

    def random_move(self) -> None:

        # there is 6xN different moves
        # N for the row/column

        row_col = randint(0,self.rubiks_dim-1)
        core_move = random.choice(list(self.core_moves.values()))
        # format better
        print("Random move: RowCol " + str(row_col) + " Core Move " + str(core_move))
        self.move(row_col,core_move)

    def is_valid_cube(self) -> bool:
        pass

    def get_sides(self) -> list[list[int]]:
        pass

    def is_solved(self) -> bool:
        pass

    def scramble(self) -> None:

        # perform a set number of random moves to 
        # randomise the cube 

        for _ in range(self.num_moves_scramble):
            self.random_move()

        pass

    def draw(self) -> None:
        pass

