import numpy as np

class Cube():
    #testy
    def __init__(self,create_cube,) -> None:

        if create_cube:
            self.cube = self.get_solved_cube()

        print(self.cube)
        self.horizontal_twist_test()

        self.colour_map = {"R":1,"O":2,"Y":3,"W":4,"B":5,"G":6} 

        pass

    def get_solved_cube(self) -> list[list[int]]:

        #look at smarter way
        #maybe numpy 

        cube = np.array([[[1,1,1],[1,1,1],[1,1,1]],\
                [[3,3,3],[3,3,3],[3,3,3]],\
                [[2,2,2],[2,2,2],[2,2,2]],\
                [[4,4,4],[4,4,4],[4,4,4]],\
                [[5,5,5],[5,5,5],[5,5,5]],\
                [[6,6,6],[6,6,6],[6,6,6]]])


        return cube

    def horizontal_twist_test(self) -> None:
        
        #pass this in
        row = 1

        #this doesnt work cube[0] is now cube[1] so cube[3] = cube[0]
        self.cube[0][row], self.cube[1][row],self.cube[2][row],self.cube[3][row] = \
             self.cube[1][row], self.cube[2][row],self.cube[3][row],self.cube[0][row] 
             
        print(self.cube)
        pass

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

