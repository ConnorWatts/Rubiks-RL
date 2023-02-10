from typing import Dict

def get_env_args(rubiks_dim: int, num_moves_reset: int) -> Dict[str, int]:

    """
    Get arguments for making cube environment

    :param rubiks_dim: int denoting the dimension of the rubiks cube

    :param num_moves_reset: int denoting the number of moves to reset 
        the cube to and random point
    
    """

    return {"dim": rubiks_dim, "num_moves_reset": num_moves_reset}