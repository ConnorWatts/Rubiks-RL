import argparse
from cube import Cube
from buffer import ReplayBuffer
from actor import Actor
from dqn import DQN
import utils

def train_dqn(config):

    cube = Cube(config,create_cube=True)

    utils.set_random_seed(config)
    schedule = utils.get_schedule(config)

    replay_buffer = ReplayBuffer(config)

    dqn = DQN(config)
    target = DQN(config)

    actor = Actor(config, replay_buffer, dqn, target)





    pass

def train_actor_critic(config):
    # to do 
    pass

def main(params):
    
    if params["rl_method"] == "DQN":
        train_dqn(params)
    elif params["rl_method"] == "Actor Critic":
        train_actor_critic(params)
    pass

def get_args() -> dict:
    parser = argparse.ArgumentParser(description='Rubiks Cube RL')

    # rubiks cube parameters
    parser.add_argument("--rubiks_dim", type=int, help="Dimension of side of cube", default=3)
    parser.add_argument("--num_moves_scramble", type=int, help="Number of moves to scramble the cube", default=4)

    # experiment details 
    parser.add_argument("--rl_method", type=str, help="RL method", default="DQN")

    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    return training_config


if __name__ == "__main__":
    main(get_args())