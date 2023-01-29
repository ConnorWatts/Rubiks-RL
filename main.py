import argparse
from buffer import ReplayBuffer
from actor import Actor
from models.qnetwork import QNetwork
import utils
#import gymnasium as gym
import gym
from tqdm import tqdm
import gym_cube

def train_dqn(config):

    env = gym.make('cube-v0')
    utils.set_random_seed(config)

    actor = Actor(config, env)

    for step in tqdm(range(config['total_steps'])):

        actor.collect_experience()

        if step > config['num_warmup_steps']:
            actor.learn_from_experience()

        actor.step_count += 1



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

    # experiment parameters
    parser.add_argument("--rl_method", type=str, help="RL method", default="DQN")

    # environment parameters
    parser.add_argument('--id', default='cube-v0', help="Environment ID",type=str)

    # replay buffer parameter
    parser.add_argument("--max_buffer_size", type=int, help="Maximum size of buffer", default=75000)

    # training parameters
    parser.add_argument("--total_steps", type=int, help="Total number of collecting/learning steps", default=100000)
    parser.add_argument("--num_warmup_steps", type=int, help="Total number of warm up steps before learning", default=1000)
    parser.add_argument("--collect_ratio", type=int, help="Number of steps of collection per single learn", default=4)
    parser.add_argument("--batch_size", type =int, help="Batch size for training the model", default=32)
    parser.add_argument("--gamma", type=float, help="Gamma discount value", default=0.9)
    parser.add_argument("--tau", type=float, help="Tau value", default=0.9)
    parser.add_argument("--target_dqn_update_interval", type=int, help="Learning steps to update target netork",default=4)

    # network parameters
    parser.add_argument("--conv_dim", type=int, help="Dimension of Conv3d layers DQN", default=[12,24])
    parser.add_argument("--emb_dim", type=int, help="Dimension of Embeddings DQN", default=4)
    parser.add_argument("--act_fnt", type=str, help="Activation function DQN", default="ReLU")




    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    return training_config


if __name__ == "__main__":
    main(get_args())