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

    replay_buffer = ReplayBuffer(config)

    dqn = QNetwork(config)
    target = QNetwork(config)

    schedule = utils.get_schedule(config)
    optimizer = utils.get_optimizer(dqn,config)

    actor = Actor(config, replay_buffer, dqn, target, env, \
        optimizer, schedule)

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

    # experiment details 
    parser.add_argument("--rl_method", type=str, help="RL method", default="DQN")

    parser.add_argument('--id', default='cube-v0', help="Environment ID",type=str)

    parser.add_argument("--total_steps", type=int, help="Total number of collecting/learning steps", default=100000)
    parser.add_argument("--num_warmup_steps", type=int, help="Total number of warm up steps before learning", default=30000)
    parser.add_argument("--collection_ratio", type=int, help="Number of steps of collection per single learn", default=4)

    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    return training_config


if __name__ == "__main__":
    main(get_args())