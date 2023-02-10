import argparse
from buffer import ReplayBuffer
from models.qnetwork import QNetwork
import utils
#import gymnasium as gym
import gym
from tqdm import tqdm
import gym_cube
from gym_cube import utils as gym_utils


def main(config) -> None:

    """
    Main function for loading environment and training/evaluating
    the agent.

    :param config: Dict denoting the command line inputs to be 
            used as the configuration of the experiment 
    
    """

    # create environment
    env = gym.make('cube-v0', **gym_utils.get_env_args(config['rubiks_dim'],config['num_moves_reset']))

    # set seeds
    utils.set_random_seed(env,config)

    # get algorithm - DQN/DDPG/PPO etc
    actor = get_algo(config, env)

    # run experiment
    if config['mode'] == 'Train':
        actor.train()

    elif config['mode'] == 'Test':
        actor.eval()

    else:
        raise NotImplementedError('Running mode {} not recognised.'.format(config['mode']))


def get_algo(config, env):

    """
    Retrieve Actor Learner for experiment [DQN, DDPG, PPO]

    :param config: Dict denoting the command line inputs to be 
            used as the configuration of the experiment 

    :param env: Env denoting the Gym environment for the Rubik's 
            cube

    :return Actor: Class denoting agent to learn/eval
    
    """

    if config["rl_method"] == "DQN":
        print("--Loading DQN Agent--")
        from algos.dqn import DQN
        return DQN(config, env)

    elif config["rl_method"] == "DDPG":
        # TO DO:
        ...
    elif config["rl_method"] == "PPO":
        # TO DO:
        ...

    else:
        raise NotImplementedError('RL method {} not recognised.'.format(config['rl_method']))


def get_args() -> dict:
    parser = argparse.ArgumentParser(description='Rubiks Cube RL')

    # rubiks cube parameters
    parser.add_argument("--rubiks_dim", type=int, help="Dimension of side of cube", default=3)
    parser.add_argument("--num_moves_reset", type=int, help="Number of moves to scramble the cube", default=3)

    # experiment parameters
    parser.add_argument("--rl_method", type=str, help="RL method", default="DQN")
    parser.add_argument("--mode", type=str, help="Mode of Experiment", default="Train")

    # environment parameters
    parser.add_argument('--id', default='cube-v0', help="Environment ID",type=str)

    # replay buffer parameter
    parser.add_argument("--max_buffer_size", type=int, help="Maximum size of buffer", default=75000)

    # training parameters
    parser.add_argument("--total_train_steps", type=int, help="Total number of collecting/learning steps", default=100000)
    parser.add_argument("--num_warmup_steps", type=int, help="Total number of warm up steps before learning", default=1000)
    parser.add_argument("--collect_ratio", type=int, help="Number of steps of collection per single learn", default=4)
    parser.add_argument("--batch_size", type =int, help="Batch size for training the model", default=32)
    parser.add_argument("--gamma", type=float, help="Gamma discount value", default=0.9)
    parser.add_argument("--tau", type=float, help="Tau value", default=0.9)
    parser.add_argument("--target_dqn_update_interval", type=int, help="Learning steps to update target netork",default=4)
    parser.add_argument("--seed", type=int, help="Seed of random", default= 4)
    parser.add_argument("--loss", type=str, help="Loss function", default= "mse")
    parser.add_argument("--epsilon_schedule", type=str, help="Epsilon Schedule", default= "linear")
    parser.add_argument("--start_epsilon", type=float, help="Starting epsilon value", default=0.1)
    parser.add_argument("--end_epsilon", type=float, help="Ending epsilon value", default=0.05)
    parser.add_argument("--optimizer", type=str, help="Optimizer scheme", default= "adam")
    parser.add_argument("--adam_learning_rate", type=float, help="Learn rate for ADAM optimizer",  default=1e-4)

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