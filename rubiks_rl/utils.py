from schedules import LinearSchedule, ConstantSchedule, ExponentialSchedule
from torch import nn
import torch
import numpy as np
import random

def get_schedule(config):
    """
    Gets the schedule for epsilon value in e-greedy regime
    """

    if config['epsilon_schedule'] == 'linear':
        return LinearSchedule(config)
    elif config['epsilon_schedule'] == 'constant':
        return ConstantSchedule(config)
    elif config['epsilon_schedule'] == 'exponential':
        return ExponentialSchedule(config)
    
def get_optimizer(network,config):
    """
    Gets the optimizer for the training of the agent
    """

    if config['optimizer'] == 'adam':
        from torch.optim import Adam
        return Adam(network.parameters(), lr=config['adam_learning_rate'])


def set_random_seed(env,config):

    """
    Sets the random seed for the experiment
    """

    #from https://github.com/gordicaleksa/pytorch-learn-reinforcement-lear \
    # ning/blob/26dd439e73bb804b2065969caa5fa5429becfdd5/utils/utils.py

    seed = config['seed']
    if seed is not None:
        torch.manual_seed(seed)  # PyTorch
        np.random.seed(seed)  # NumPy
        random.seed(seed)  # Python
        env.action_space.seed(seed)  # probably redundant but I found an article where somebody had a problem with this
        #env.seed(seed)  # OpenAI gym

        # todo: AB test impact on FPS metric
        # Deterministic operations for CuDNN, it may impact performances
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    ...

def get_loss(config):
    """
    Gets the loss function for the experiment
    """

    if config['loss'] == 'mse':
        loss = nn.MSELoss()
        return loss
