from torch.utils.tensorboard import SummaryWriter
from buffer import ReplayBuffer
from models.qnetwork import QNetwork
import numpy as np
import torch
import random
import utils 
from tqdm import tqdm


class DQN:
    def __init__(self, config, env) -> None:
        super().__init__()

        """
        Constructs the DQN Class for the experiment
        
       :param config: Dict denoting the command line inputs to be 
            used as the configuration of the experiment 
        :param env: gym Env denoting Rubik's Cube environment
        
        """

        self.total_train_steps = config['total_train_steps']
        self.warmup_count = config['num_warmup_steps']
        self.collect_ratio = config['collect_ratio']
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.target_dqn_update_interval  = config['target_dqn_update_interval']
        self.tau = config['tau']

        self.env = env

        self.replay_buffer = ReplayBuffer(config)
        self.state = env.reset()
       
        self.q_net = QNetwork(config)
        self.target_q_net = QNetwork(config)

        self.writer = SummaryWriter()

        self.loss_ft = utils.get_loss(config)
        self.schedule = utils.get_schedule(config)
        self.optimizer = utils.get_optimizer(self.q_net,config)

        self.global_step_count = 0
        self.learn_step_count = 0
        self.loss = []

    def train(self) -> None:

        """
        Train the agent. First the agent will soley store samples in 
        a replay buffer before thern using the buffer to train the 
        Q-Network

        """

        for step in tqdm(range(self.total_train_steps)):

            self.collect_experience()

            if step > self.warmup_count:
                self.learn_from_experience()

            self.global_step_count += 1


    def eval(self) -> None:
        """
        Evaluate the agent a set number of steps.

        """ 
        

    def collect_experience(self) -> None:

        """
        Collect experience and store in Replay Buffer

        """

        for _ in range(self.collect_ratio):

            action = self.sample_action(self.state)
            next_state, reward, done, _ , _= self.env.step(action)
            self.replay_buffer.store_transition(self.state, next_state, action, reward, done)
            
            if done:
                self.state = self.env.reset()
                # TO DO: maybe all the agents can use the same one (utils)
                self.log_episode()
            else:
                self.state = next_state


    def learn_from_experience(self) -> None:

        """
        Use a sample from Replay Buffer to train the current Q-Network

        """

        current_states, next_states, actions, rewards, dones = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():

            #[0] as [0] is value of largest and [1] is indice
            next_states_max_q_val = self.target_q_net(next_states).max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_states_max_q_val
        
        current_q_values = self.q_net(current_states).gather(dim=1, index=torch.unsqueeze(actions,0))[0]
        loss = self.loss_ft(target_q_values, current_q_values)
        self.loss.append(loss)

        self.optimizer.zero_grad()
        loss.backward() 

        # potentially clip gradients for stability reasons

        self.optimizer.step()
        self.learn_step_count += 1

        if self.learn_step_count % self.target_dqn_update_interval == 0:
            self.update_target_q_network()
            
    def update_target_q_network(self) -> None:

        """
        Update target Q-Network parameters with current Q-Network 
        parameters

        """

        if self.tau == 1.:
                self.target_q_net.load_state_dict(self.q_net.state_dict())
        else:  
            for target_network_param, q_network_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_network_param.data.copy_(
                    self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
                )


    def sample_action(self, state: np.ndarray) -> int:

        """
        Sample an action given a state value of the environement. Initially this
        will be random while the replay buffer is filled. After a set number of
        "warm up" steps the action will then be determined by e-greedy routine

        :param state: np.ndarray denoting the current state of the environment
            (the current cube)

        :return action: int denoting the move to perform on the cube. For details 
            on the mapping from int to move see CubeEnv in gym_cube/env
        """

        if self.global_step_count < self.warmup_count:
            action = self.env.action_space.sample()  
        else:
            with torch.no_grad():
                epsilon = self.get_epsilon()
                if random.random() < epsilon:
                    action = self.env.action_space.sample()  
                else:

                    q_values = self.q_net(torch.unsqueeze(torch.LongTensor(state),0))
                    action = torch.argmax(q_values, dim=1).item()

        return action


    def get_epsilon(self):

        """
        Get epsilon value for e-greedy routine. Depending on the schedule this 
        could be determined by how far the agent is through training

        :return epsilon: float denoting the epsilon value
        """

        # TO DO: Use schedule

        return 0.1

    def log_episode(self): 

        num_episodes = len(self.env.rewards)
        self.writer.add_scalar('Rewards per episode', self.env.rewards[-1], num_episodes)

    def save_best(self):
        ...
        


if __name__ == "__main__":

    # for testing
    ...
