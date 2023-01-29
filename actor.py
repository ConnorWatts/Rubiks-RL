from torch.utils.tensorboard import SummaryWriter
from buffer import ReplayBuffer
from models.qnetwork import QNetwork
import numpy as np
import torch
import random
import utils 

class Actor:
    def __init__(self, config, env) -> None:
        super().__init__()

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

        self.step_count = 0
        self.learn_step_count = 0
        self.loss = []

    def collect_experience(self) -> None:

        for _ in range(self.collect_ratio):

            action = self.sample_action(self.state)
            next_state, reward, done, _ , _= self.env.step(action)
            self.replay_buffer.store_transition(self.state, next_state, action, reward, done)
            
            if done:
                self.state = self.env.reset()
                # log episode result
            else:
                self.state = next_state


    def learn_from_experience(self) -> None:

        current_states, next_states, actions, rewards, dones = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():

            #[0] as [0] is value of largest and [1] is indice
            next_states_max_q_val = self.target_q_net(next_states).max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_states_max_q_val
        
        current_q_values = self.q_network(current_states).gather(dim=1, index=actions)
        loss = self.loss_ft(target_q_values, current_q_values)
        self.loss.append(loss)

        self.optimizer.zero_grad()
        loss.backward() 

        # potentially clip gradients for stability reasons

        self.optimizer.step()
        self.learn_step_count += 1

        if self.learn_step_count % self.target_dqn_update_interval == 0:
            if self.tau == 1.:
                self.target_q_net.load_state_dict(self.q_net.state_dict())
            else:  
                for target_network_param, q_network_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                    target_network_param.data.copy_(
                        self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
                    )

    def sample_action(self, state: np.ndarray) -> int:

        if self.step_count < self.warmup_count:
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

        # make smarter
        # based on self.schedule

        return 0.1
        


if __name__ == "__main__":

    # for testing

    actor = Actor()
