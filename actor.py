from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import random

class Actor:
    def __init__(self, config, replay_buffer, q_net, target_q_net, env, opt ,schedule) -> None:
        super().__init__()

        self.warmup_count = config['num_warmup_steps']
        self.collect_ratio = config['collect_ratio']
        self.batch_size = config['batch_size']

        self.state = env.reset()
        self.replay_buffer = replay_buffer

        self.q_net = q_net
        self.target_q_net = target_q_net

        self.writer = SummaryWriter()
        self.schedule = schedule
        self.optimizer = opt

        self.step_count = 0
        self.learn_step_count = 0

    def collect_experience(self) -> None:

        for _ in range(self.collect_ratio):

            action = self.sample_action(self.state)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.store_transition(self.state, next_state, action, reward, done)
            
            if done:
                self.state = self.env.reset()
                # log episode result
            else:
                self.state = next_state


    def learn_from_experience(self) -> None:

        current_states, actions, rewards, next_states, dones = self.replay_buffer.sample_states(self.batch_size)

        ...

    def sample_action(self, state: np.ndarray) -> int:

        if self.step_count < self.warmup_count:
            action = self.env.action_space.sample()  
        else:
            with torch.no_grad():
                epsilon = self.get_epsilon()
                if random.random() < epsilon:
                    action = self.env.action_space.sample()  
                else:
                    q_values = self.q_net(torch.Tensor(state))
                    action = torch.argmax(q_values, dim=1).cpu().numpy()

        return action

    def get_epsilon(self):

        # make smarter
        # based on self.schedule

        return 0.1
        


if __name__ == "__main__":

    # for testing

    actor = Actor()
