import pandas as pd
import random
import numpy as np
import torch

from typing import Dict, List, Tuple, Union

class ReplayBuffer:
    def __init__(self,config) -> None:
        super().__init__()

        # individual buffers
        # maybe store as array of zeros - might be better on memory
        self.state_buffer = np.array([])
        self.next_state_buffer =np.array([])
        self.action_buffer = np.array([])
        self.reward_buffer = np.array([])
        self.done_buffer = np.array([])

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray] :
        rand_idx = random.sample(range(0, len(self.state_buffer)), batch_size)
        states = torch.tensor(self.state_buffer[rand_idx])
        next_states = torch.tensor(self.next_state_buffer[rand_idx])
        action = self.action_buffer[rand_idx]
        reward = self.reward_buffer[rand_idx]
        done = self.done_buffer[rand_idx]
        return states, next_states, action, done

    def store_transition(self, state, next_state, action, reward, done) -> None:
        self.state_buffer.append(state)
        self.next_state_buffer.append(next_state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)

    def prune(self) -> None:
        # delete early entries of the buffer
        ...