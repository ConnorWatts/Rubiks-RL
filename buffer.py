import pandas as pd
import numpy as np

from typing import Dict, List, Tuple, Union

class ReplayBuffer:
    def __init__(self,config) -> None:
        super().__init__()

        self.buffer = pd.DataFrame(columns=['state', 'next state','action', 'reward','done'])

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] :
        sample = self.buffer.sample(batch_size)
        return sample['state'].to_numpy(), sample['next state'].to_numpy(), sample['action'], sample['reward'], sample['done']

    def store_transition(self, state, next_state, action, reward, done) -> None:
        d = {'state':state.tolist(), 'next state':next_state.tolist(),'action':action, 'reward':reward,'done':done}
        store_df = pd.DataFrame(data= d)
        self.buffer.append(store_df, ignore_index=True)

    def prune(self) -> None:
        # delete early entries of the buffer
        ...