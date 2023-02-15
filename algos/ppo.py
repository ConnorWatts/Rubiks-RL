from torch.utils.tensorboard import SummaryWriter

class PPO:
    def __init__(self, config, env) -> None:
        super().__init__()

        """
        Constructs the PPO Class for the experiment
        
       :param config: Dict denoting the command line inputs to be 
            used as the configuration of the experiment 
        :param env: gym Env denoting Rubik's Cube environment
        
        """

        self.writer = SummaryWriter()

        self.env = env

    def train(self) -> None:
        ...

    def eval(self) -> None:
        ...