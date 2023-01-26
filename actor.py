from torch.utils.tensorboard import SummaryWriter


class Actor:
    def __init__(self, config, replay_buffer, dqn, target, env, opt ,schedule) -> None:
        super().__init__()

        self.collect_ratio = config['collect_ratio']

        self.state = env.reset()
        self.replay_buffer = replay_buffer

        self.dqn = dqn
        self.target = target

        self.tensorboard_writer = SummaryWriter()
        self.schedule = schedule
        self.optimizer = opt

    def collect_experience(self) -> None:

        for _ in range(self.collect_ratio):
            continue
        ...

    def learn_from_experience(self) -> None:
        ...

if __name__ == "__main__":

    # for testing

    actor = Actor()
