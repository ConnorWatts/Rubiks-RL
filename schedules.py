
class LinearSchedule:
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.start_epsilon = config['start_epsilon']
        self.end_epsilon = config['end_epsilon']
        self.total_steps = config['total_train_steps']
        self.warm_up_steps = config['num_warmup_steps']

    def __call__(self, time_step: int) -> float:
        schedule_step = time_step - self.warm_up_steps
        total_schedule_steps = self.total_steps - self.warm_up_steps
        return self.start_epsilon + (schedule_step*(self.end_epsilon - self.start_epsilon)/total_schedule_steps )


class ConstantSchedule:
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.start_epsilon = config['start_epsilon']

    def __call__(self, time_step: int) -> float:
        return self.start_epsilon

class ExponentialSchedule:
    def __init__(self, config) -> None:
        super().__init__()
        ...

    def __call__(ts):
        ...

if __name__ == "__main__":

    # TO DO - Test Linear Schedule
    # TO DO - Create Exp Schedule

    ...