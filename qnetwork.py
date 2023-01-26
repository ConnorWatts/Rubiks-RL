from torch import nn

class QNetwork(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
