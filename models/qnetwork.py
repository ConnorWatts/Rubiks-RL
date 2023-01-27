from torch import nn
import numpy as np
import torch

class QNetwork(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()

        #self.conv_dim = config['conv_dim']
        #self.emb_dim = config['emb_dim']

        self.conv1 = nn.Conv3d(6,12,2)
        # put relu in as well
        self.conv2 = nn.Conv3d(12,24,2)
        self.emb = nn.Embedding(12,4)
        self.fc = nn.Linear(48,18)



    def forward(self,state):

        #make neater

        test = torch.unsqueeze(state,4)
        test = self.emb(test)
        test = torch.squeeze(test)
        test = self.conv1(test)
        test = self.conv2(test)
        test = torch.flatten(test,start_dim=1)
        test = self.fc(test)
        return test


if __name__ == "__main__":

    # for testing

    cube = np.array([[[1,1,1],[1,1,1],[1,1,1]],\
                [[2,2,2],[2,2,2],[2,2,2]],\
                [[3,3,3],[3,3,3],[3,3,3]],\
                [[4,4,4],[4,4,4],[4,4,4]],\
                [[5,5,5],[5,5,5],[5,5,5]],\
                [[6,6,6],[6,6,6],[6,6,6]]])

    config = {}

    cube = torch.Tensor([cube for _ in range(32)]).to(torch.int64)

    net = QNetwork(config)
    out = net(cube)
    print('done')


    
