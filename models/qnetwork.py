from torch import nn
import numpy as np
import torch

class QNetwork(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()

        self.conv_dim = [12,24] #config['conv_dim']
        self.emb_dim = 4 #config['emb_dim']

        self.emb = nn.Embedding(12,4)

        #if config['act_fnt'] == "ReLU":
        self.act_fnt = nn.ReLU()

        #cnn_modules = []

        #for i in range(len(self.conv_dim)-1):
            #cnn_modules.append(nn.Conv3d())

        #self.cnn_part = nn.Sequential(
            #*cnn_modules,
            #nn.Flatten()  
        #)

        self.conv1 = nn.Conv3d(6,12,2)
        self.conv2 = nn.Conv3d(12,24,2)
        self.fc = nn.Linear(48,18)



    def forward(self,state):

        #make neater

        unsqueeze_layer = torch.unsqueeze(state,4)
        emb_layer = torch.squeeze(self.emb(unsqueeze_layer))
        conv_layer1 = self.act_fnt(self.conv1(emb_layer))
        conv_layer2 = self.act_fnt(self.conv2(conv_layer1))
        return self.fc(torch.flatten(conv_layer2,start_dim=1))


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
    max = out.max(dim=1)[0]
    print('done')


    
