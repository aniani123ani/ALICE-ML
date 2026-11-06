######First MODEL
import torch
import torch.nn as nn
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3,8)
        self.fc2 = nn.Linear(8,1)

    def forward (self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x    
    ############SECOND MODEL VAEs




    ###########Third Model CNNs