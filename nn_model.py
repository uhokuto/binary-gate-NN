import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size, hidden_size,  output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)        
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x): # x : 入力
        yin = self.fc1(x)
        yout = F.sigmoid(yin)        
        zin = self.fc2(yout)
        return zin