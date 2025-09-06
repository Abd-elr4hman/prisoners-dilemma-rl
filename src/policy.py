import torch.nn as nn

class PolicyNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)
    