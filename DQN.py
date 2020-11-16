import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_size, num_actions):
        super(DQN, self).__init__()
        self.fc1= nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.head = nn.Linear(128, num_actions)

    # Called to determine next action, or optimization
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)