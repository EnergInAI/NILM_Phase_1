import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2PointCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 30, 10)
        self.conv2 = nn.Conv1d(30, 30, 8)
        self.conv3 = nn.Conv1d(30, 40, 6)
        self.conv4 = nn.Conv1d(40, 50, 5)
        self.conv5 = nn.Conv1d(50, 50, 5)

        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = x.mean(dim=2)
        return self.fc(x).squeeze()
