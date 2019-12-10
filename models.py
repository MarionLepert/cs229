import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        N1 = 50
        F1 = 5
        stride_1 = 1
        pool1 = 1
        out1 = (N1 - F1)/stride_1 + 1
        
        N2 = out1 / pool1 # use max pooling
        F2 = 4
        stride_2 = 1
        pool2 = 1
        out2 = (N2 - F2)/stride_2 + 1
        
        self.out_features = out2 / pool2 # use max pooling 
        
        self.conv1 = nn.Conv1d(in_channels=384, out_channels=64,  kernel_size=F1, stride=stride_1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=F2, stride=stride_2)
        self.fc1 = nn.Linear(16 * int(self.out_features), 400)
        self.fc2 = nn.Linear(400, 250)
        self.fc3 = nn.Linear(250, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
#         x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = F.relu(self.conv2(x))
#         x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * int(self.out_features))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


