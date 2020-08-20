import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):                                     # Constructor, used to set the network layer
        super(Net, self).__init__()                         # Standard statement
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)   # First con layer
        self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)  # Second con layer

        self.fc1 = nn.Linear(50*50*16, 128)                 # first full connect layer
        self.fc2 = nn.Linear(128, 64)                       # second full connect layer
        self.fc3 = nn.Linear(64, 2)                         # third full connect layer

    def forward(self, x):                   #
        x = self.conv1(x)                   # first conv
        x = F.relu(x)                       #
        x = F.max_pool2d(x, 2)              # first pooling

        x = self.conv2(x)                   # second conv
        x = F.relu(x)                       #
        x = F.max_pool2d(x, 2)              # second pooling

        x = x.view(x.size()[0], -1)         # Since the input of the fully connected layer is a one-dimensional tensor,
                                            # it is necessary to arrange the input data in the [50×50×16] format into [40000×1]
        x = F.relu(self.fc1(x))             # first full connection
        x = F.relu(self.fc2(x))             # 2nd full connection
        y = self.fc3(x)                     # 3rd full connection

        return y

