from torch import nn
import torch.nn.functional as F
from torchvision import models


#TODO: actually use the specified model_name instead of the hardcoded version
def get_occupancy_classifier(model_name: str):
    return CNN100_3Conv_3Pool_3FC()


class CNN100_3Conv_3Pool_3FC(nn.Module):

    input_size = 100, 100
    pretrained = False

    def __init__(self):
        super().__init__()
        # Input size: 100x100
        self.conv1 = nn.Conv2d(3, 16, 5)  # 96
        self.pool1 = nn.MaxPool2d(2, 2)  # 48
        self.conv2 = nn.Conv2d(16, 32, 5)  # 44
        self.pool2 = nn.MaxPool2d(2, 2)  # 22
        self.conv3 = nn.Conv2d(32, 64, 3)  # 20
        self.pool3 = nn.MaxPool2d(2, 2)  # 10
        self.fc1 = nn.Linear(64 * 10 * 10, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class AlexNet(nn.Module):
    input_size = 100, 100
    pretrained = True

    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        n = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(n, 2)
        self.params = {
            "head": list(self.model.classifier[6].parameters())
        }

    def forward(self, x):
        return self.model(x)