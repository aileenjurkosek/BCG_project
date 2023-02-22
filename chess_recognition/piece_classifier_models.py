from torch import nn
import torch.nn.functional as F


#TODO: actually use the specified model_name instead of the hardcoded version
def get_piece_classifier(model_name: str):
    return CNN100_3Conv_3Pool_3FC()

class CNN100_3Conv_3Pool_3FC(nn.Module):

    input_size = 100, 200
    pretrained = False

    def __init__(self):
        super().__init__()
        # Input size: 100x200
        self.conv1 = nn.Conv2d(3, 16, 5)  # 96x196
        self.pool1 = nn.MaxPool2d(2, 2)  # 48x98
        self.conv2 = nn.Conv2d(16, 32, 5)  # 44x94
        self.pool2 = nn.MaxPool2d(2, 2)  # 22x47
        self.conv3 = nn.Conv2d(32, 64, 3)  # 20x45
        self.pool3 = nn.MaxPool2d(2, 2)  # 10x22
        self.fc1 = nn.Linear(64 * 10 * 22, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, len({"pawn", "knight", "bishop", "rook", "queen", "king"}) * 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 10 * 22)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x