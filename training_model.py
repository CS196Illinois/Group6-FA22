import torch
from torch import nn 
import torchvision
from torchsummary import summary

class ModelBoilerplate(nn.Module):
    def __init__(self):
        super(ModelBoilerplate, self).__init__()
    
    def forward(self, x):
        return x

# Our first model will be a simple Convolutional Neural Network
sample_data = torch.rand(4, 3, 224, 224) # Four images of resolution 224x224.
# The 3 is before the H/W by convention, called 'channels-first'

num_classes = 10


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(VideoClassifier, self).__init__()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(3)
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=7, padding='same') 
        nn.ReLU(),
        self.conv2 = nn.Conv3d(32, 32, 3, padding='same')
        nn.ReLU(),
        self.conv3 = nn.Conv3d(32, 32, 3, padding='same')
        nn.ReLU(),
        self.maxpool = nn.MaxPool3d(3)
        self.conv4 = nn.Conv3d(32, 64, 3, padding = "same")
        nn.ReLU(),
        self.maxpool = nn.MaxPool3d(3)
        
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        
        print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
    model = VideoClassifier(num_classes=10)
    model(sample_data).shape 