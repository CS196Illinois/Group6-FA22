#!/usr/bin/env python
# coding: utf-8

# # Group 6 Model Making
# 
# 
# ### Topics:
# 
# Loading your data
# 
# Creating a model

# In[1]:


import torch
from torch import nn 
import torchvision
from torchsummary import summary # pip install torchsummary


# In[2]:


# Some questions:
# What is the shape of our input data? 
# What is the shape of our output?


# # What is the shape of our input data? 
# 
# From our dataset, a video is generally [n_frames, 720, 1280, 3]. 
# Where n_frames ranges from 30-300 (by inspection, might be different). 
# 
# While accounting for an unknown duration (n_frames) is a valuable property for a model to have, we will train our model-making skills with three tasks:
# 
# 1. Create an image classifier
# 2. Create a video classifier
# 3. Create a video classifier for a dynamic number of frames

# In[3]:


# Creating an image classifier
# Below is the backbone for a model. 
# The init function creates layers, the forward function applies the layers to the data (until you return the output)

# This model has no weights and performs no operations
class ModelBoilerplate(nn.Module):
    def __init__(self):
        super(ModelBoilerplate, self).__init__()
    
    def forward(self, x):
        return x


# In[4]:


# Our first model will be a simple Convolutional Neural Network
sample_data = torch.rand(4, 3, 224, 224) # Four images of resolution 224x224.
# The 3 is before the H/W by convention, called 'channels-first'

num_classes = 10
# We will use the following layers:
# Conv2d - https://towardsdatascience.com/pytorch-conv2d-weights-explained-ff7f68f652eb
# ReLU - max(0, x) applied to each element in the activation
# MaxPool - https://computersciencewiki.org/index.php/Max-pooling_/_Pooling

# If you aren't familiar with these layers, please take a moment to review them with the links provided
# I've implemented the first two layers as an example

# General design principles:
    # Reduce the spatial resolution (using maxpool) as images progress to later layers
    # Increase the number of channels as image process to later layers
    # Always ReLU after a conv
# Good references:
    # VGG16: https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png
    # LeNet?: https://i.imgur.com/idpYjBW.png 

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ImageClassifier, self).__init__()
        # common practice to do 5-6 'stages'
        # 224res
        # 112res
        # 56 res
        # 27 res
        # 14 res
        # 7 res
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        # ill do each 2x filters and 3 convs
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding='same') 
        self.conv2 = nn.Conv2d(32, 32, 3, padding='same')
        self.conv3 = nn.Conv2d(32, 32, 3, padding='same')
        
        # more succinct 
        self.block2 = nn.Sequential( 
            nn.Conv2d(32,  64,  3,  padding='same'),
            nn.ReLU(),
            nn.Conv2d(64,  64,  3,  padding='same'),
            nn.ReLU(),
            nn.Conv2d(64,  64,  3,  padding='same'),
            nn.ReLU()
        )
        # You can even use functions!
        self.block3 = self.conv_block(64, 128)
        self.block4 = self.conv_block(128, 256)
        self.block5 = self.conv_block(256, 256) # keep at 256 because why not
        self.block6 = self.conv_block(256, 256)
        # 7x7x256 = 12k input dim, seems large so I'll do another maxpool right before
        
        self.fc1 = nn.Linear(2304, 1024)
        self.dropout = nn.Dropout(.3) # look it up! what does it do? Why did I add it? :) 
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        
    def conv_block(self, in_chan, out_chan):
        return  nn.Sequential( #more succinct 
            nn.Conv2d(in_chan,  out_chan,  3,  padding='same'),
            nn.ReLU(),
            nn.Conv2d(out_chan,  out_chan,  3,  padding='same'),
            nn.ReLU(),
            nn.Conv2d(out_chan,  out_chan,  3,  padding='same'),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.block2(x)
        x = self.maxpool(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.block4(x)
        x = self.maxpool(x)
        x = self.block5(x)
        x = self.maxpool(x)
        x = self.block6(x)
        x = self.maxpool(x)
        x = x.flatten(start_dim=1)
        
        print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
model = ImageClassifier(num_classes=10)
model(sample_data)
# A successful model should return 4 (batch), 10 (num_classes)


# In[5]:


# Let's analyze your model:
summary(model, (3, 224, 224))


# In[6]:


# And compare to VGG16
summary(torchvision.models.vgg16(), (3, 224, 224))


# # Task 1
# Answer the following and please describe which layer(s) (or general design) you have may contribute to differences:
# 
# - How do the number of parameters in each model differ? (# of weights)
# - How does the number of convolutional layers in each model differ? (depth)
# - How does the forward/backward pass size of each model differ? (# of activations)
# 
# 

# In[7]:


# Our first model will be a variant of a specific 3D Convolutional Neural Network - C3D. 

# This model was originally designed  for data of shape  16, 3, 112, 112
# Our input data will be of shape  3, 30(time), 224, 224

sample_data = torch.rand(4, 3, 30, 224, 224) 
# Batch, Channel, Time, Height, Width
num_classes = 10

# Please use this as a reference: https://github.com/JJBOY/C3D-pytorch/blob/master/network.py
# Please don't just copy, you'll need to edit layers (and keep track of shape)
# Feel free to make any adjustments to the network as you see fit

class VideoClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(VideoClassifier, self).__init__()

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(3)

        
        # TODO add more layers
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=7, padding='same') 
        nn.ReLU(),
        self.conv2 = nn.Conv3d(32, 32, 3, padding='same')
        nn.ReLU(),
        self.conv3 = nn.Conv3d(32, 32, 3, padding='same')
        nn.ReLU(),
        self.maxpool = nn.MaxPool3d(3)
        self.conv4 = nn.Conv3d(32, 64, 3, padding = "same")
        nn.ReLU(),
        self.conv5 = nn.Conv3d(64, 64, 3, padding = "same")
        nn.ReLU(),
        self.conv6 = nn.Conv3d(64, 128, 3, padding = "same")
        nn.ReLU(),
        self.maxpool = nn.MaxPool3d(3)


        
    def forward(self, x):
        # TODO add more layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = x.flatten(start_dim=1)
        
        print(x.shape)
        
        x = torch.sigmoid(x)
        return x


        self.fc1 = nn.Linear(2304, 1024)
        self.dropout = nn.Dropout(.3) # look it up! what does it do? Why did I add it? :) 
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        
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
# A successful model should return 4 (batch), 10 (num_classes)


# In[8]:


summary(model, (3, 30, 224, 224))


# # Task 2
# 
# Compare your VideoClassifier to your image classifier. In particular:
# 
# - How do the number of weights differ?
# - How does the forward/backward pass size differ?

# In[9]:


# Extra Credit/Bonus Problem:

# One approach to classifying videos of arbitrary length is to encode each image seperately (i.e. predict pose for each image)
# Then apply a temporal model (Recurrant Neural Network) to the sequence 
# https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

# RNN example:
rnn = nn.RNN(input_size=256, hidden_size=20, num_layers=2)
x = torch.randn(4, 40, 256) # Batch, Sequence, Encoding
hidden_input = torch.zeros(2, 40, 20) # Layers, Sequence, HiddenDim
output, hn = rnn(x, hidden_input)
print(output.shape) # Batch, Sequence, OutputDim
# The last element in the sequence has seen all of the time steps
output = output[:, -1, :]
print(output.shape) # Batch, OutputDim


# In[10]:




class ImageEncoder(nn.Module):
    def __init__(self, encoding_size=256):
        super(ImageEncoder, self).__init__()
        # TODO add layers (feel free to use ImageClassifier)
        
    def forward(self, x):
        # x.shape = B, T, 3, H, W
        
        # TODO add layers (feel free to use ImageClassifier)
        
        # Return B, T, self.encoding_size
        return x
    
class AnyLengthVideoClassifier(nn.Module):
    def __init__(self, encoding_size=256, num_classes=10):
        super(AnyLengthVideoClassifier, self).__init__()
        self.image_encoder = ImageEncoder(encoding_size=encoding_size)

        rnn_hidden_dim = 256 # feel free to change
        self.temporal_encoder = RNN() # TODO fill out details
        
        self.to_classes = nn.Linear(rnn_hidden_dim, num_classes)
        
    def forward(self, x):
        # x.shape = B, T, 3, H, W
        x = self.image_encoder(x)
        # x.shape = B, T, encoding_size
        x = self.temporal_encoder(x)
        # x.shape = B, T, Hidden_dim
        
        x = self.to_classes(x)
        
        return x


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




