## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

# Implementation of modified version of NamishNet Layer-Wise Architecture
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
    
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # Convolutional Layers (conv1-conv4)
        
        # initial tensor input = (1,224,224)
        # output size = (32,221,221)
        # after one pool layer, this becomes (32,110,110)
        # kernel size 4x4
        self.conv1 = nn.Conv2d(1, 32, 4)
        
        # initial tensor input = (32,110,110)
        # output size = (64,108,108)
        # after one pool layer, this becomes (64,54,54)
        # kernel size 3x3
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # initial tensor input = (64,54,54)
        # output size = (128,53,53)
        # after one pool layer, this becomes (128,26,26)
        # kernel size 2x2
        self.conv3 = nn.Conv2d(64,128,2)
        
        # initial tensor input = (128,26,26)
        # output size = (256,26,26)
        # after one pool layer, this becomes (256,13,13)
        # kernel size 1x1
        self.conv4 = nn.Conv2d(128,256,1)
        
        # Fully-Connected/Dense Layers
        self.fc1 = nn.Linear(256*13*13,1000)
        self.fc2 = nn.Linear(1000,136)
       
        
        # we will be using this pool layer 4 times
        self.pool = nn.MaxPool2d(2,2)
        
        # Dropout layer of 0.1
        self.drop = nn.Dropout(p=0.1)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        # The four convolutional layers with other stuff in between
        x = self.drop(F.selu(self.pool(self.conv1(x))))
        x = self.drop(F.selu(self.pool(self.conv2(x))))
        x = self.drop(F.selu(self.pool(self.conv3(x))))
        x = self.drop(F.selu(self.pool(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0),-1)
        
        # The three dense layers with the other stuff in between
        x = self.drop(F.selu(self.fc1(x)))
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
