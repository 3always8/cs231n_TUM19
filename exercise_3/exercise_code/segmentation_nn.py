"""SegmentationNN"""
import torch
import torch.nn as nn


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        """
        Design and initialize a convolutional neural network architecture that has input (N, C, H, W) and output (N, num_classes, H, W) and is based on an already pretrained network.
        
        TOTALLY NO IDEA AT ALL
        """
                
        self.conv1 = nn.Conv2d(3,64,3, padding=1)
        self.conv2 = nn.Conv2d(64,128,3, padding=1)
        self.conv3 = nn.Conv2d(128,256,3, padding=1)
        self.conv4 = nn.Conv2d(256,512,3, padding=1)
        self.conv5 = nn.Conv2d(512,100,3, padding=1)
        self.conv0 = nn.Conv2d(100,num_classes,9, padding=4)
        self.up = nn.Upsample(size=240)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        """
        self.model1 = nn.Sequential(
            nn.Conv2d(3,64,3),
            nn.ReLU(),
            nn.Conv2d(64,64,3),
            nn.ReLU(),
#            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.MaxPool2d()
        )
        self.model2 = nn.Sequential(
                        
            nn.Conv2d(3, num_classes, 3),
            nn.Upsample(size=[128,128], mode='bilinear'),
            
            nn.ConvTranspose2d(
                num_classes, num_classes, 4, stride=2, bias=False)
            
        )
        """


        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

#        self.model1(x)
#        self.model2(x)
        n, c, w, h = x.size()

        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.maxpool(x)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = self.maxpool(x)
        x = nn.functional.relu(self.conv5(x))
        x = self.conv0(x)
        x = self.up(x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
