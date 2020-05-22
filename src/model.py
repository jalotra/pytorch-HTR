import torch 
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    
    def __init__(self, is_training):    
        super(BaseModel, self).__init__()
        self.is_training = is_training


        # Input image is of the shape 
        # NCWH (batch x channel x width x height)
        # BLOCK 1
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # BLOCK2
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # BLOCK 3
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 256, out_channles = 256, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool3 = nn.MaxPool2d(kernel_size = (1, 2), stride = 2)

        #Block 4
        self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        # Batch Norm here

        # Block 5
        self.conv6 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        #Batch Norm here

        #Block 6
        self.maxpool4 = nn.MaxPool2d(kernel_size = (1, 2), stride = 2)
        self.conv7 = nn.Conv2d(in_channels = 512, out_channels= 512, stride = 1)

        "------------------------------------------------------------------------------------------"
        "Defining Rnn Layers Now"

    def cnn_forward(self, x):
        bs, c, w, h = x.shape
        assert c == 1
        assert w == 128
        assert h == 32

        # Block 1
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = F.leaky_relu(x)
        print("BLOCK 1 : ", x.shape)

        # Block 2
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = F.leaky_relu(x)
        print("BLOCK 2 : ", x.shape)

        #Block 3
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool3(x)
        x = F.leaky_relu(x)
        print("BLOCK 3 : ", x.shape)

        #Block 4
        x = self.conv5(x)
        x = nn.BatchNorm2d(num_features = x.shape[1])
        x = F.leaky_relu(x)
        print("BLOCK 4 : ", x.shape)

        #Block 5
        x = self.conv6(x)
        x = nn.BatchNorm2d(num_features = x.shape[1])
        x = F.leaky_relu(x)
        print("BLOCK 5 : ", x.shape)

        #Block 6
        x = self.maxpool4(x)
        x = self.conv7(x)
        x = F.leaky_relu(x)
        print("BLOCK 6 : ", x.shape)

        return x 
