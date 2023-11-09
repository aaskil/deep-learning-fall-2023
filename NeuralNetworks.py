"""
version:        1.4
version_hist:   updated to use BCELoss instead of CrossEntropyLoss, fixed SixConvOneDenseCNNnet (mum_featres -> num_features)
author:         Askil Folgerø
Date:           2023-11-1
Description:    This file contains the final models used in the project.

notes on importing:
    please import into your training environment.
    from NeuralNetworks import FouConvThrDenseCNNnet, FivConvOneDenseCNNnet, SixConvOneDenseCNNnet
    
    alternatively:
    from NeuralNetworks import *

notes on reproducibility:
    use seed 42 for reproducibility
    torch.manual_seed(42)
    we are still using shuffle in dataloader, so the order of the data is not reproducible, therefor there will be some variation in the results

notes on batch normalization:
    as agreed upon in the project group, batch normalization is used after we have found the optimal parameters for the model.

rationale for the models:
    after testing 96 different models, with different combinations of convolutional layers, fully connected layers, sizes of the fully connected layers, reductions in spacial size of the convolutional layers, ive found that the following models give the best results. ive included 
        - two models with 1 fully connected layer, since they perform well, 
        - and one model with 3 fully connected layers, since it performs almost as well, and it would be interesting to  see if it improves, after parameter tuning and regularization.

"""
import torch
import torch.nn as nn

class FouConvThrDenseCNNnet(nn.Module):
    """
    torch.manual_seed(42)
    should give an accuracy of 0.70 - 0.75 without batch normalization
    should give an accuracy of 0.73 - 0.78 with batch normalization 
    input size: 256x256x3

    recommended use:
    NeuralNetworks.fouConvTwoDenseCNNnet(0, 8, 16, 32, 64, False) = 4k
    results in a model with 4k spacial input for the first fully connected layer

    alternative use:
    NeuralNetworks.fouConvTwoDenseCNNnet(0, 4, 8, 16, 32, False)    = 2k
    NeuralNetworks.fouConvTwoDenseCNNnet(0, 16, 32, 64, 128, False) = 8k

    params:
    dropout_rate:   dropout rate for the fully connected layers, decide if you want to use both
    out_1:          number of output channels for the first convolutional layer
    out_2:          number of output channels for the second convolutional layer
    out_3:          number of output channels for the third convolutional layer
    out_4:          number of output channels for the fourth convolutional layer
    use_batchnorm:  if True, batch normalization is used after each convolutional layer, if False, Identity is used, which returns the input
    """
    def __init__(self, dropout_rate, out_1, out_2, out_3, out_4, use_batchnorm: bool):
        super(FouConvThrDenseCNNnet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels = out_1, kernel_size = 3, padding = 1, stride = 2) # [x, 128, 128]
        self.bn1 = nn.BatchNorm2d(num_features = out_1) if use_batchnorm else nn.Identity()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2) # [x, 64, 64]

        self.conv2 = nn.Conv2d(in_channels = out_1, out_channels = out_2, kernel_size = 3, padding = 1, stride = 1) # [x, 64, 64]
        self.bn2 = nn.BatchNorm2d(num_features = out_2) if use_batchnorm else nn.Identity()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2) # [x, 32, 32]

        self.conv3 = nn.Conv2d(in_channels = out_2, out_channels = out_3, kernel_size = 3, padding =1 , stride = 1) # [x, 32, 32]
        self.bn3 = nn.BatchNorm2d(num_features = out_3) if use_batchnorm else nn.Identity()
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2) # [x, 16, 16]

        self.conv4 = nn.Conv2d(in_channels = out_3, out_channels = out_4, kernel_size = 3, padding = 1, stride = 1) # [x, 16, 16]
        self.bn4  = nn.BatchNorm2d(num_features = out_4) if use_batchnorm else nn.Identity()
        self.maxpool4 = nn.MaxPool2d(kernel_size = 2) # [x, 8, 8]

        self.final_output_size = out_4 * 8 * 8 # 64 * 8 * 8 = 4096
        self.fc1 = nn.Linear(self.final_output_size, 32) 
        self.fc2 = nn.Linear(32, 16) 
        self.fc3 = nn.Linear(16, 1)

        self.dropout = nn.Dropout(dropout_rate)
        self.ReLu = nn.ReLU()

    def forward(self, x):
        x = self.ReLu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        x = self.ReLu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)

        x = self.ReLu(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)

        x = self.ReLu(self.bn4(self.conv4(x)))
        x = self.maxpool4(x)

        x = x.view(x.size(0), -1)
        
        x = self.ReLu(self.fc1(x))
        x = self.dropout(x)
        x = self.ReLu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x

    def dense_input_size(self):
        return self.final_output_size

    def name(self):
        return "NeuralNetworks.FouConvThrDenseCNNnet"

class FivConvOneDenseCNNnet(nn.Module):
    """
    torch.manual_seed(42)
    should give an accuracy of 0.70 - 0.75 without batch normalization
    should give an accuracy of 0.75 - 0.80 with batch normalization

    input size: 256x256x3


    recommended use:
    NeuralNetworks.fivConvOneDenseCNNnet(0, 16, 32, 64, 128, 256, False) = 4k
    results in a model with 4k spacial input for the first fully connected layer

    alternative use:
    NeuralNetworks.fivConvOneDenseCNNnet(0, 4, 8, 16, 32, 64, False) = 1k
    NeuralNetworks.fivConvOneDenseCNNnet(0, 8, 16, 32, 64, 128, False) = 4k
    NeuralNetworks.fivConvOneDenseCNNnet(0, 32, 64, 128, 256, 512, False) = 8k

    params:
    dropout_rate:   dropout rate for the fully connected layers, is a parameter, but not in the models, since only one dense layer is used
    out_1:          number of output channels for the first convolutional layer
    out_2:          number of output channels for the second convolutional layer
    out_3:          number of output channels for the third convolutional layer
    out_4:          number of output channels for the fourth convolutional layer
    out_5:          number of output channels for the fifth convolutional layer
    use_batchnorm:  if True, batch normalization is used after each convolutional layer, if False, Identity is used, which returns the input
    """
    def __init__(self, dropout_rate, out_1, out_2, out_3, out_4, out_5, use_batchnorm: bool):
        super(FivConvOneDenseCNNnet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = out_1, kernel_size = 3, padding = 1, stride = 2) # [x, 128, 128]
        self.bn1 = nn.BatchNorm2d(num_features = out_1) if use_batchnorm else nn.Identity()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2) # [x, 64, 64]

        self.conv2 = nn.Conv2d(in_channels = out_1, out_channels = out_2, kernel_size = 3, padding = 1, stride = 1) # [x, 64, 64]
        self.bn2 = nn.BatchNorm2d(num_features = out_2) if use_batchnorm else nn.Identity()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2) # [x, 32, 32]

        self.conv3 = nn.Conv2d(in_channels = out_2, out_channels = out_3, kernel_size = 3, padding = 1, stride = 1) # [x, 32, 32]
        self.bn3 = nn.BatchNorm2d(num_features = out_3) if use_batchnorm else nn.Identity()
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2) # [x, 16, 16]

        self.conv4 = nn.Conv2d(in_channels = out_3, out_channels = out_4, kernel_size = 3, padding = 1, stride = 1) # [x, 16, 16]
        self.bn4 = nn.BatchNorm2d(num_features = out_4) if use_batchnorm else nn.Identity()
        self.maxpool4 = nn.MaxPool2d(kernel_size = 2) # [x, 8, 8]

        self.conv5 = nn.Conv2d(in_channels = out_4, out_channels = out_5, kernel_size = 3, padding = 1, stride = 1) # [x, 8, 8]
        self.bn5 = nn.BatchNorm2d(num_features = out_5) if use_batchnorm else nn.Identity()
        self.maxpool5 = nn.MaxPool2d(kernel_size =  2) # [x, 4, 4]

        self.final_output_size = out_5 * 4 * 4 # 256 * 4 * 4 = 4096
        self.fc1 = nn.Linear(self.final_output_size, 1) 

        self.dropout = nn.Dropout(dropout_rate)
        self.ReLu = nn.ReLU()

    def forward(self, x):
        x = self.ReLu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        x = self.ReLu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)

        x = self.ReLu(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)

        x = self.ReLu(self.bn4(self.conv4(x)))
        x = self.maxpool4(x)

        x = self.ReLu(self.bn5(self.conv5(x)))
        x = self.maxpool5(x)

        x = x.view(x.size(0), -1)  
        x = self.dropout(x)
        x = torch.sigmoid(self.fc1(x))

        return x

    def dense_input_size(self):
        return self.final_output_size

    def name(self):
        return "NeuralNetworks.FivConvOneDenseCNNnet"

class SixConvOneDenseCNNnet(nn.Module):
    """
    torch.manual_seed(42)
    should give an accuracy of 0.68 - 0.73 without batch normalization
    should give an accuracy of 0.76 - 0.82 with batch normalization

    input size: 256x256x3

    recommended use:
    NeuralNetworks.sixConvOneDenseCNNnet(0, 32, 64, 128, 256, 512 False) = 2k
    results in a model with 2k spacial input for the first fully connected layer

    alternative use:
    NeuralNetworks.sixConvOneDenseCNNnet(0, 8, 16, 32, 64, 128, False) = 512
    NeuralNetworks.sixConvOneDenseCNNnet(0, 16, 32, 64, 128, 256, False) = 1k
    NeuralNetworks.sixConvOneDenseCNNnet(0, 64, 128, 256, 512, 1024, False) = 4k
    NeuralNetworks.sixConvOneDenseCNNnet(0, 128, 256, 512, 1024, 2048, False) = 8k

    params:
    dropout_rate:   dropout rate for the fully connected layers, is a parameter, but not in the models, since only one dense layer is used
    out_1:          number of output channels for the first convolutional layer
    out_2:          number of output channels for the second convolutional layer
    out_3:          number of output channels for the third convolutional layer
    out_4:          number of output channels for the fourth convolutional layer
    out_5:          number of output channels for the fifth convolutional layer
    out_6:          number of output channels for the sixth convolutional layer
    use_batchnorm:  if True, batch normalization is used after each convolutional layer, if False, Identity is used, which returns the input
    """
    def __init__(self, dropout_rate, out_1, out_2, out_3, out_4, out_5, out_6, use_batchnorm: bool):
        super(SixConvOneDenseCNNnet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = out_1, kernel_size = 3, padding = 1, stride = 2) # [x, 128, 128]
        self.bn1 = nn.BatchNorm2d(num_features = out_1) if use_batchnorm else nn.Identity()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2) # [x, 64, 64]

        self.conv2 = nn.Conv2d(in_channels = out_1, out_channels = out_2, kernel_size = 3, padding = 1, stride = 1) # [x, 64, 64]
        self.bn2 = nn.BatchNorm2d(num_features = out_2) if use_batchnorm else nn.Identity()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2) # [x, 32, 32]

        self.conv3 = nn.Conv2d(in_channels = out_2, out_channels = out_3, kernel_size = 3, padding = 1, stride = 1)  # [x, 32, 32]
        self.bn3 = nn.BatchNorm2d(num_features = out_3) if use_batchnorm else nn.Identity()
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2) # [x, 16, 16]

        self.conv4 = nn.Conv2d(in_channels = out_3, out_channels = out_4, kernel_size = 3, padding = 1, stride = 1) # [x, 16, 16]
        self.bn4 = nn.BatchNorm2d(num_features = out_4) if use_batchnorm else nn.Identity()
        self.maxpool4 = nn.MaxPool2d(kernel_size = 2) # [x, 8, 8]

        self.conv5 = nn.Conv2d(in_channels = out_4, out_channels = out_5, kernel_size = 3, padding = 1, stride = 1) # [x, 8, 8]
        self.bn5 = nn.BatchNorm2d(num_features = out_5) if use_batchnorm else nn.Identity()
        self.maxpool5 = nn.MaxPool2d(kernel_size = 2) # [x, 4, 4]
        
        self.conv6 = nn.Conv2d(in_channels = out_5, out_channels = out_6, kernel_size = 3, padding = 1, stride = 1) # [x, 4, 4]
        self.bn6 = nn.BatchNorm2d(num_features = out_6) if use_batchnorm else nn.Identity()
        self.maxpool6 = nn.MaxPool2d(kernel_size = 2) # [x, 2, 2]

        self.final_output_size = out_6 * 2 * 2 # 512 * 2 * 2 = 2048
        self.fc1 = nn.Linear(self.final_output_size, 1) 

        self.dropout = nn.Dropout(dropout_rate)
        self.ReLu = nn.ReLU()

    def forward(self, x):
        x = self.ReLu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        x = self.ReLu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)

        x = self.ReLu(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)

        x = self.ReLu(self.bn4(self.conv4(x)))
        x = self.maxpool4(x)

        x = self.ReLu(self.bn5(self.conv5(x)))
        x = self.maxpool5(x)

        x = self.ReLu(self.bn6(self.conv6(x)))
        x = self.maxpool6(x)

        x = x.view(x.size(0), -1)

        x = torch.sigmoid(self.fc1(x))

        return x

    def dense_input_size(self):
        return self.final_output_size

    def name(self):
        return "NeuralNetworks.SixConvOneDenseCNNnet"