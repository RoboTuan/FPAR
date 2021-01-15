import torch
# For Colab:
from ML_DL_Project.Scripts.cma_doubleResnet import *
# For local:
#import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
# For Colab
#from ML_DL_Project.Scripts.cma_doubleResnet import *
# For local:
from ML_DL_Project.Scripts.MyConvLSTMCell import *


class doubleresnet_lstm_Model(nn.Module):
    """
        In this class we build the model for the standard ego-rnn project.
        In this script are initialized the ResNet34 model, the convLSTM and the 
        CAM and attentionMAP computations. See comments for each part. 
    """
    def __init__(self, num_classes=61, mem_size=512, seqLen=7, flowModel='', rgbModel=''):
        super(doubleresnet_lstm_Model, self).__init__()
        self.num_classes = num_classes
        # Initialize the ResNet
        self.resNet = crossModresnet34(flowModel, rgbModel, True, True)
        self.mem_size = mem_size
        # Get the weighs of the last fc layer of the ResNet,
        # we need this to comput the CAM and the attentionMAP
        self.weight_softmax = self.resNet.cm_rgb_fc.weight
        # Initialize the convLSTM
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        # Here I initialize another avgpool needed after the convLSTM
        self.avgpool = nn.AvgPool2d(7)
        self.conv1_1 = nn.Conv2d(seqLen, 1, kernel_size=1, stride = 1, padding =1 , bias=False)
        self.dropout = nn.Dropout(0.7)
        # Here I initialize the last classifier
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

    def forward(self, inputVariableFlow, inputVariableFrame):
        # Initialize states for the convolutional lstm cell
        state = (Variable(torch.zeros((inputVariableFrame.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariableFrame.size(1), self.mem_size, 7, 7)).cuda()))
        flow_features_maps = []
        # Iterate over temporally sequential images 
        for t in range(inputVariableFrame.size(0)):
            # The logits are the result of the last layer of the cnn (the raw ouput) without softmax
            # Pass the image to the resnet and get back the featuremap at the end of the resnet in "logit"
            # get returned in feature_conv and feature_convNBN the features map of the 4th layer of the resnet
            logit, feature_conv, feature_convNBN, flow_features = self.resNet(inputVariableFlow[t],inputVariableFrame[t])
            #stack the flow features along a new dimension:
            flow_features_maps.append(flow_features)
            bz, nc, h, w = feature_conv.size()
            # Fescale feature conv from (bz,nc,h,w) to (bz,nc,h*w)
            feature_conv1 = feature_conv.view(bz, nc, h*w)
            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            # Here the CAM is computed
            # bmm is the batch matrix-matrix product, example:
                # pay attention that in pytorch this convention is followed in tensors -> [N, C, H, W]
                # that means [Batch_size, channels (depth), height, width],
                # if I multiply 2 tensors with respectively dimensions [10, 3, 20, 10] and [10, 3, 10, 30]
                # I will get with bmm a result tensor of dimension [10, 3, 20, 30] (the number of colums of the first
                # matrix is equal to the number of rows of the second one, considering only H and W. If I change the order of 
                # the tensors in bmm I will have different results or errors).
            # the self.weight_softmax are basically the weights of the last classifier of the ResNet.
            # In the ResNet implementation of Pytorch, we don't find the softmax because 
            # it's trained with a cross entropy loss. In Pytorch, this criterion combines nn.LogSoftmax() 
            # and nn.NLLLoss() in one single class (consists of nn.LogSoftmax and then nn.NLLLoss). 
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            # To compute the attentionMAP (check paper), first we pass the CAM though a softmax
            # and then we multiply this result by the feature_convNBN
            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
            #convLSTM
            state = self.lstm_cell(attentionFeat, state)
        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        flow_feats = torch.stack(flow_features_maps,0)
        flow_feats = self.conv1_1(flow_feats)
        feats = self.classifier(feats1)
        return feats, feats1, flow_feats
