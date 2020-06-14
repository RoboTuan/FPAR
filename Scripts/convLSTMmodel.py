import torch
# For Colab:
import ML_DL_Project.Scripts.resnetMod as resnetMod
# For local:
#import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
# For Colab
from ML_DL_Project.Scripts.MyConvLSTMCell import *
# For local:
#from MyConvLSTMCell import *


class CLSTM_Model(nn.Module):
    def __init__(self, num_classes=61, mem_size=512):
        super(attentionModel, self).__init__()
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

    def forward(self, inputVariable):
        state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
            attentionFeat = feature_convNBN 
            state = self.lstm_cell(attentionFeat, state)
        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        return feats, feats1
