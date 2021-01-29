import torch
from ML_DL_Project.Scripts.cma_doubleRes_p_LSTM import *
import torch.nn as nn


class crossAttentionModel(nn.Module):
    def __init__(self, stackSize=5, memSize=512, num_classes=61,seqLen=7):
        super(crossAttentionModel, self).__init__()
        self.Model = doubleresnet_lstm_Model(num_classes=num_classes, mem_size=memSize, seqLen=seqLen)
        self.fc2 = nn.Linear(512 * 2, num_classes, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(self.dropout, self.fc2)

    def forward(self, inputVariableFlow, inputVariableFrame):
        _, rgbFeats, flowFeats = self.Model(inputVariableFlow, inputVariableFrame)
        twoStreamFeats = torch.cat((flowFeats, rgbFeats), 1)
        return self.classifier(twoStreamFeats)