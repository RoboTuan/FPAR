import torch
from ML_DL_Project.Scripts.flow_resnet import *
from ML_DL_Project.Scripts.objectAttentionModelConvLSTM import *
import torch.nn as nn


class twoStreamAttentionModel(nn.Module):
    def __init__(self, idtModel='', frameModel='', idtSeqLen=5, stackSize=5, memSize=512, num_classes=61):
        super(twoStreamAttentionModel, self).__init__()
        self.idtModel = flow_resnet34(False, channels=1*idtSeqLen, num_classes=num_classes)
        if idtModel != '':
            self.idtModel.load_state_dict(torch.load(idtModel))
        self.frameModel = attentionModel(num_classes, memSize)
        if frameModel != '':
            self.frameModel.load_state_dict(torch.load(frameModel))
        self.fc2 = nn.Linear(512 * 2, num_classes, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(self.dropout, self.fc2)

    def forward(self, inputVariableIdt, inputVariableFrame):
        _, idtFeats = self.idtModel(inputVariableIdt)
        _, rgbFeats = self.frameModel(inputVariableFrame)
        twoStreamFeats = torch.cat((idtFeats, rgbFeats), 1)
        return self.classifier(twoStreamFeats)