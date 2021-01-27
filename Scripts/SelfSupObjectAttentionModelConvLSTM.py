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

class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SelfSupAttentionModel(nn.Module):
    def __init__(self, num_classes=61, mem_size=512, REGRESSOR=False, Flow=False):
        super(SelfSupAttentionModel, self).__init__()
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

        # Adding flag for the regression option
        self.REGR = REGRESSOR
        print("Regression: ", self.REGR)
        self.FLOW = Flow
        print("FLow: ", self.FLOW)

        #Secondary, self-supervised, task branch
        #Relu+conv+flatten+fullyconnected to get a 2*7*7 = 96 length 
        self.mmapPredictor = nn.Sequential()
        self.mmapPredictor.add_module('mmap_relu',nn.ReLU(True))
        self.mmapPredictor.add_module('convolution', nn.Conv2d(512, 100, kernel_size=1))
        self.mmapPredictor.add_module('flatten',Flatten())
        
        # Different dimensions for the standard selfSup and regSelfSul tasks
        if self.FLOW is True:
            if self.REGR == True:
                self.mmapPredictor.add_module('fc_2',nn.Linear(100*7*7,14*7))
            else:
                self.mmapPredictor.add_module('fc_2',nn.Linear(100*7*7,2*14*7))
        else:
            if self.REGR == True:
                self.mmapPredictor.add_module('fc_2',nn.Linear(100*7*7,7*7))
            else:
                self.mmapPredictor.add_module('fc_2',nn.Linear(100*7*7,2*7*7))




    def forward(self, inputVariable):
        # Initialize states for the convolutional lstm cell
        state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))

        # Iterate over temporally sequential images
        for t in range(inputVariable.size(0)):

            # Pass the image to the resnet and get back the featuremap at the end of the resnet in "logit"
            # get returned in feature_conv and feature_convNBN the features map of the 4th layer of the resnet
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h*w)

            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)

            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)

            # Prediction of the mmap
            # SelfSupervised task
            # Gget a copy of the resnet 4th layer feature map
            feature_conv2 = feature_conv.clone()
            # If is the first image of the temporal sequence create a new feature map vector and
            # put the output of the selfsupervised net with the copy of before in it
            if t == 0:
              map_predictions = self.mmapPredictor(feature_conv2)
            
            # Otherwise if is not the first image simply concatenate along dim=0 the output of the selfsup net
            else:
              prediction = self.mmapPredictor(feature_conv2) #This can be feature_conv
              map_predictions = torch.cat([map_predictions,prediction],dim=0)

            state = self.lstm_cell(attentionFeat, state)
        
        # IMPORTANTE LEVARE QUESTA PARTE
        print(map_predictions.shape)
        idt = map_predictions[0]
        idt = torch.reshape(idt, (7,7))
        print(idt.shape)
        fig = plt.figure()
        plt.imshow(idt.numpy())
        sys.exit()
        # FINE IMPORTANTE

        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        # In the end return the feature maps of the conv lstm cell and
        # the motionmap predictions obtained by the selfsupervised task 
        return feats, feats1, map_predictions
