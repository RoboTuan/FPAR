from __future__ import print_function, division
import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from torch.autograd import Variable
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import glob
import sys
from ML_DL_Project.Scripts.MyConvLSTMCell import *
from ML_DL_Project.Scripts.cma_makeDatasetTwoStream import *
from ML_DL_Project.Scripts.spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import argparse

import sys

#__all__ = ['crossModresnet34']


model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes,stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, noBN=False):
        super(BasicBlock, self).__init__()
        self.noBN = noBN
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # print('noBN in basicBlock = ', self.noBN)
        outBN = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        outBN = outBN + residual
        outBN = self.relu(outBN)
        if self.noBN is False:
            return outBN
        else:
            outBN = out + residual
            return outBN

class cmaBlock(nn.Module):
    expansion = 1

    def __init__(self, planes, stride=1):
        super(cmaBlock, self).__init__()
        self.stride = stride
        self.convQ = conv1x1(planes, int(planes/2), stride)
        self.convK = conv1x1(planes, int(planes/2), stride)
        self.convV = conv1x1(planes, int(planes/2), stride)
        self.convZ = conv1x1(int(planes/2), planes, stride)
        self.softmax1 = torch.nn.Softmax()
        self.bn = nn.BatchNorm2d(planes)
        

    def forward(self, x, y):
        residual = x
        V = self.convV(y)
        Q = self.convQ(x)
        K = self.convK(y)
        M = self.softmax1(torch.matmul(Q,K.t()))
        Z = torch.mm(M,V)
        Z = self.convZ(Z)
        Z = self.bn(Z)
        out = residual + Z

        return out

class doubleResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000,fl_num_classes=61,noBN=False,channels=10):
        self.inplanes = 64
        self.noBN = noBN
        super(doubleResNet, self).__init__()
        self.cm_rgb_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.cm_rgb_bn1 = nn.BatchNorm2d(64)
        self.cm_rgb_relu = nn.ReLU(inplace=True)
        self.cm_rgb_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cm_rgb_layer1 = self._make_layer(block, 64, layers[0])
        self.cm_rgb_layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.cm_rgb_layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.cm_rgb_cma1= cmaBlock(256, stride=1)
        self.cm_rgb_layer4 = self._make_layer(block, 512, layers[3], stride=2,noBN=self.noBN)
        self.cm_rgb_avgpool = nn.AvgPool2d(7, stride=1)
        self.cm_rgb_fc = nn.Linear(512 * block.expansion, num_classes)

        #2nd resnet34 for flow processing
        self.cm_fl_conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.cm_fl_bn1 = nn.BatchNorm2d(64)
        self.cm_fl_relu = nn.ReLU(inplace=True)
        self.cm_fl_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cm_fl_layer1 = self._make_layer_flow(block, 64, layers[0])
        self.cm_fl_layer2 = self._make_layer_flow(block, 128, layers[1], stride=2)
        self.cm_fl_layer3 = self._make_layer_flow(block, 256, layers[2], stride=2)
        self.cm_fl_cma1= cmaBlock(256, stride=1)
        self.cm_fl_layer4 = self._make_layer_flow(block, 512, layers[3], stride=2)
        self.cm_fl_avgpool = nn.AvgPool2d(7)
        self.dp = nn.Dropout(p=0.5)
        self.cm_fl_fc = nn.Linear(512 * block.expansion, fl_num_classes)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, cmaBlock): 
              for mm in m.modules():
                if isinstance(mm, nn.BatchNorm2d):
                  mm.weight.data.fill_(0)
                  mm.bias.data.zero_()
        #initialize weight of cma_batchnorm as 0:
            
            #trovare nome

    def _make_layer(self, block, planes, blocks, stride=1, noBN=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # print('blocks = ', blocks)
        if noBN is False:
            # print('with BN')
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
        else:
            # print('no BN')
            if blocks > 2:
                # print('blocks > 2')
                for i in range(1, blocks-1):
                    layers.append(block(self.inplanes, planes))
                layers.append(block(self.inplanes, planes, noBN=True))
            else:
                # print('blocks <= 2')
                layers.append(block(self.inplanes, planes, noBN=True))

        return nn.Sequential(*layers)
    
    def _make_layer_flow(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, y, x):
        x = self.cm_rgb_conv1(x)
        y = self.cm_fl_conv1(y)

        x = self.cm_rgb_bn1(x)
        y = self.cm_fl_bn1(y)
        
        x = self.cm_rgb_relu(x)
        y = self.cm_fl_relu(y)

        x = self.cm_rgb_maxpool(x)
        y = self.cm_fl_maxpool(y)


        x = self.cm_rgb_layer1(x)
        y = self.cm_fl_layer1(y)

        x = self.cm_rgb_layer2(x)
        y = self.cm_fl_layer2(y)

        x = self.cm_rgb_layer3(x)
        y = self.cm_fl_layer3(x)

        x = self.cm_rgb_cma1(x,y) 
        y = self.cm_fl_cma1(y,x)
        
        if self.noBN:
            conv_layer4BN, conv_layer4NBN = self.cm_rgb_layer4(x)
        else:
            conv_layer4BN = self.cm_rgb_layer4(x)
        
        y = self.cm_fl_layer4(y)

        # conv_layer4BN.size() is equal to [32, 512, 7, 7]
        # and the avgpool is performed with a kernel of 7x7,
        #print(conv_layer4BN.size())
        x = self.cm_rg_avgpool(conv_layer4BN)
        y = self.cm_fl_avgpool(y)

        # x.size() after the GAP is equal to  [32, 512, 1, 1] as expected
        #print(x.size())
        x = x.view(x.size(0), -1)
        y1 = y.view(y.size(0),-1)

        y = self.dp(y1)

        x = self.cm_rgb_fc(x)
        y = self.cm_fl_fc(y)

        if self.noBN:
            return x, conv_layer4BN, conv_layer4NBN ,y
        else:
            return x, conv_layer4BN, y1

def crossModresnet34(flow_model_dict_PATH, rgb_model_dict_PATH, pretrained=False, noBN=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = doubleResNet(BasicBlock, [3, 4, 6, 3], noBN=noBN, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(rgb_model_dict_PATH), strict=False)
        model.load_state_dict(torch.load(flow_model_dict_PATH), strict=False)
    return model

def change_key_names(old_params, in_channels):
    new_params = collections.OrderedDict()
    layer_count = 0
    allKeyList = old_params.keys()
    for layer_key in allKeyList:
        if layer_count >= len(allKeyList) - 2:
            # exclude fc layers
            continue
        else:
            if layer_count == 0:
                rgb_weight = old_params[layer_key].data
                rgb_weight_mean = torch.mean(rgb_weight, dim=1)
                flow_weight = rgb_weight_mean.unsqueeze(1).repeat(1, in_channels, 1, 1)
                new_params[layer_key] = flow_weight
                layer_count += 1
            else:
                new_params[layer_key] = old_params[layer_key]
                layer_count += 1

    return new_params



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
        #self.conv1_1 = nn.Conv2d(seqLen, 1, kernel_size=1, stride = 1, padding =1 , bias=False)
        self.avgpool_flow = nn.AvgPool3d((seqLen,1))
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
        flow_feats = self.avgpool_flow(flow_feats)
        feats = self.classifier(feats1)
        return feats, feats1, flow_feats



class crossAttentionModel(nn.Module):
    def __init__(self, flowModel='', frameModel='', stackSize=5, memSize=512, num_classes=61,seqLen=7):
        super(crossAttentionModel, self).__init__()
        self.Model = doubleresnet_lstm_Model(num_classes=num_classes, mem_size=memSize, seqLen=seqLen, flowModel=flowModel, rgbModel=frameModel)
        self.fc2 = nn.Linear(512 * 2, num_classes, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(self.dropout, self.fc2)

    def forward(self, inputVariableFlow, inputVariableFrame):
        _, rgbFeats, flowFeats = self.Model(inputVariableFlow, inputVariableFrame)
        twoStreamFeats = torch.cat((flowFeats, rgbFeats), 1)
        return self.classifier(twoStreamFeats)



def gen_split(root_dir, stackSize):
    DatasetX = []
    DatasetY = []
    DatasetF = []
    Labels = []
    NumFrames = []
    root_dir = os.path.join(root_dir, 'flow_x_processed')
    for dir_user in sorted(os.listdir(root_dir)):
        if not dir_user.startswith('.') and dir_user:
          class_id = 0
          directory = os.path.join(root_dir, dir_user)
          action = sorted(os.listdir(directory))
          for target in sorted(os.listdir(directory)):
              if not target.startswith('.'):
                directory1 = os.path.join(directory, target)
                insts = sorted(os.listdir(directory1))
                if insts != []:
                    for inst in insts:
                        inst_dir = os.path.join(directory1, inst)
                        numFrames = len(glob.glob1(inst_dir, '*.png'))
                        if numFrames >= stackSize:
                            DatasetX.append(inst_dir)
                            DatasetY.append(inst_dir.replace('flow_x', 'flow_y'))
                            inst_dir = inst_dir.replace('flow_x_processed', 'processed_frames2')
                            inst_dir = inst_dir + "/rgb"
                            DatasetF.append(inst_dir)
                            Labels.append(class_id)
                            NumFrames.append(numFrames)
                class_id += 1
    return DatasetX, DatasetY, DatasetF, Labels, NumFrames, action


class makeDataset2Stream(Dataset):
    def __init__(self, root_dir, spatial_transform=None, sequence=False, stackSize=5,
                 train=True, numSeg=5, fmt='.png', phase='train', seqLen = 25):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.imagesX, self.imagesY, self.imagesF, self.labels, self.numFrames, self.action = gen_split(
            root_dir, stackSize)
        self.spatial_transform = spatial_transform
        self.train = train
        self.numSeg = numSeg
        self.sequence = sequence
        self.stackSize = stackSize
        self.fmt = fmt
        self.phase = phase
        self.seqLen = seqLen

    def __len__(self):
        return len(self.imagesX)

    def __getitem__(self, idx):
        vid_nameX = self.imagesX[idx]
        vid_nameY = self.imagesY[idx]
        vid_nameF = self.imagesF[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeqSegs = []
        self.spatial_transform.randomize_parameters()
        if self.sequence is True:
            if numFrame <= self.stackSize:
                frameStart = np.ones(self.numSeg)
            else:
                frameStart = np.linspace(1, numFrame - self.stackSize, self.numSeg)
            if self.phase == 'train':
                    startFrame = random.randint(1, numFrame - self.stackSize)
            else:
                    startFrame = startFrame = np.ceil((numFrame - self.stackSize)/2)
            for j in range(self.seqLen):
                inpSeq = []
                for k in range(self.stackSize):
                    i = k + int(startFrame)
                    fl_name = vid_nameX + '/flow_x_' + str(int(round(i))).zfill(5) + '.png'
                    img = Image.open(fl_name)
                    inpSeq.append(self.spatial_transform(img.convert('L'), inv=True, flow=True))
                    # fl_names.append(fl_name)
                    fl_name = vid_nameY + '/flow_y_' + str(int(round(i))).zfill(5) + '.png'
                    img = Image.open(fl_name)
                    inpSeq.append(self.spatial_transform(img.convert('L'), inv=False, flow=True))
                inpSeqSegs.append(torch.stack(inpSeq, 0).squeeze())
            inpSeqSegs = torch.stack(inpSeqSegs, 0)
        else:
            if numFrame <= self.stackSize:
                startFrame = 1
            else:
                if self.phase == 'train':
                    startFrame = random.randint(1, numFrame - self.stackSize)
                else:
                    startFrame = np.ceil((numFrame - self.stackSize)/2)
            time_flow_seq=[]
            inpSeq = []
            for k in range(self.stackSize):
                i = k + int(startFrame)
                fl_name = vid_nameX + '/flow_x_' + str(int(round(i))).zfill(5) + '.png'
                img = Image.open(fl_name)
                inpSeq.append(self.spatial_transform(img.convert('L'), inv=True, flow=True))
                # fl_names.append(fl_name)
                fl_name = vid_nameY + '/flow_y_' + str(int(round(i))).zfill(5) + '.png'
                img = Image.open(fl_name)
                inpSeq.append(self.spatial_transform(img.convert('L'), inv=False, flow=True))
            inpSeqSegs = torch.stack(inpSeq, 0).squeeze(1)
            
        inpSeqF = []
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            fl_name = vid_nameF + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeqF.append(self.spatial_transform(img.convert('RGB')))
        inpSeqF = torch.stack(inpSeqF, 0)
        return inpSeqSegs, inpSeqF, label#, vid_nameF#, fl_name
    def __getLabel__(self):
        return self.action

def main_run(dataset, flowModel, rgbModel, stackSize, seqLen, memSize, trainDatasetDir, valDatasetDir, outDir,
             trainBatchSize, valBatchSize, lr1, numEpochs, decayRate, stepSize):


    if dataset == 'gtea61':
        num_classes = 61
    elif dataset == 'gtea71':
        num_classes = 71
    elif dataset == 'gtea_gaze':
        num_classes = 44
    elif dataset == 'egtea':
        num_classes = 106
    else:
        print('Dataset not found')
        sys.exit()

    # Setting Device
    DEVICE = "cuda"

    model_folder = os.path.join('./', outDir, dataset, 'crossModAtt')  # Dir for saving models and log files
    # Create the dir
    if os.path.exists(model_folder):
        print('Dir {} exists!'.format(model_folder))
        sys.exit()
    os.makedirs(model_folder)

    # Log files
    writer = SummaryWriter(model_folder)
    train_log_loss = open((model_folder + '/train_log_loss.txt'), 'w')
    train_log_acc = open((model_folder + '/train_log_acc.txt'), 'w')
    val_log_loss = open((model_folder + '/val_log_loss.txt'), 'w')
    val_log_acc = open((model_folder + '/val_log_acc.txt'), 'w')


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize = Normalize(mean=mean, std=std)

    spatial_transform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
                                ToTensor(), normalize])

    vid_seq_train = makeDataset2Stream(trainDatasetDir,spatial_transform=spatial_transform,
                                sequence=True, numSeg=1, stackSize=stackSize, fmt='.png', seqLen=seqLen)

    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                            shuffle=True, num_workers=4, pin_memory=True)

    if valDatasetDir is not None:

        vid_seq_val = makeDataset2Stream(valDatasetDir,
                                    spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                                    sequence=False, numSeg=1, stackSize=stackSize, fmt='.png', phase='Test',
                                    seqLen=seqLen)

        val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize,
                                shuffle=False, num_workers=2, pin_memory=True)
        valSamples = vid_seq_val.__len__()

        

    model = crossAttentionModel(flowModel=flowModel, frameModel=rgbModel, stackSize=stackSize, memSize=memSize,
                                        num_classes=num_classes)

    for params in model.parameters():
        params.requires_grad = False

    model.train(False)
    train_params = []

    for params in model.classifier.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.Model.lstm_cell.parameters():
        train_params += [params]
        params.requires_grad = True
    
    for params in model.Model.resNet.cm_rgb_cma1.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.Model.resNet.cm_rgb_layer4[0].conv1.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.Model.resNet.cm_rgb_layer4[0].conv2.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.Model.resNet.cm_rgb_layer4[1].conv1.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.Model.resNet.cm_rgb_layer4[1].conv2.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.Model.resNet.cm_rgb_layer4[2].conv1.parameters():
        params.requires_grad = True
        train_params += [params]
    #
    for params in model.Model.resNet.cm_rgb_layer4[2].conv2.parameters():
        params.requires_grad = True
        train_params += [params]
    #
    for params in model.Model.resNet.cm_rgb_fc.parameters():
        params.requires_grad = True
        train_params += [params]

    base_params = []
    for params in model.Model.resNet.cm_fl_layer4.parameters():
        base_params += [params]
        params.requires_grad = True
    
    for params in model.Model.resNet.cm_fl_cma1.parameters():
        base_params += [params]
        params.requires_grad = True

    model = model.to(DEVICE)

    trainSamples = vid_seq_train.__len__()
    min_accuracy = 0

    loss_fn = nn.CrossEntropyLoss()
    optimizer_fn = torch.optim.SGD([
        {'params': train_params},
        {'params': base_params, 'lr': 1e-4},
    ], lr=lr1, momentum=0.9, weight_decay=5e-4)

    optim_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_fn, step_size=stepSize, gamma=decayRate)
    
    train_iter = 0


    for epoch in range(numEpochs):
        epoch_loss = 0
        numCorrTrain = 0
        iterPerEpoch = 0
        model.classifier.train(True)
        model.Model.resNet.cm_fl_layer4.train(True)
        for j, (inputFlow, inputFrame, targets) in enumerate(train_loader):
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            inputVariableFlow = inputFlow.to(DEVICE)
            inputVariableFrame = inputFrame.permute(1, 0, 2, 3, 4).to(DEVICE)
            labelVariable = Variable(targets.cuda())
            output_label = model(inputVariableFlow, inputVariableFrame)
            loss = loss_fn(F.log_softmax(output_label, dim=1), labelVariable)
            loss.backward()
            optimizer_fn.step()
            _, predicted = torch.max(output_label.data, 1)
            numCorrTrain += (predicted == targets.cuda()).sum()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / iterPerEpoch
        trainAccuracy = torch.true_divide(numCorrTrain, trainSamples) * 100
        print('Average training loss after {} epoch = {} '.format(epoch + 1, avg_loss))
        print('Training accuracy after {} epoch = {}% '.format(epoch + 1, trainAccuracy))
        writer.add_scalar('train/epoch_loss', avg_loss, epoch + 1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch + 1)
        train_log_loss.write('Training loss after {} epoch = {}\n'.format(epoch + 1, avg_loss))
        train_log_acc.write('Training accuracy after {} epoch = {}\n'.format(epoch + 1, trainAccuracy))
        optim_scheduler.step()
        
        if valDatasetDir is not None:
            if (epoch + 1) % 1 == 0:
                model.train(False)
                val_loss_epoch = 0
                val_iter = 0
                numCorr = 0
                # wrapping with torch.no_grad() because it wasn't present
                # check if it makes sense
                with torch.no_grad():
                    for j, (inputFlow, inputFrame, targets) in enumerate(val_loader):
                        val_iter += 1
                        inputVariableFlow = inputFlow.to(DEVICE)
                        inputVariableFrame = inputFrame.permute(1, 0, 2, 3, 4).to(DEVICE)
                        labelVariable = targets.to(DEVICE)
                        output_label = model(inputVariableFlow, inputVariableFrame)
                        loss = loss_fn(F.log_softmax(output_label, dim=1), labelVariable)
                        val_loss_epoch += loss.item()
                        _, predicted = torch.max(output_label.data, 1)
                        numCorr += (predicted == labelVariable.data).sum()
                val_accuracy = torch.true_divide(numCorr, valSamples) * 100
                avg_val_loss = val_loss_epoch / val_iter
                print('Val Loss after {} epochs, loss = {}'.format(epoch + 1, avg_val_loss))
                print('Val Accuracy after {} epochs = {}%'.format(epoch + 1, val_accuracy))
                writer.add_scalar('val/epoch_loss', avg_val_loss, epoch + 1)
                writer.add_scalar('val/accuracy', val_accuracy, epoch + 1)
                val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))
                val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy))
                if val_accuracy > min_accuracy:
                    save_path_model = (model_folder + '/model_crossModAtt_state_dict.pth')
                    torch.save(model.state_dict(), save_path_model)
                    min_accuracy = val_accuracy
            else:
                if (epoch + 1) % 10 == 0:
                    save_path_model = (model_folder + '/model_crossModAtt_state_dict_epoch' + str(epoch + 1) + '.pth')
                    torch.save(model.state_dict(), save_path_model)



    train_log_loss.close()
    train_log_acc.close()
    val_log_acc.close()
    val_log_loss.close()
    #writer.export_scalars_to_json(model_folder + "/all_scalars.json")
    writer.flush()
    writer.close()


def __main__(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--flowModel', type=str, default='./experiments/gtea61/flow/best_model_state_dict.pth',
                        help='Flow model')
    parser.add_argument('--rgbModel', type=str, default='./experiments/gtea61/rgb/best_model_state_dict.pth',
                        help='RGB model')
    parser.add_argument('--trainDatasetDir', type=str, default='./dataset/gtea_warped_flow_61/split2/train',
                        help='Train set directory')
    parser.add_argument('--valDatasetDir', type=str, default=None,
                        help='Validation set directory')
    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--stackSize', type=int, default=5, help='Number of opticl flow images in input')
    parser.add_argument('--trainBatchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--valBatchSize', type=int, default=32, help='Validation batch size')
    parser.add_argument('--numEpochs', type=int, default=250, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--stepSize', type=float, default=1, help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.99, help='Learning rate decay rate')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')

    args, _ = parser.parse_known_args(argv)

    dataset = args.dataset
    flowModel = args.flowModel
    rgbModel = args.rgbModel
    trainDatasetDir = args.trainDatasetDir
    valDatasetDir = args.valDatasetDir
    outDir = args.outDir
    stackSize = args.stackSize
    seqLen = args.seqLen
    trainBatchSize = args.trainBatchSize
    valBatchSize = args.valBatchSize
    numEpochs = args.numEpochs
    lr1 = args.lr
    stepSize = args.stepSize
    decayRate = args.decayRate
    memSize = args.memSize

    main_run(dataset, flowModel, rgbModel, stackSize, seqLen, memSize, trainDatasetDir, valDatasetDir, outDir,
             trainBatchSize, valBatchSize, lr1, numEpochs, stepSize, decayRate)

cross_mod_att = ["--trainDatasetDir", "./GTEA61",
              "--outDir", "outDir",
              "--valDatasetDir", "./GTEA61_val",
              "--stackSize", "5",
              "--seqLen", "7",
              "--numEpochs", "3",
              "--flowModel", "./drive/MyDrive/model_flow_state_dict_300.pth",
              "--rgbModel", "./drive/MyDrive//model_rgb_state_dict.pth",  
]

cmr.__main__(cross_mod_att)