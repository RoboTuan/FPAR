import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


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
            out = out + residual
            return outBN, out

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

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, noBN=False):
        self.inplanes = 64
        self.noBN = noBN
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.cma1 = self._make_cma_layer(cmaBlock, 256,stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, noBN=self.noBN)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_cma_layer(self,block,planes,stride=1):
        layers = []
        layers.append(cmaBlock(planes,stride))
        return nn.Sequential(*layers)

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

    def forward(self, y, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.cma1(x,y) 
        if self.noBN:
            conv_layer4BN, conv_layer4NBN = self.layer4(x)
        else:
            conv_layer4BN = self.layer4(x)

        # Debugging print to see if this avgpoll is a GAP
        # Yes, it's a GAP, because:
        # conv_layer4BN.size() is equal to [32, 512, 7, 7]
        # and the avgpool is performed with a kernel of 7x7,
        #print(conv_layer4BN.size())
        x = self.avgpool(conv_layer4BN)
        # x.size() after the GAP is equal to  [32, 512, 1, 1] as expected
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.noBN:
            return x, conv_layer4BN, conv_layer4NBN
        else:
            return x, conv_layer4BN

def resnet34(pretrained=False, noBN=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], noBN=noBN, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model



class doubleResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000,fl_num_classes=61, noBN=False,channels=10):
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
        self.cm_rgb_layer4 = self._make_layer(block, 512, layers[3], stride=2, noBN=self.noBN)
        self.cm_rgb_avgpool = nn.AvgPool2d(7, stride=1)
        self.cm_rgb_fc = nn.Linear(512 * block.expansion, num_classes)

        #2nd resnet34 for flow processing
        super(doubleResNet, self).__init__()
        self.cm_fl_conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.cm_fl_bn1 = nn.BatchNorm2d(64)
        self.cm_fl_relu = nn.ReLU(inplace=True)
        self.cm_fl_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cm_fl_layer1 = self._make_layer(block, 64, layers[0])
        self.cm_fl_layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.cm_fl_layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.cm_fl_cma1= cmaBlock(256, stride=1)
        self.cm_fl_layer4 = self._make_layer(block, 512, layers[3], stride=2)
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
        #initialize weight of cma_batchnorm as 0:
            
            #trovare nome

    def _make_layer(self, block, planes, blocks, stride=1):
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
        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        

        return nn.Sequential(*layers)

    def forward(self, y, x):
        x = self.conv1_cm_rgb(x)
        x = self.bn1_cm_rgb(x)
        x = self.relu_cm_rgb(x)
        x = self.maxpool_cm_rgb(x)

        x = self.layer1_cm_rgb(x)
        x = self.layer2_cm_rgb(x)
        x = self.layer3_cm_rgb(x)
        x = self.cma1_cm_rgb(x,y) 
        if self.noBN:
            conv_layer4BN, conv_layer4NBN = self.layer4_cm_rgb(x)
        else:
            conv_layer4BN = self.layer4_cm_rgb(x)
        # conv_layer4BN.size() is equal to [32, 512, 7, 7]
        # and the avgpool is performed with a kernel of 7x7,
        #print(conv_layer4BN.size())
        x = self.avgpool_cm_rgb(conv_layer4BN)
        # x.size() after the GAP is equal to  [32, 512, 1, 1] as expected
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.noBN:
            return x, conv_layer4BN, conv_layer4NBN
        else:
            return x, conv_layer4BN


def crossModresnet34(pretrained=False, noBN=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = doubleResNet(BasicBlock, [3, 4, 6, 3], noBN=noBN, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model

def resnet34(pretrained=False, noBN=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], noBN=noBN, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def flow_resnet34(pretrained=False, channels=20, num_classes=61):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], channels=channels, num_classes=num_classes)

    if pretrained:
        in_channels = channels
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = model.state_dict()

        new_pretrained_dict = change_key_names(pretrained_dict, in_channels)
        # 1. filter out unnecessary keys
        new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

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