import torch.nn as nn
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

    def __init__(self, inplanes, planes, stride=1):
        super(CmaBlock, self).__init__()
        self.noBN = noBN
        self.stride = stride
        self.convQ = conv1x1(inplanes, planes, stride)
        self.convK = conv1x1(inplanes, planes, stride)
        self.convV = conv1x1(inplanes, planes, stride)
        self.convZ = conv1x1(inplanes, planes, stride)
        self.softmax1 = torch.nn.Softmax()
        

    def forward(self, x, y):
        residual = x
        V = self.convV(y)
        Q = self.convQ(x)
        K = self.convK(y)
        M = self.softmax1(torch.matmul(Q,K))
        Z = torch.matmul(M,V)
        Z = self.convZ(Z)
        out = residual + Z
        out = self.relu(out)

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
        layers.append(cmaBlock(planes,planes,stride))
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

    def forward(self, x):
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


