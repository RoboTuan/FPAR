import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
import copy

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
                     padding=0, bias=False)

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
            return outBN,out

class BasicBlockFlow(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockFlow, self).__init__()
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
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class cmaBlock(nn.Module):
    expansion = 1

    def __init__(self, planes, stride=1):
        super(cmaBlock, self).__init__()
        self.stride = stride
        self.convQ = conv1x1(planes, int(planes/2), stride)
        self.convK = conv1x1(planes, int(planes/2), stride)
        self.convV = conv1x1(planes, int(planes/2), stride)
        self.convZ = conv1x1(int(planes/2), planes, stride)
        self.softmax2d = torch.nn.Softmax2d()
        self.bn = nn.BatchNorm2d(planes)
        

    def forward(self, x, y):
        residual = x
        V = self.convV(y)
        V = torch.flatten(V,2,3)
        Q = self.convQ(x)
        Q = torch.flatten(Q,2,3)
        K = self.convK(y)
        K = torch.flatten(K,2,3)
        M = torch.matmul(Q.permute(0,2,1),K)
        M = torch.unsqueeze(M,1)
        M = self.softmax2d(M)
        M = torch.squeeze(M)
        #print(f'M{M.size()},V{V.size()}')
        Z = torch.matmul(V,M) 
        #print(f'Z{Z.size()}')
        Z = torch.reshape(Z,(-1,128,14,14))
        #print(f'Z{Z.size()}')
        Z = self.convZ(Z)
        Z = self.bn(Z)
        out = residual + Z

        return out

class doubleResNet(nn.Module):

    def __init__(self, block, flowblock, layers, num_classes=1000,fl_num_classes=61,noBN=False,channels=10):
        self.inplanes = 64
        self.inplanes_flow = 64
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
        self.cm_fl_layer1 = self._make_layer_flow(flowblock, 64, layers[0])
        self.cm_fl_layer2 = self._make_layer_flow(flowblock, 128, layers[1], stride=2)
        self.cm_fl_layer3 = self._make_layer_flow(flowblock, 256, layers[2], stride=2)
        self.cm_fl_cma1= cmaBlock(256, stride=1)
        self.cm_fl_layer4 = self._make_layer_flow(flowblock, 512, layers[3], stride=2)
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

        for m in self.modules(): ##initialize weights and bias of batch norm of cma blocs to 0
            if isinstance(m, cmaBlock):
              for mm in m.modules():
                if isinstance(mm, nn.BatchNorm2d):                 
                  with torch.no_grad():
                    mm.weight.zero_()
                    mm.bias.zero_()



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
        if stride != 1 or self.inplanes_flow != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_flow, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_flow, planes, stride, downsample))
        self.inplanes_flow = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_flow, planes))

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
        y = self.cm_fl_layer3(y)

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
        x = self.cm_rgb_avgpool(conv_layer4BN)
        y = self.cm_fl_avgpool(y)

        # x.size() after the GAP is equal to  [32, 512, 1, 1] as expected
        #print(x.size())
        x = x.view(x.size(0), -1)
        y1 = y.view(y.size(0),-1)

        y = self.dp(y1)

        x = self.cm_rgb_fc(x)
        y = self.cm_fl_fc(y)

        if self.noBN:
            return x, conv_layer4BN, conv_layer4NBN ,y1
        else:
            return x, conv_layer4BN, y1

def crossModresnet34(pretrained=False, noBN=False, **kwargs):

    model = doubleResNet(BasicBlock, BasicBlockFlow, [3, 4, 6, 3], noBN=noBN, **kwargs)
    if pretrained == True:
        resPretrained=model_zoo.load_url(model_urls['resnet34'])

        cma_dict = copy.deepcopy(resPretrained)

        for key in resPretrained:
            cma_dict['cm_rgb_' + key] = cma_dict.pop(key)



        for key,value in resPretrained.items():
            if (key != "fc.weight" and key!="fc.bias"):
                if(key == "conv1.weight"):
                    rgb_weight = value
                    rgb_weight_mean = torch.mean(rgb_weight, dim=1)
                    flow_weight = rgb_weight_mean.unsqueeze(1).repeat(1, 10, 1, 1)
                    cma_dict['cm_fl_' + key] = flow_weight

                else:
                    cma_dict['cm_fl_'+key] = resPretrained[key]

        model.load_state_dict(cma_dict,strict=False)

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