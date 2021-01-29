from __future__ import print_function, division
from ML_DL_Project.Scripts.spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
import torch.nn as nn
from ML_DL_Project.Scripts.cmaModel import *
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from ML_DL_Project.Scripts.cma_makeDatasetTwoStream import *
import argparse

import sys


def main_run(dataset, stackSize, seqLen, memSize, trainDatasetDir, valDatasetDir, outDir,
             trainBatchSize, valBatchSize, lr1, numEpochs, decayRate, stepSize,flowModelDict,rgbModelDict):

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
                            shuffle=True, num_workers=4, pin_memory=True, drop_last= True)

    if valDatasetDir is not None:

        vid_seq_val = makeDataset2Stream(valDatasetDir,
                                    spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                                    sequence=True, numSeg=1, stackSize=stackSize, fmt='.png', phase='Test',
                                    seqLen=seqLen)

        val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize,
                                shuffle=False, num_workers=2, pin_memory=True,drop_last = True)
        valSamples = vid_seq_val.__len__()

        

    model = crossAttentionModel(stackSize=stackSize, memSize=memSize,
                                        num_classes=num_classes)
    model.Model.resNet.load_state_dict(torch.load(flowModelDict),strict = False)
    model.Model.load_state_dict(torch.load(rgbModelDict),strict=False)

    for params in model.parameters():
        params.requires_grad = False

    model.train(False)
    train_params = []
    cma_params = []
    rgb_params = [] 
    flow_params = []

    for params in model.classifier.parameters():
        params.requires_grad = True
        train_params += [params]
    
    for params in model.Model.lstm_cell.parameters():
        params.requires_grad = True
        rgb_params += [params]
    
    
    for params in model.Model.resNet.cm_rgb_cma1.parameters():
        params.requires_grad = True
        cma_params += [params]

    for params in model.Model.resNet.cm_rgb_layer4[0].conv1.parameters():
        params.requires_grad = True
        rgb_params += [params]

    for params in model.Model.resNet.cm_rgb_layer4[0].conv2.parameters():
        params.requires_grad = True
        rgb_params += [params]

    for params in model.Model.resNet.cm_rgb_layer4[1].conv1.parameters():
        params.requires_grad = True
        rgb_params += [params]

    for params in model.Model.resNet.cm_rgb_layer4[1].conv2.parameters():
        params.requires_grad = True
        rgb_params += [params]

    for params in model.Model.resNet.cm_rgb_layer4[2].conv1.parameters():
        params.requires_grad = True
        rgb_params += [params]
    #
    for params in model.Model.resNet.cm_rgb_layer4[2].conv2.parameters():
        params.requires_grad = True
        rgb_params += [params]
    #
    for params in model.Model.resNet.cm_rgb_fc.parameters():
        params.requires_grad = True
        rgb_params += [params]
    
 
    '''
    for params in model.Model.resNet.cm_fl_conv1.parameters():
        params.requires_grad = True
        base_params += [params]
    for params in model.Model.resNet.cm_fl_bn1.parameters():
        params.requires_grad = True
        base_params += [params]
    for params in model.Model.resNet.cm_fl_layer1.parameters():
        params.requires_grad = True
        base_params += [params]
    for params in model.Model.resNet.cm_fl_layer2.parameters():
        params.requires_grad = True
        base_params += [params]

    for params in model.Model.resNet.cm_fl_layer3.parameters():
        params.requires_grad = True
        base_params += [params]
    '''
    for params in model.Model.resNet.cm_fl_layer4.parameters():
        params.requires_grad = True
        flow_params += [params]
   

    for params in model.Model.resNet.cm_fl_cma1.parameters():
        params.requires_grad = True
        cma_params += [params]
    
    model = model.to(DEVICE)

    trainSamples = vid_seq_train.__len__()
    min_accuracy = 0

    loss_fn = nn.CrossEntropyLoss()
    optimizer_fn = torch.optim.SGD([
        {'params': rgb_params, 'lr': 1e-3},
        {'params': train_params, 'lr': 1e-3},
        {'params': flow_params, 'lr': 1e-3},
        {'params': cma_params, 'lr': 1e-3}
    ], lr=lr1, momentum=0.9, weight_decay=5e-4)

    optim_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_fn, step_size=stepSize, gamma=decayRate)
    
    train_iter = 0


    for epoch in range(numEpochs):
        epoch_loss = 0
        numCorrTrain = 0
        iterPerEpoch = 0
        model.classifier.train(True)
        model.Model.resNet.cm_rgb_layer4.train(True)
        model.Model.resNet.cm_rgb_fc.train(True)
        model.Model.resNet.cm_fl_layer4.train(True)
        model.Model.lstm_cell.train(True)
        model.Model.resNet.cm_rgb_cma1.train(True)
        model.Model.resNet.cm_fl_cma1.train(True)
        
        for j, (inputFlow, inputFrame, targets) in enumerate(train_loader):
            #print(f'Batch{j}')
            train_iter += 1
            iterPerEpoch += 1
            
            optimizer_fn.zero_grad()
            inputVariableFlow = inputFlow.permute(1,0,2,3,4).to(DEVICE)
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
        #print('Training average loss after {} epoch = {} '.format(epoch + 1, avg_loss))
        #print('Training accuracy after {} epoch = {}% '.format(epoch + 1, trainAccuracy))
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
                        inputVariableFlow = inputFlow.permute(1, 0, 2, 3, 4).to(DEVICE)
                        inputVariableFrame = inputFrame.permute(1, 0, 2, 3, 4).to(DEVICE)
                        labelVariable = targets.to(DEVICE)
                        output_label = model(inputVariableFlow, inputVariableFrame)
                        loss = loss_fn(F.log_softmax(output_label, dim=1), labelVariable)
                        val_loss_epoch += loss.item()
                        _, predicted = torch.max(output_label.data, 1)
                        numCorr += (predicted == labelVariable.data).sum()
                val_accuracy = torch.true_divide(numCorr, valSamples) * 100
                avg_val_loss = val_loss_epoch / val_iter
                #print('Val Loss after {} epochs, loss = {}'.format(epoch + 1, avg_val_loss))
                #print('Val Accuracy after {} epochs = {}%'.format(epoch + 1, val_accuracy))
                print('[{}/{}] [|TRAIN| Loss={:.3f} | Acc={:.3f}] [|VALID| Loss={:.3f} | Acc={:.3f}]'.format(epoch + 1,numEpochs,avg_loss,trainAccuracy, avg_val_loss,val_accuracy))
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
        else:   
            print('[{}/{}] [|TRAIN| Loss={:.3f} | Acc={:.3f}]'.format(epoch + 1,numEpochs, avg_loss,trainAccuracy))


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
    parser.add_argument('--flowModelDict', type=str, help='Flow model dict path')
    parser.add_argument('--rgbModelDict', type=str, help='rgb model dict path')

    args, _ = parser.parse_known_args(argv)

    dataset = args.dataset
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
    flowModelDict = args.flowModelDict
    rgbModelDict = args.rgbModelDict
    main_run(dataset, stackSize, seqLen, memSize, trainDatasetDir, valDatasetDir, outDir,
             trainBatchSize, valBatchSize, lr1, numEpochs, decayRate, stepSize, flowModelDict,rgbModelDict )
