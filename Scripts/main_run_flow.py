from __future__ import print_function, division
from ML_DL_Project.Scripts.flow_resnet import *
from ML_DL_Project.Scripts.spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from ML_DL_Project.Scripts.makeDatasetFlow import *
import argparse
import sys

def main_run(dataset, stage1_dict, pretrained_first_stage, train_data_dir, val_data_dir, out_dir, stackSize, trainBatchSize, valBatchSize, numEpochs, lr1,
             decayRate, stepSize, stage=2):


    if dataset == 'gtea61':
        if stage == 1:
            num_classes = 10
        else:
            num_classes = 61
    elif dataset == 'gtea71':
        if stage == 1:
            num_classes = 10
        else:
            num_classes = 71
    elif dataset == 'gtea_gaze':
        if stage == 1:
            num_classes = 10
        else:
            num_classes = 44
    elif dataset == 'egtea':
        if stage == 1:
            num_classes = 10
        else:
            num_classes = 106
    else:
        print('Dataset not found')
        sys.exit()

    # Setting Device
    DEVICE = "cuda"

    model_folder = os.path.join('./', out_dir, dataset, 'flow', 'stage'+str(stage))
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


    # Data loader
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    spatial_transform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
                                ToTensor(), normalize])




    if stage == 1:
        vid_seq_train = makeDatasetFlow(train_data_dir, spatial_transform=spatial_transform, sequence=False,
                                    stackSize=stackSize, fmt='.png', LSTA=True)

        train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                                shuffle=True, sampler=None, num_workers=4, pin_memory=True)
        if val_data_dir is not None:

            vid_seq_val = makeDatasetFlow(val_data_dir, spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                                        sequence=False, stackSize=stackSize, fmt='.png', phase='Test', LSTA=True)

            val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize, shuffle=False, num_workers=2, pin_memory=True)
            valInstances = vid_seq_val.__len__()    
            trainInstances = vid_seq_train.__len__()
            print('Number of samples in the dataset: training = {} | validation = {}'.format(trainInstances, valInstances))


    else:

        vid_seq_train = makeDatasetFlow(train_data_dir, spatial_transform=spatial_transform, sequence=False,
                                    stackSize=stackSize, fmt='.png')

        train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                                shuffle=True, sampler=None, num_workers=4, pin_memory=True)
        if val_data_dir is not None:

            vid_seq_val = makeDatasetFlow(val_data_dir, spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                                        sequence=False, stackSize=stackSize, fmt='.png', phase='Test')

            val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize, shuffle=False, num_workers=2, pin_memory=True)
            valInstances = vid_seq_val.__len__()


            trainInstances = vid_seq_train.__len__()
            print('Number of samples in the dataset: training = {} | validation = {}'.format(trainInstances, valInstances))

    if stage == 1:
        model = flow_resnet34(True, channels=2*stackSize, num_classes=num_classes)
    else:
        if pretrained_first_stage is True:
            model = flow_resnet34(True, channels=2*stackSize, num_classes=10)
            #print(model)
            #sys.exit()
            model.load_state_dict(torch.load(stage1_dict))
            # In the flow_resnet script we have:
            # self.fc_action = nn.Linear(512 * block.expansion, num_classes) at line 113.
            # Since "block" is an instance of the class BasicBlock,
            #  exapansion is set to 1 at line 28 (in that class)
            model.fc_action = nn.Linear(512, 61)
        else:
            model = flow_resnet34(True, channels=2*stackSize, num_classes=num_classes)

    model.train(True)
    train_params = list(model.parameters())

    model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    optimizer_fn = torch.optim.SGD(train_params, lr=lr1, momentum=0.9, weight_decay=5e-4)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=stepSize, gamma=decayRate)

    min_accuracy = 0
    train_iter = 0

    for epoch in range(numEpochs):
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        model.train(True)
        writer.add_scalar('lr', optimizer_fn.param_groups[0]['lr'], epoch+1)
        for i, (inputs, targets) in enumerate(train_loader):
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            inputVariable = inputs.to(DEVICE)
            labelVariable = targets.to(DEVICE)
            trainSamples += inputs.size(0)
            output_label, _ = model(inputVariable)
            loss = loss_fn(output_label, labelVariable)
            loss.backward()
            optimizer_fn.step()
            _, predicted = torch.max(output_label.data, 1)
            numCorrTrain += (predicted == targets.cuda()).sum()
            epoch_loss += loss.item()
        avg_loss = epoch_loss/iterPerEpoch
        trainAccuracy = torch.true_divide(numCorrTrain, trainSamples) * 100
        print('Train: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch + 1, avg_loss, trainAccuracy))
        writer.add_scalar('train/epoch_loss', avg_loss, epoch+1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch+1)
        train_log_loss.write('Training loss after {} epoch = {}\n'.format(epoch+1, avg_loss))
        train_log_acc.write('Training accuracy after {} epoch = {}\n'.format(epoch+1, trainAccuracy))
        optim_scheduler.step()
        
        if val_data_dir is not None:
            if (epoch+1) % 1 == 0:
                model.train(False)
                val_loss_epoch = 0
                val_iter = 0
                val_samples = 0
                numCorr = 0
                # wrapping with torch.no_grad() because it wasn't present, see issuea with volatie=True
                # volatile keyword is deprecated, check is it's correct
                with torch.no_grad():
                    for j, (inputs, targets) in enumerate(val_loader):
                        val_iter += 1
                        val_samples += inputs.size(0)
                        inputVariable = inputs.to(DEVICE)
                        #inputVariable = Variable(inputs.cuda(), volatile=True)
                        #labelVariable = Variable(targets.cuda(async=True), volatile=True)
                        #vedere se "non_blockign=True" va bene
                        labelVariable = targets.to(DEVICE, non_blocking=True)
                        output_label, _ = model(inputVariable)
                        val_loss = loss_fn(output_label, labelVariable)
                        val_loss_epoch += val_loss.item()
                        _, predicted = torch.max(output_label.data, 1)
                        numCorr += (predicted == targets.cuda()).sum()
                val_accuracy = torch.true_divide(numCorr, val_samples) * 100
                avg_val_loss = val_loss_epoch / val_iter
                print('Validation: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch + 1, avg_val_loss, val_accuracy))
                writer.add_scalar('val/epoch_loss', avg_val_loss, epoch + 1)
                writer.add_scalar('val/accuracy', val_accuracy, epoch + 1)
                val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))
                val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy))
                if val_accuracy > min_accuracy:
                    save_path_model = (model_folder + '/model_flow_state_dict.pth')
                    torch.save(model.state_dict(), save_path_model)
                    min_accuracy = val_accuracy
            else:
                if (epoch+1) % 10 == 0:
                    save_path_model = (model_folder + '/model_flow_state_dict_epoch' + str(epoch+1) + '.pth')
                    torch.save(model.state_dict(), save_path_model)

    train_log_loss.close()
    train_log_acc.close()
    val_log_acc.close()
    val_log_loss.close()
    #writer.export_scalars_to_json(model_folder + "/all_scalars.json")
    writer.flush()
    writer.close()



# Added argv as input
def __main__(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--stage1Dict', type=str, default='./experiments/gtea61/rgb/stage1/best_model_state_dict.pth', 
                        help='Stage 1 model path')
    parser.add_argument('--pretrained_first_stage', type=bool, default=False)
    parser.add_argument('--trainDatasetDir', type=str, default='./dataset/gtea_warped_flow_61/split2/train',
                        help='Train set directory')
    parser.add_argument('--valDatasetDir', type=str, default=None,
                        help='Validation set directory')
    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--stackSize', type=int, default=5, help='Length of sequence')
    parser.add_argument('--trainBatchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--valBatchSize', type=int, default=32, help='Validation batch size')
    parser.add_argument('--numEpochs', type=int, default=750, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--stepSize', type=float, default=[150, 300, 500], nargs="+", help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.5, help='Learning rate decay rate')

    #args = parser.parse_args()

    # Added args, _ = parser.parse_known_args() for colab parses issues
    # THIS FIXED THE PROBLEM
    # added argv, see input of main
    args, _ = parser.parse_known_args(argv)

    dataset = args.dataset
    trainDatasetDir = args.trainDatasetDir
    valDatasetDir = args.valDatasetDir
    outDir = args.outDir
    stage1Dict = args.stage1Dict
    pretrained_first_stage = args.pretrained_first_stage
    stackSize = args.stackSize
    trainBatchSize = args.trainBatchSize
    valBatchSize = args.valBatchSize
    numEpochs = args.numEpochs
    lr1 = args.lr
    stepSize = args.stepSize
    decayRate = args.decayRate

    main_run(dataset, stage1Dict, pretrained_first_stage, trainDatasetDir, valDatasetDir, outDir, stackSize,
             trainBatchSize, valBatchSize, numEpochs, lr1, decayRate, stepSize)

#__main__()
