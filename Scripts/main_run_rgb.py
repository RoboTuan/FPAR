from __future__ import print_function, division
# This is without attention, we must address this better
#from ML_DL_Project.Scripts.convLSTMmodel import *
from ML_DL_Project.Scripts.objectAttentionModelConvLSTM import *
from ML_DL_Project.Scripts.spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
from torch.utils.tensorboard import SummaryWriter
from ML_DL_Project.Scripts.makeDatasetRGB import *
import argparse
import sys


def main_run(dataset, stage, train_data_dir, val_data_dir, stage1_dict, out_dir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1, decayRate, stepSize, memSize):

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

    model_folder = os.path.join('./', out_dir, dataset, 'rgb', 'stage'+str(stage))  # Dir for saving models and log files
    # Create the dir
    # TODO:
    # see if is necessary other if as in colab
    if os.path.exists(model_folder):
        print('Directory {} exists!'.format(model_folder))
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

    vid_seq_train = makeDataset(train_data_dir,
                                spatial_transform=spatial_transform, seqLen=seqLen, fmt='.png')

    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                            shuffle=True, num_workers=4, pin_memory=True)
    if val_data_dir is not None:
        vid_seq_val = makeDataset(val_data_dir,spatial_transform = Compose([Scale(256),
                                                                            CenterCrop(224),
                                                                            ToTensor(),
                                                                            normalize]),
                                    seqLen=seqLen, fmt='.png')

        val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize, shuffle=False, num_workers=2, pin_memory=True)
        valInstances = vid_seq_val.__len__()
        
        trainInstances = vid_seq_train.__len__()


    train_params = []
    if stage == 1:
        
        # DO this fo no attention, we must address it better
        #model = clstm_Model(num_classes=num_classes, mem_size=memSize)
        model = attentionModel(num_classes=num_classes, mem_size=memSize)
        model.train(False)
        for params in model.parameters():
            params.requires_grad = False
    else:

        # DO this fo no attention, we must address it better
        #model = clstm_Model(num_classes=num_classes, mem_size=memSize)
        model = attentionModel(num_classes=num_classes, mem_size=memSize)
        model.load_state_dict(torch.load(stage1_dict))
        model.train(False)
        for params in model.parameters():
            params.requires_grad = False
        #
        for params in model.resNet.layer4[0].conv1.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[0].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[1].conv1.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[1].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[2].conv1.parameters():
            params.requires_grad = True
            train_params += [params]
        #
        for params in model.resNet.layer4[2].conv2.parameters():
            params.requires_grad = True
            train_params += [params]
        #
        for params in model.resNet.fc.parameters():
            params.requires_grad = True
            train_params += [params]

        model.resNet.layer4[0].conv1.train(True)
        model.resNet.layer4[0].conv2.train(True)
        model.resNet.layer4[1].conv1.train(True)
        model.resNet.layer4[1].conv2.train(True)
        model.resNet.layer4[2].conv1.train(True)
        model.resNet.layer4[2].conv2.train(True)
        model.resNet.fc.train(True)

    for params in model.lstm_cell.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.classifier.parameters():
        params.requires_grad = True
        train_params += [params]


    model.lstm_cell.train(True)

    model.classifier.train(True)

    model = model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    optimizer_fn = torch.optim.Adam(train_params, lr=lr1, weight_decay=4e-5, eps=1e-4)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=stepSize, gamma=decayRate)


    train_iter = 0
    min_accuracy = 0


    for epoch in range(numEpochs):
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        model.lstm_cell.train(True)
        model.classifier.train(True)
        writer.add_scalar('lr', optimizer_fn.param_groups[0]['lr'], epoch+1)
        if stage == 2:
            model.resNet.layer4[0].conv1.train(True)
            model.resNet.layer4[0].conv2.train(True)
            model.resNet.layer4[1].conv1.train(True)
            model.resNet.layer4[1].conv2.train(True)
            model.resNet.layer4[2].conv1.train(True)
            model.resNet.layer4[2].conv2.train(True)
            model.resNet.fc.train(True)
        for i, (inputs, targets) in enumerate(train_loader):
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            inputVariable = Variable(inputs.permute(1, 0, 2, 3, 4).to(DEVICE))
            labelVariable = Variable(targets.to(DEVICE))
            trainSamples += inputs.size(0)
            output_label, _ = model(inputVariable)
            loss = loss_fn(output_label, labelVariable)
            loss.backward()
            optimizer_fn.step()
            _, predicted = torch.max(output_label.data, 1)
            numCorrTrain += (predicted == targets.to(DEVICE)).sum()
            # see if loss.item() has to be multiplied by inputs.size(0)
            epoch_loss += loss.item()
        avg_loss = epoch_loss/iterPerEpoch
        # This is deprecated, see if the below "torch.true_divide" is correct
        #trainAccuracy =  (numCorrTrain / trainSamples) * 100
        trainAccuracy =  torch.true_divide(numCorrTrain, trainSamples) * 100
        optim_scheduler.step()

        print('Train: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch+1, avg_loss, trainAccuracy))
        writer.add_scalar('train/epoch_loss', avg_loss, epoch+1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch+1)
        train_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_loss))
        train_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, trainAccuracy))
        if val_data_dir is not None:
            if (epoch+1) % 1 == 0:
                model.train(False)
                val_loss_epoch = 0
                val_iter = 0
                val_samples = 0
                numCorr = 0
                with torch.no_grad():
                    for j, (inputs, targets) in enumerate(val_loader):
                        val_iter += 1
                        val_samples += inputs.size(0)
                        # Deprecated
                        #inputVariable = Variable(inputs.permute(1, 0, 2, 3, 4).cuda(), volatile=True)
                        #labelVariable = Variable(targets.cuda(async=True), volatile=True)
                        inputVariable = inputs.permute(1, 0, 2, 3, 4).to(DEVICE)
                        labelVariable = targets.to(DEVICE)
                        output_label, _ = model(inputVariable)
                        val_loss = loss_fn(output_label, labelVariable)
                        val_loss_epoch += val_loss.item()
                        _, predicted = torch.max(output_label.data, 1)
                        numCorr += (predicted == targets.cuda()).sum()
                # This is deprecated, see if the below "torch.true_divide" is correct
                #val_accuracy = (numCorr / val_samples) * 100
                val_accuracy = torch.true_divide(numCorr, val_samples) * 100
                avg_val_loss = val_loss_epoch / val_iter
                print('Val: Epoch = {} | Loss {} | Accuracy = {}'.format(epoch + 1, avg_val_loss, val_accuracy))
                writer.add_scalar('val/epoch_loss', avg_val_loss, epoch + 1)
                writer.add_scalar('val/accuracy', val_accuracy, epoch + 1)
                val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))
                val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy))
                if val_accuracy > min_accuracy:
                    save_path_model = (model_folder + '/model_rgb_state_dict.pth')
                    torch.save(model.state_dict(), save_path_model)
                    min_accuracy = val_accuracy
            else:
                if (epoch+1) % 10 == 0:
                    save_path_model = (model_folder + '/model_rgb_state_dict_epoch' + str(epoch+1) + '.pth')
                    torch.save(model.state_dict(), save_path_model)

    train_log_loss.close()
    train_log_acc.close()
    val_log_acc.close()
    val_log_loss.close()
    #writer.export_scalars_to_json(model_folder + "/all_scalars.json")
    writer.flush()
    writer.close()

# Renamed main
# def __main__():
# Added argv as input
def __main__(argv=None):
    # Added prog='myprogram', description='Foo' for colab parses issues
    # THIS DIDN'T FIX THE PROBLEM
    parser = argparse.ArgumentParser(prog='myprogram', description='Foo')
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--stage', type=int, default=1, help='Training stage')
    parser.add_argument('--trainDatasetDir', type=str, default='./dataset/gtea_warped_flow_61/split2/train',
                        help='Train set directory')
    parser.add_argument('--valDatasetDir', type=str, default=None,
                        help='Val set directory')
    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--stage1Dict', type=str, default='./experiments/gtea61/rgb/stage1/best_model_state_dict.pth',
                        help='Stage 1 model path')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--trainBatchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--valBatchSize', type=int, default=64, help='Validation batch size')
    parser.add_argument('--numEpochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--stepSize', type=float, default=[25, 75, 150], nargs="+", help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')

    #args = parser.parse_args()

    # Added args, _ = parser.parse_known_args() for colab parses issues
    # THIS FIXED THE PROBLEM
    # added argv, see input of main
    args, _ = parser.parse_known_args(argv)

    dataset = args.dataset
    stage = args.stage
    trainDatasetDir = args.trainDatasetDir
    valDatasetDir = args.valDatasetDir
    outDir = args.outDir
    stage1Dict = args.stage1Dict
    seqLen = args.seqLen
    trainBatchSize = args.trainBatchSize
    valBatchSize = args.valBatchSize
    numEpochs = args.numEpochs
    lr1 = args.lr
    stepSize = args.stepSize
    decayRate = args.decayRate
    memSize = args.memSize

    main_run(dataset, stage, trainDatasetDir, valDatasetDir, stage1Dict, outDir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1, decayRate, stepSize, memSize)

# Don't run it automatically for colab
#__main__()

#if __name__ == "__main__":
#    main()


