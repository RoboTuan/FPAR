from __future__ import print_function, division
from ML_DL_Project.Scripts.attentionModelLSTA import *
from ML_DL_Project.Scripts.spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
from torch.utils.tensorboard import SummaryWriter
from ML_DL_Project.Scripts.makeDatasetRGB import *
import sys
import argparse
import os


#TODO: create separate dirs for stage1 and stage 2

# def main_run(dataset, stage, root_dir, out_dir, seqLen, trainBatchSize, numEpochs, lr1, decay_factor,
#              decay_step, memSize, outPool_size, split, evalInterval):
def main_run(dataset, stage, train_data_dir, val_data_dir, stage1_dict, out_dir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1, decayRate, stepSize, memSize, outPool_size):


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
    
    DEVICE = "cuda"
    c_cam_classes = outPool_size

    model_folder = os.path.join('./', out_dir, dataset, 'rgb_lsta', 'stage'+str(stage))

    if not os.path.exists(model_folder):
        print('Directory {} exitst!'.format(model_folder))
        sys.exit()
    os.makedirs(model_folder)


    # Log files
    writer = SummaryWriter(model_folder)
    train_log_loss = open((model_folder + '/train_log_loss.txt'), 'w')
    train_log_acc = open((model_folder + '/train_log_acc.txt'), 'w')
    val_log_loss = open((model_folder + '/val_log_loss.txt'), 'w')
    val_log_acc = open((model_folder + '/val_log_acc.txt'), 'w')




    # stage = stage
    # test_split = split
    # seqLen = seqLen
    # memSize = memSize
    # c_cam_classes = outPool_size
    # dataset = dataset
    # best_acc = 0

    # if stage == 1:
    #     trainBatchSize = trainBatchSize
    #     testBatchSize = trainBatchSize
    #     lr1 = lr1
    #     decay_factor = decay_factor
    #     decay_step = decay_step
    #     numEpochs = numEpochs
    # elif stage == 2:
    #     trainBatchSize = trainBatchSize
    #     testBatchSize = trainBatchSize
    #     lr1 = lr1
    #     decay_factor = decay_factor
    #     decay_step = decay_step
    #     numEpochs = numEpochs




    # Data loader

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    normalize = Normalize(mean=mean, std=std)


    spatial_transform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
                                 ToTensor(), normalize])

    vid_seq_train = makeDataset(train_data_dir,
                                spatial_transform=spatial_transform, fmt='.png', seqLen=seqLen)

    print('Number of train samples = {}'.format(vid_seq_train.__len__()))
    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize, shuffle=True, num_workers=4, pin_memory=True)


    if val_data_dir is not None:
        vid_seq_val = makeDataset(val_data_dir, spatial_transform=Compose([Scale(256),
                                                                            CenterCrop(224),
                                                                            ToTensor(),
                                                                            normalize]),
                                fmt='.png', seqLen=seqLen)

        print('Number of test samples = {}'.format(vid_seq_val.__len__()))
        val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize, shuffle=False, num_workers=2, pin_memory=True)


    train_params = []
    if stage == 1:
        model = attentionModelLSTA(num_classes=num_classes, mem_size=memSize, c_cam_classes=c_cam_classes)
        model.train(False)
        for params in model.parameters():
            params.requires_grad = False
    elif stage == 2:
        model = attentionModelLSTA(num_classes=num_classes, mem_size=memSize, c_cam_classes=c_cam_classes)
        
        model.load_state_dict(torch.load(stage1_dict))
        model.train(False)

        for params in model.parameters():
            params.requires_grad = False

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

        for params in model.resNet.layer4[2].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.fc.parameters():
            params.requires_grad = True
            train_params += [params]

    for params in model.lsta_cell.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.classifier.parameters():
        params.requires_grad = True
        train_params += [params]

    model.classifier.train(True)
    model = model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    optimizer_fn = torch.optim.Adam(train_params, lr=lr1, weight_decay=5e-4, eps=1e-4)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=stepSize, gamma=decayRate)

    train_iter = 0
    min_accuracy = 0

    for epoch in range(numEpochs):
        optim_scheduler.step()
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        model.classifier.train(True)
        writer.add_scalar('lr', optimizer_fn.param_groups[0]['lr'], epoch+1)
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
            epoch_loss += loss.item()
        avg_loss = epoch_loss/iterPerEpoch
        trainAccuracy = torch.true_divide(numCorrTrain / trainSamples) * 100
        print('Train: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch+1, avg_loss, trainAccuracy))
        writer.add_scalar('train/epoch_loss', avg_loss, epoch+1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch+1)
        train_log_loss.write('train Loss after {} epochs = {}\n'.format(epoch + 1, avg_loss))
        train_log_acc.write('train Accuracy after {} epochs = {}%\n'.format(epoch + 1, trainAccuracy))

        # save_path_model = os.path.join(model_folder, 'last_checkpoint_stage' + str(stage) + '.pth.tar')
        # save_file = {
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer_fn.state_dict(),
        #         'best_acc': best_acc,
        #     }
        # torch.save(save_file, save_path_model)

        if val_data_dir is not None:
            if (epoch+1) % evalInterval == 0:
                model.train(False)
                val_loss_epoch = 0
                val_iter = 0
                val_samples = 0
                numCorr = 0
                with torch.no_grad():
                    for j, (inputs, targets) in enumerate(val_loader):
                        val_iter += 1
                        val_samples += inputs.size(0)
                        inputVariable = inputs.permute(1, 0, 2, 3, 4).to(DEVICE)
                        labelVariable = targets.to(DEVICE)
                        output_label, _ = model(inputVariable)
                        val_loss = loss_fn(output_label, labelVariable)
                        val_loss_epoch += val_loss.item()
                        _, predicted = torch.max(output_label.data, 1)
                        numCorr += (predicted == targets.cuda()).sum()

                val_accuracy = torch.true_divide(numCorr / val_samples) * 100
                avg_val_loss = val_loss_epoch / val_iter
                print('Val: Epoch = {} | Loss {} | Accuracy = {}'.format(epoch + 1, avg_val_loss, val_accuracy))
                writer.add_scalar('val/epoch_loss', avg_val_loss, epoch + 1)
                writer.add_scalar('val/accuracy', val_accuracy, epoch + 1)
                val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))
                val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy))

                if val_accuracy > min_accuracy:
                    min_accuracy = val_accuracy
                    save_path_model = (model_folder + '/model_rgb_lsta_state_dict.pth')
                    torch.save(model.state_dict(), save_path_model)
                    min_accuracy = val_accuracy
            else:
                if (epoch+1) % 10 == 0:
                    save_path_model = (model_folder + '/model_rgb_lsta_state_dict_epoch' + str(epoch+1) + '.pth')
                    torch.save(model.state_dict(), save_path_model)



    train_log_loss.close()
    train_log_acc.close()
    val_log_acc.close()
    val_log_loss.close()
    writer.flush()
    writer.close()


def __main__(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--stage', type=int, default=1, help='Training stage')
    parser.add_argument('--trainDatasetDir', type=str, default='./dataset/gtea_warped_flow_61/split2/train',
                        help='Train set directory')
    parser.add_argument('--valDatasetDir', type=str, default=None,
                        help='Val set directory')
    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--trainBatchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--valBatchSize', type=int, default=64, help='Validation batch size')
    parser.add_argument('--numEpochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--stepSize', type=float, default=[25, 75, 150], nargs="+", help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')
    parser.add_argument('--outPoolSize', type=int, default=100, help='Output pooling size')
    parser.add_argument('--evalInterval', type=int, default=1, help='Evaluation interval')


    args, _ = parser.parse_known_args(argv)

    dataset = args.dataset
    stage = args.stage
    trainDatasetDir = args.trainDatasetDir
    valDatasetDir = args.valDatasetDir
    outDir = args.outDir
    seqLen = args.seqLen
    trainBatchSize = args.trainBatchSize
    valBatchSize = args.valBatchSize
    numEpochs = args.numEpochs
    lr1 = args.lr
    stepSize = args.stepSize
    decayRate = args.decayRate
    memSize = args.memSize
    outPool_size = args.outPoolSize
    evalInterval = args.evalInterval

    # def main_run(dataset, stage, train_data_dir, val_data_dir, stage1_dict, out_dir, seqLen, trainBatchSize,
    #          valBatchSize, numEpochs, lr1, decayRate, stepSize, memSize, outPool_size):


    main_run(dataset=dataset, stage=stage, train_data_dir=trainDatasetDir, valDatasetDir=valDatasetDir, out_dir=outDir, seqLen=seqLen,
             trainBatchSize=trainBatchSize, valBatchSize=valBatchSize, numEpochs=numEpochs, lr1=lr1, decay_factor=decayRate,
             decay_step=stepSize, memSize=memSize, outPool_size=outPool_size, evalInterval=evalInterval)