from __future__ import print_function, division
from ML_DL_Project.Scripts.spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
from ML_DL_Project.Scripts.makeMmaps import *
from ML_DL_Project.Scripts.makeDatasetTwoStream import *

from ML_DL_Project.Scripts.SelfSupObjectAttentionModelConvLSTM import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import argparse
import sys

def main_run(dataset, model_state_dict, dataset_dir, seqLen, stackSize, memSize, regression, Flow):

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

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    normalize = Normalize(mean=mean, std=std)
    spatial_transform = Compose([Scale(256), CenterCrop(224), ToTensor(), normalize])

    
    

    if Flow == True:
      vid_seq_test = makeDataset2Stream(dataset_dir, spatial_transform=spatial_transform, stackSize=stackSize, seqLen=seqLen, fmt='.png', selfSup=True)
    else:
      vid_seq_test = makeDataset(dataset_dir,
                               spatial_transform=spatial_transform,
                               seqLen=seqLen, fmt='.png')
      
    actions =vid_seq_test.__getLabel__()


    test_loader = torch.utils.data.DataLoader(vid_seq_test, batch_size=1,
                            shuffle=False, num_workers=2, pin_memory=True,drop_last=True)
    
    
    model = SelfSupAttentionModel(num_classes=num_classes, mem_size=memSize, REGRESSOR=regression,Flow=Flow)
    
    model.load_state_dict(torch.load(model_state_dict))

    for params in model.parameters():
        params.requires_grad = False

    model.train(False)
    model.to(DEVICE)
    test_samples = vid_seq_test.__len__()
    print('Number of samples = {}'.format(test_samples))
    print('Evaluating...')
    numCorr = 0
    true_labels = []
    predicted_labels = []
   
    
    with torch.no_grad():
      for inputs, inputMmap, targets in test_loader:

        inputVariable = inputs.permute(1, 0, 2, 3, 4).to(DEVICE)
        inputMmap = inputMmap.to(DEVICE)
        output_label, _ , mmapPrediction = model(inputVariable)

        if Flow == True:
          inputMmap = torch.reshape(inputMmap, (1, 14, 1, 7, 7))
        
        if regression==True:
          mmapPrediction = mmapPrediction.view(-1)
          #Regression -> float number for the input motion maps
          inputMmap = torch.reshape(inputMmap, (-1,)).float()                            
        else:
          mmapPrediction = mmapPrediction.view(-1,2)
          inputMmap = torch.reshape(inputMmap, (-1,))
          inputMmap = torch.round(inputMmap).long()

        _, predicted = torch.max(output_label.data, 1)
        numCorr += (predicted == targets.to(DEVICE)).sum()
        true_labels.append(targets)
        #.cpu() because confusion matrix is from scikit-learn
        predicted_labels.append(predicted.cpu())
            
            
    test_accuracy = (numCorr / test_samples) * 100
    print('Test Accuracy = {}%'.format(test_accuracy))     

    # ebug
    print(true_labels)
    print(predicted_labels)

    cnf_matrix = confusion_matrix(true_labels, predicted_labels).astype(float)
    cnf_matrix_normalized = cnf_matrix / cnf_matrix.sum(axis=1)[:, np.newaxis]
    if Flow ==True:

      ticks = [str(action + str(i) ) for i, action in enumerate(actions)]
      plt.figure(figsize=(20,20))
      plt.imshow(cnf_matrix_normalized, interpolation='none', cmap='YlOrRd')
      plt.colorbar()
      plt.xticks(np.arange(num_classes),labels = set(ticks), fontsize=10, rotation = 90)
      plt.yticks(np.arange(num_classes),labels = set(ticks), fontsize=10)
      plt.xlabel("Predicted")
      plt.ylabel("True")
      plt.grid(True)
      plt.clim(0, 1)
      plt.savefig(dataset + '-selfSupFlow.jpg', bbox_inches='tight')
      plt.show()
    else:
      ticks = [str(action + str(i) ) for i, action in enumerate(actions)]
      plt.figure(figsize=(20,20))
      plt.imshow(cnf_matrix_normalized, interpolation='none', cmap='Reds')
      plt.colorbar()
      plt.xticks(np.arange(num_classes),labels = set(ticks), fontsize=10, rotation = 90)
      plt.yticks(np.arange(num_classes),labels = set(ticks), fontsize=10)
      plt.xlabel("Predicted")
      plt.ylabel("True")
      plt.grid(True)
      plt.clim(0, 1)
      plt.savefig(dataset + '-selfSup.jpg', bbox_inches='tight')
      plt.show()


def __main__(argv=None):
    parser = argparse.ArgumentParser(prog='myprogram', description='Foo')
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--datasetDir', type=str, default=None,
                        help='Test set directory')
    parser.add_argument('--model_state_dict', type=str, default='./experiments/gtea61/rgb/stage1/best_model_state_dict.pth',
                        help='Model to test path')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--stackSize', type=int, default=5, help='Number of opticl flow images in input')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')
    parser.add_argument('--regression', type=bool, default=True, help='Do the motion segmentation task (selfSup) with regression ')
    parser.add_argument('--Flow', type=bool, default=False, help='Use warp flow or improved dense trajectories ')


    args, _ = parser.parse_known_args(argv)
    
    dataset = args.dataset
    model_state_dict = args.model_state_dict
    dataset_dir = args.datasetDir
    seqLen = args.seqLen
    stackSize = args.stackSize
    memSize = args.memSize
    regression= args.regression
    Flow= args.Flow

    main_run(dataset, model_state_dict, dataset_dir, seqLen, memSize, regression, Flow)
