import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random

from ML_DL_Project.Scripts.spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = Normalize(mean=mean, std=std)
spatial_transform2 = Compose([Scale((7,7)), ToTensor()]) 

def gen_split(root_dir, stackSize):
    Dataset = []
    Labels = []
    NumFrames = []
    Maps = []
    FramesMaps = []
    root_dir = os.path.join(root_dir, 'processed_frames2')
    # print for debugging
    #print(f"root_dir: {root_dir}")
    for dir_user in sorted(os.listdir(root_dir)):
      if not dir_user.startswith('.') and dir_user:
        class_id = 0
        directory = os.path.join(root_dir, dir_user)
        action = sorted(os.listdir(directory))
        # print for debugging
        #print(f"directory: {directory}")
        for target in sorted(os.listdir(directory)):
          if not target.startswith('.'):
            directory1 = os.path.join(directory, target)
            # print for debugging
            #print(f"directory1: {directory1}")
            insts = sorted(os.listdir(directory1))
            # print for debugging
            #print(f"insts: {insts}")
            if insts != []:
               for inst in insts:
                 if not inst.startswith('.'):
                   # adding "rgb" to path becasue after the number there are
                   # both "rgb" and "mmap" directories
                   root_position_for_map = os.path.join(directory1, inst)
                   # print for debugging
                   #print(f"root_position_for_map: {root_position_for_map}")
                   #print(f"root_position_for_map: {root_position_for_map}")

                   inst = inst + "/rgb"
                   inst_dir = os.path.join(directory1, inst)
                   # print for debugging
                   #print(f"inst_dir: {inst_dir}")
                   numFrames = len(glob.glob1(inst_dir, '*.png'))
                   # print for debugging
                   #print(f"numFrames: {numFrames}")
                   if numFrames >= stackSize:
                     Dataset.append(inst_dir)
                     Labels.append(class_id)
                     NumFrames.append(numFrames)
                   
                   #inst_dir = os.path.join(inst_dir, inst+"/mmaps") 
                   inst_dir_map = root_position_for_map + "/mmaps"
                   

                   #print(f"inst_dir_map: {inst_dir_map}")
                   #print(f"inst_dir_maps: {inst_dir_maps}")

                   numFramesMaps = len(glob.glob1(inst_dir_map, '*.png'))
                   #print(f"numFramesMaps: {numFramesMaps}")

                   # Non numMapFrames perchè non ci interessa
                   if numFrames >= stackSize:
                     Maps.append(inst_dir_map)
                     FramesMaps.append(numFramesMaps)

            class_id += 1
    return Dataset, Maps, Labels, NumFrames, FramesMaps, action

class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, seqLen=20,
                 train=True, stackSize=5, mulSeg=False, numSeg=1, fmt='.png'):

        self.images, self.maps, self.labels, self.numFrames, self.numMapFrames, self.action = gen_split(root_dir, stackSize)
        self.spatial_transform = spatial_transform
        self.train = train
        self.mulSeg = mulSeg
        self.numSeg = numSeg
        self.seqLen = seqLen
        # print for debugging
        #print(self.seqLen)
        self.fmt = fmt

    def __len__(self):
        return len(self.images)
    
    def __getLabel__(self):

      return self.action

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        map_name = self.maps[idx]

        label = self.labels[idx]
        numFrame = self.numFrames[idx]


        inpSeq = []
        mapSeq = []
         # For debugging
        #print(numFrame, self.seqLen)
        self.spatial_transform.randomize_parameters()
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
          #print(i)
          # Corrected with "rgb" instead of "image_" and zfill(4) instead of zfill(5)
          fl_name = vid_name + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
          #print(fl_name)
          img = Image.open(fl_name)
          # For debugging
          #print(img)

          flag=1
          j=i
          while(flag):
            maps_name = map_name + '/' + 'map' + str(int(np.floor(j))).zfill(4) + self.fmt
            #print(maps_name)
            try:
              mappa = Image.open(maps_name)
              flag=0
            except:
              if j<=i:
                j= 2*i-j+1 #j=i --> j=i +1 ; j=i-1 j-i=-1 --> j=i-(-1)+1
              else:
                j= 2*i-j #j=i+1 j-i=1 --> j=i-1
              continue
          inpSeq.append(self.spatial_transform(img.convert('RGB')))
          mapSeq.append(spatial_transform2(mappa.convert('L')))

        inpSeq = torch.stack(inpSeq, 0)
        mapSeq = torch.stack(mapSeq, 0)
        return inpSeq, mapSeq, label
      
     
