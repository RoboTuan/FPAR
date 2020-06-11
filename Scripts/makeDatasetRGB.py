import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random


def gen_split(root_dir, stackSize):
    Dataset = []
    Labels = []
    NumFrames = []
    root_dir = os.path.join(root_dir, 'processed_frames2')
    # print for debugging
    #print(f"root_dir: {root_dir}")
    for dir_user in sorted(os.listdir(root_dir)):
      if not dir_user.startswith('.') and dir_user != "S2" :
        class_id = 0
        directory = os.path.join(root_dir, dir_user)
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
            class_id += 1
    return Dataset, Labels, NumFrames

class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, seqLen=20,
                 train=True, mulSeg=False, numSeg=1, fmt='.png'):

        self.images, self.labels, self.numFrames = gen_split(root_dir, 5)
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

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeq = []
         # For debugging
        #print(numFrame, self.seqLen)
        self.spatial_transform.randomize_parameters()
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
          # Corrected with "rgb" instead of "image_" and zfill(4) instead of zfill(5)
          fl_name = vid_name + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
          img = Image.open(fl_name)
          # For debugging
          #print(img)
          inpSeq.append(self.spatial_transform(img.convert('RGB')))
        inpSeq = torch.stack(inpSeq, 0)
        return inpSeq, label
