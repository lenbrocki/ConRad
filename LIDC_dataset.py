import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import random
import numpy as np

# labels can either be target(benign/malignant) or concepts(diameter, spiculation..)
class LIDC(Dataset):
    def __init__(self, 
                 data_dir="dataset_masks",
                 transform=None, 
                 label_transform=None,
                 train_mode=True,
                 return_mask=False,
                 apply_mask=False,
                 full_vol=False,
                 labels="targets",
                 finetune=False):
        
        self.data_dir = data_dir
        crop_dir = f"{data_dir}/crops"
        mask_dir = f"{data_dir}/masks"
        df_all_labels = pd.read_pickle(f"{data_dir}/annotations_df.pkl")
        
        self.targets = df_all_labels["target"]
        df = df_all_labels[["subtlety", 
                         "calcification", 
                         "margin", 
                         "lobulation", 
                         "spiculation", 
                         "diameter", 
                         "texture", 
                         "sphericity"]].copy()
        self.concepts = df.to_numpy()
        self.views = ["axial", "coronal", "sagittal"]

        # the dataset is small, so we can load it into memory at once
        self.images = []
        self.masks = []
        for idx in range(len(self.targets)):
            img_path = f"{crop_dir}/{df_all_labels['path'][idx]}"
            mask_path = f"{mask_dir}/{df_all_labels['path'][idx]}"
            image = torch.load(img_path).float()
            mask = torch.load(mask_path).float()
            self.images.append(image)
            self.masks.append(mask)
            
            
        self.transform = transform
        self.label_transform = label_transform
        self.train_mode = train_mode
        self.return_mask = return_mask
        self.apply_mask = apply_mask
        self.full_vol = full_vol
        self.labels = labels
        self.finetune = finetune

    def __len__(self):
        return len(self.targets)
    
    def process_image(self, view, idx, slice_=16):
        image = self.images[idx]
        mask = self.masks[idx]
        # extract chosen slice
        
        if(not self.full_vol):
            if(view == self.views[0]):
                image = image[:,:,slice_]
                mask = mask[:,:,slice_]

            if(view == self.views[1]):
                image = image[:,slice_,:]
                mask = mask[:,slice_,:]

            if(view == self.views[2]):
                image = image[slice_,:,:]
                mask = mask[slice_,:,:]

        # clamp HU in this range to filter air and bone regions
        image = torch.clamp(image, -1000, 400)
        
        if(len(image.shape) < 3):
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            
        assert image.shape == mask.shape
        

          # optionally scale from range [-1000, 400] to [0,1] and apply standard ResNet in DataModule
#         image -= -1000
#         image = image/1400
        
        if(self.apply_mask):
            image = image*mask
            
        if(self.finetune):
            image = image.repeat(3,1,1)
                    
        if(self.transform is not None):
            image = self.transform(image)
        
        return image.float(), mask.float()

    def __getitem__(self, idx):
        if(self.labels == "targets"):
            label = self.targets[idx]
        elif(self.labels == "concepts"):
            concepts1 = self.concepts[idx]
            if(not self.label_transform == None):
                scaler = self.label_transform
                concepts1 = scaler.transform(np.expand_dims(concepts1, axis=0))[0]
            label = torch.tensor(concepts1).float()
        else:
            print("Unknown label chosen! Options are: 1)targets 2)concepts")
#         print(label.shape)
        # for training: randomly choose one of the views of the nodule
        # augment the training by choosing a random slice close to center
        slices = np.linspace(14, 18, 5).astype(int)
        if(self.train_mode):
            view = random.choice(self.views)
            slice_ = random.choice(slices)
            image, mask = self.process_image(view, idx, slice_)
            if(self.return_mask):
                return [image, label, mask]
            else:
                return [image, label]
        
        # for testing return all three views, testing only on center slice
        else:
            images = []
            masks = []
            for view in self.views:
                image, mask = self.process_image(view, idx, slice_=16)
                images.append(image)
                masks.append(mask)
                
            if(self.return_mask):
                return [images, label, masks]
            else:
                return [images, label]
            
    def get_target(self, idx):
        target = self.targets[idx]
        return target
        