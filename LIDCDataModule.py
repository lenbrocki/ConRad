import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from LIDC_dataset import LIDC
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection, preprocessing
import sys
import torch
import numpy as np

from util.MyRotation import MyRotation

from torchvision import transforms

class LIDCDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir = "dataset", 
                 fold=0, 
                 batch_size=32, 
                 num_workers=8, 
                 return_mask=False,
                 apply_mask=False,
                 ResNet_norm=False,
                 full_vol=False,
                 extract=False,
                 labels="targets",
                 finetune=False):
        super().__init__()
        self.data_dir = data_dir
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_mask = return_mask
        self.apply_mask = apply_mask
        self.ResNet_norm = ResNet_norm
        self.full_vol = full_vol
        self.extract = extract
        self.labels=labels
        self.finetune = finetune
        
    def setup(self, stage=None):
        # first load dataset without any transforms to extract mean and std for z-score scaling
        # train_mode argument only controls whether a random or all three views are returned
        full_train = LIDC(data_dir=self.data_dir, 
                          labels=self.labels,
                          train_mode=False, 
                          finetune=self.finetune
        )

        num_full = len(full_train)
        indices_full = list(range(num_full))
        
        # extract labels for stratified k-folds
        all_labels = np.array([full_train.get_target(i) for i in range(num_full)])
        
        # use 5-fold cross validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        gen_splits = skf.split(indices_full, all_labels)
        
        train_idx_folds = []
        test_idx_folds = []
        for train_idx, test_idx in gen_splits:
            train_idx_folds.append(train_idx)
            test_idx_folds.append(test_idx)
            
        # choose k-fold
        train_idx = train_idx_folds[self.fold]
        test_idx = test_idx_folds[self.fold]
              
        if(not self.ResNet_norm):
            # we calculate mean and std of the training data for the z-score normalization
            # for the concepts we similarly obtain the StandardScaler
            # the same scaling is then applied to the test data
            train_imgs = []
            train_concepts = []
            for idx in train_idx:
                image = full_train[idx][0][0]
                train_imgs.append(image)
                if(self.labels == "concepts"):
                    concepts = full_train[idx][1]
                    train_concepts.append(concepts)
            scaler = None   
            if(self.labels == "concepts"):
                train_concepts = np.stack(train_concepts, axis=0)
                scaler = preprocessing.StandardScaler().fit(train_concepts)

            train_imgs = torch.stack(train_imgs, axis=0)
            channels_mean = torch.mean(train_imgs)
            channels_std = torch.std(train_imgs)
        else:
            # can also experiment with the standard ResNet normalization (images must be scaled to range [0,1])
            channels_mean = [0.485, 0.456, 0.406]
            channels_std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.Normalize(mean=channels_mean, std=channels_std),
                MyRotation([0, 90, 180, 270]),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.Normalize(mean=channels_mean, std=channels_std),
            ]
        )   
        
        if(self.extract == True):
            train_mode=False
        else:
            train_mode=True

        train = LIDC(
            data_dir=self.data_dir, 
            train_mode=train_mode, 
            transform=train_transform, 
            label_transform=scaler,
            return_mask=self.return_mask,
            apply_mask=self.apply_mask,
            full_vol=self.full_vol,
            labels=self.labels,
            finetune=self.finetune
        )
        test = LIDC(
            data_dir=self.data_dir, 
            train_mode=False, 
            transform=test_transform, 
            label_transform=scaler,
            return_mask=self.return_mask,
            apply_mask=self.apply_mask,
            full_vol=self.full_vol,
            labels=self.labels,
            finetune=self.finetune
        )
                
        self.train_data = torch.utils.data.Subset(train, train_idx)
        self.val_data = torch.utils.data.Subset(test, test_idx)
        self.test_data = torch.utils.data.Subset(test, test_idx)
        
    def train_dataloader(self):
        if(self.extract == True):
            shuffle=False
        else:
            shuffle=True
        return DataLoader(self.train_data,
                          batch_size=self.batch_size, 
                          shuffle=shuffle, 
                          num_workers=self.num_workers
                          )

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)