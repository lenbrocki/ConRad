# with this model we predict concepts from the LIDC dataset
# concepts are: diameter, spiculation, lobulation, margin

from pytorch_lightning import LightningModule, Trainer

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from torchvision import transforms
import random


from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau

import torchvision.models as models


def create_model():
    model = models.resnet.resnet50(weights="DEFAULT")
    model.fc = nn.Linear(2048, 8)    
    return model

class conceptModelFinetune(LightningModule):
    def __init__(self, 
                 learning_rate=1e-3,
                 weight_decay=1e-4,
                 huber_delta=0.5):

        super().__init__()

        # Set our init args as class attributes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = create_model()
#         self.criterion = nn.MSELoss()
        self.criterion = nn.HuberLoss(delta=huber_delta)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # on validation we return a list with all three views
        # final output is averaged over these views
        x, y = batch
        logits_views = torch.zeros((3, y.shape[0], 8))
        logits_views = logits_views.type_as(x[0])
        for i in range(3):
            
            logits_views[i] = self(x[i])
        logits = torch.mean(logits_views, axis=0)
        
        loss = self.criterion(logits, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)
        lr_scheduler = {
                'scheduler': MultiStepLR(optimizer, milestones=[20,40], gamma=0.1),
                'monitor': 'val_loss',
                'name': 'log_lr'
         }

        return [optimizer], [lr_scheduler]