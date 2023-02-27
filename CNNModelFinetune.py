from pytorch_lightning import LightningModule, Trainer

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, AUROC, Precision, Recall, ROC
from torchvision import transforms
import random
from sklearn import metrics

from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau

import torchvision.models as models

def create_model(feature_extract, train_layers):
    model = models.resnet.resnet50(weights="DEFAULT")
#     model.load_state_dict(torch.load(f"/home/lbrocki/BYOL/weights/old/epoch_19.pt"))
    # freeze layers, except for last train_layers
    if feature_extract:
        num_param = len(list(model.parameters()))
        for i, param in enumerate(model.parameters()):
            if(i < num_param - train_layers):
                param.requires_grad = False
    model.fc = nn.Linear(2048, 1)    
    return model

class CNNModelFinetune(LightningModule):
    def __init__(self, 
                 learning_rate=1e-3, 
                 momentum=0.0,
                 feature_extract=False,
                 train_layers = 10):

        super().__init__()

        # Set our init args as class attributes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.feature_extract = feature_extract
        
        self.model = create_model(feature_extract, train_layers)
        
        self.accuracy = Accuracy(task="binary")
        self.auroc = AUROC(task="binary")
        self.precision_ = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.roc = ROC(task="binary")
        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, = batch
        logits = self(x)
        # logits_views = torch.zeros((3, y.shape[0], 2), device=self.device)
        # for i in range(3):
        #     logits_views[i] = self(x[i])
        # logits = torch.mean(logits_views, axis=0)
        
        #preds = torch.argmax(logits, dim=1)
        preds = logits.squeeze()
        acc = self.accuracy(preds, y)

        loss = self.criterion(preds, y.float())
        #loss = self.criterion(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # on validation we return a list with all three views
        # final output is averaged over these views
        x, y = batch
        logits_views = torch.zeros((3, y.shape[0], 1), device=self.device)
        for i in range(3):
            logits_views[i] = self(x[i])
        logits = torch.mean(logits_views, axis=0).squeeze()
        
        #preds = torch.argmax(logits, dim=1)
        preds=logits
    
        m = nn.Sigmoid()
        preds_sig = m(preds)
        acc = self.accuracy(preds_sig, y)
        auroc = self.auroc(preds_sig, y)
        precision = self.precision_(preds_sig,y)
        recall = self.recall(preds_sig, y)
        roc = self.roc(preds_sig, y)
        loss = self.criterion(logits, y.float())
        
#         print(metrics.roc_auc_score(y.cpu().numpy(), preds_sig.cpu().numpy()))
#         print(auroc)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_auroc", auroc, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_precision", precision, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        params_to_update = self.model.parameters()
        print("Params to learn:")
        if self.feature_extract:
            params_to_update = []
            for name,param in self.model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
#         else:
#             for name,param in self.model.named_parameters():
#                 if param.requires_grad == True:
#                     print("\t",name)        
        
        
#         optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)
        optimizer = torch.optim.Adam(params_to_update, self.learning_rate)
        lr_scheduler = {
                'scheduler': MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1),
                #'scheduler': ReduceLROnPlateau(optimizer),
                'monitor': 'val_loss',
                'name': 'log_lr'
         }

        return [optimizer], [lr_scheduler]