import numpy as np
import random
import torch
import torchvision
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss

import torch.optim as optim
from PIL import ImageFile
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.metrics import Precision, Recall, F1
import torchvision.models as models

import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import logging
from datetime import datetime
from downstream_modules.accida_classification_utils import BasicConv, resnet50, count_parameters, jigsaw_generator

from pytorch_lightning.callbacks import ProgressBar, GradientAccumulationScheduler, ModelCheckpoint, EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger 

device = "cuda:0"

## define PMG model
class PMG(pl.LightningModule):

    def __init__(self, model, feature_size, lr, loss, classes_num, reg, batch_size=8, num_workers=6, root='car_data'):
        super(PMG, self).__init__()

        self.features = model  
        self.max1 = nn.MaxPool2d(kernel_size=56, stride=56)
        self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU(inplace=True)
        
        self.root = root
        self.reg = reg
        self.loss = loss
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers

        """
        ----------------------------------------
        Refer to graph on page 6.
        This is Conv block L-2 and classfier L-2
        ----------------------------------------
        """
        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs//4, feature_size, kernel_size=1,
                      stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2,
                      kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        """
        -----------------------------------------
        This is Conv block L-1 and classfier L-1
        -----------------------------------------
        """
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs//2, feature_size, kernel_size=1,
                      stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2,
                      kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        """
        ----------------------------------------
        This is Conv block L and classfier L
        ----------------------------------------
        """
        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1,
                      stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2,
                      kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        """
        ----------------------------------------
        Refer to graph on page 6 and 7.
        This is the classifier concat
        ----------------------------------------
        """
        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

    def forward(self, x):

        xf1, xf2, xf3, xf4, xf5 = self.features(x)

        xl1 = self.conv_block1(xf3)
        xl2 = self.conv_block2(xf4)
        xl3 = self.conv_block3(xf5)

        xl1 = self.max1(xl1)
        xl1 = xl1.view(xl1.size(0), -1)
        xc1 = self.classifier1(xl1)

        xl2 = self.max2(xl2)
        xl2 = xl2.view(xl2.size(0), -1)
        xc2 = self.classifier2(xl2)

        xl3 = self.max3(xl3)
        xl3 = xl3.view(xl3.size(0), -1)
        xc3 = self.classifier3(xl3)

        """
        xcl, xc2, xc3 will be used to calculate the loss.
        x_concat is just concat of layer features (1,2,3) 
        """
        x_concat = torch.cat((xl1, xl2, xl3), -1)
        x_concat = self.classifier_concat(x_concat)
        return xc1, xc2, xc3, x_concat

    # lightning will add optimizer inside the model
    def configure_optimizers(self):

        optimizer = optim.SGD([
            {'params': self.classifier_concat.parameters(), 'lr': self.lr},
            {'params': self.conv_block1.parameters(), 'lr': self.lr},
            {'params': self.classifier1.parameters(), 'lr': self.lr},
            {'params': self.conv_block2.parameters(), 'lr': self.lr},
            {'params': self.classifier2.parameters(), 'lr': self.lr},
            {'params': self.conv_block3.parameters(), 'lr': self.lr},
            {'params': self.classifier3.parameters(), 'lr': self.lr},
            {'params': self.features.parameters(), 'lr': self.lr/10}
        ],
            momentum=0.9, weight_decay=5e-4)
        
        # Learning rate optimizer options.
        cosineAnneal = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)#, verbose=True)
        #plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True) # you need to specify what you are monitoring. 

        step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

        warm_restart = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)#, verbose=True)
        
        return {'optimizer': optimizer, 'lr_scheduler': cosineAnneal}

    def training_step(self, batch, batch_idx):
      
        inputs, targets = batch
        loss_function = self.loss

        # step 1 (start from fine-grained jigsaw n=8)
        inputs1 = jigsaw_generator(inputs, 8)
        output_1, _, _, _ = self(inputs1) 
        loss1 = loss_function(output_1, targets) * 1 

        # step 2
        inputs2 = jigsaw_generator(inputs, 4)
        _, output_2, _, _ = self(inputs2)
        loss2 = loss_function(output_2, targets) * 1  

        # step 3
        inputs3 = jigsaw_generator(inputs, 2)
        _, _, output_3, _ = self(inputs3)
        loss3 = loss_function(output_3, targets) * 1

        # step 4 whole image, and vanilla loss. 
        _, _, _, output_concat = self(inputs)

        if self.reg == None:
            concat_loss = loss_function(output_concat, targets) * 2 
            train_loss = loss1 + loss2 + loss3 + concat_loss

        if self.reg == 'large_margin':
            pass

        elif self.reg == 'jacobian':
            pass

        # accuracy 
        _, predicted = torch.max(output_concat.data, 1)
        train_acc = accuracy(predicted, targets)

        metrics = {'loss': train_loss, 'accuracy': train_acc}

        self.log('accuracy', train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return metrics

    # * Not entirely the same as the training step. No jigsaw puzzle here.
    def validation_step(self, batch, batch_idx):

        inputs, targets = batch

        loss_function = self.loss
        
        #TODO: JOKER LOSS is probably not needed for validation
        output_1, output_2, output_3, output_concat = self(inputs)
        outputs_com = output_1 + output_2 + output_3 + output_3 + output_concat

        val_loss = loss_function(output_concat, targets)

        """
        There is the individual accuracy, and combined accuracy
        """
        _, predicted = torch.max(output_concat.data, 1)
        _, predicted_com = torch.max(outputs_com.data, 1)
        

        valid_acc = accuracy(predicted, targets)
        valid_acc_en = accuracy(predicted_com, targets)
        
        precision = Precision(num_classes = 2, average = 'micro')
        precision = precision.to(device)
        valid_prec = precision(predicted, targets)
        valid_prec_en = precision(predicted_com, targets)
        
        recall = Recall(num_classes = 2, average = 'micro')
        recall = recall.to(device)
        valid_recall = recall(predicted, targets)
        valid_recall_en = recall(predicted_com, targets)
        
        f1 = F1(num_classes = 2, average = 'micro')
        f1 = f1.to(device)
        valid_f1 = f1(predicted, targets)
        valid_f1_en = f1(predicted_com, targets)
        
        
        
        metrics = {'val_loss':  val_loss, 
                   'val_acc': valid_acc, 
                   'val_acc_en': valid_acc_en, 
                   'val_precision' : valid_prec, 
                   'val_precision_en' : valid_prec_en, 
                   'val_recall' : valid_recall, 
                   'val_recall_en' : valid_recall_en, 
                   'val_f1' : valid_f1, 
                   'val_f1_en' : valid_f1_en
                  }

        self.log('val_acc', valid_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_en', valid_acc_en, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', valid_prec, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision_en', valid_prec_en, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall_en', valid_recall_en, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', valid_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1_en', valid_f1_en, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', valid_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        


        return metrics

    
    def test_step(self, batch, batch_idx):
        # validation set이랑 똑같이 적용하는 형태 
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test_loss':  metrics['val_loss'],
                   'test_acc' : metrics['val_acc'],
                   'test_acc_en': metrics['val_acc_en'], 
                   'test_precision_en' : metrics['val_precision_en'], 
                   'test_precision' : metrics['val_precision'],
                   'test_recall' : metrics['val_recall'],
                   'test_recall_en' : metrics['val_recall_en'], 
                   'test_f1_en' : metrics['val_f1_en'], 
                   'test_f1' : metrics['val_f1']
                  }
        self.log_dict(metrics)


    def train_dataloader(self):

        transform_train = transforms.Compose([
            transforms.Resize((550, 550)),
            transforms.RandomCrop(448, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
#         trainset = torchvision.datasets.ImageFolder(
#             root= '../dataset/few_shot/Accida_10/Training', transform=transform_train)
        trainset = torchvision.datasets.ImageFolder(
            root= '../../research_few_shot/dataset/carclassAccida/support_v2', transform=transform_train)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

        return trainloader

    def val_dataloader(self):

        transform_test = transforms.Compose([
            transforms.Resize((550, 550)),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        ImageFile.LOAD_TRUNCATED_IMAGES = True

#         testset = torchvision.datasets.ImageFolder(root='../dataset/few_shot/Accida_10/Test',
#                                                    transform=transform_test)
        testset = torchvision.datasets.ImageFolder(root='../../research_few_shot/dataset/carclassAccida/query',
transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, shuffle=False, num_workers=self.num_workers, pin_memory=True) # batch_size=self.batch_size,

        return testloader
        
## Loss function
class SmoothCrossEntropyLoss(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean', smoothing=0.2):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class LargeMarginInSoftmaxLoss(nn.CrossEntropyLoss):
    """
    This combines the Softmax Cross-Entropy Loss (nn.CrossEntropyLoss) and the large-margin inducing
    """
    def __init__(self, reg_lambda=0.1, deg_logit=None, 
                weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(LargeMarginInSoftmaxLoss, self).__init__(weight=weight, size_average=size_average, 
                                ignore_index=ignore_index, reduce=reduce, reduction=reduction)
        self.reg_lambda = reg_lambda
        self.deg_logit = deg_logit

    def forward(self, input, target):
        N = input.size(0) # number of samples
        C = input.size(1) # number of classes
        Mask = torch.zeros_like(input, requires_grad=False)
        Mask[range(N),target] = 1

        if self.deg_logit is not None:
            input = input - self.deg_logit * Mask
        
        loss = F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

        X = input - 1.e6 * Mask # [N x C], excluding the target class
        reg = 0.5 * ((F.softmax(X, dim=1) - 1.0/(C-1)) * F.log_softmax(X, dim=1) * (1.0-Mask)).sum(dim=1)
        if self.reduction == 'sum':
            reg = reg.sum()
        elif self.reduction == 'mean':
            reg = reg.mean()
        elif self.reduction == 'none':
            reg = reg

        return loss + self.reg_lambda * reg


class ComplementEntropy(nn.Module):
    '''Compute the complement entropy of complement classes.'''
    def __init__(self, num_classes=2):
        super(ComplementEntropy, self).__init__()
        self.classes = num_classes
        self.batch_size = None

    def forward(self, y_hat, y):
        self.batch_size = len(y)
        y_hat = F.softmax(y_hat, dim=1)
        Yg = torch.gather(y_hat, 1, torch.unsqueeze(y, 1))
        Yg_ = (1 - Yg) + 1e-7
        Px = y_hat / Yg_.view(len(y_hat), 1)
        Px_log = torch.log(Px + 1e-10)
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_\
            (1, y.view(self.batch_size, 1).data.cpu(), 0)
        output = Px * Px_log * y_zerohot.cuda()
        entropy = torch.sum(output)
        entropy /= float(self.batch_size)
        entropy /= float(self.classes)
        return entropy


class ComplementCrossEntropy(nn.Module):
    def __init__(self, num_classes=2, gamma=5):
        super(ComplementCrossEntropy, self).__init__()
        self.gamma = gamma
        self.cross_entropy = nn.CrossEntropyLoss()
        self.complement_entropy = ComplementEntropy(num_classes)

    def forward(self, y_hat, y):
        l1 = self.cross_entropy(y_hat, y)
        l2 = self.complement_entropy(y_hat, y)
        return l1 + self.gamma * l2

