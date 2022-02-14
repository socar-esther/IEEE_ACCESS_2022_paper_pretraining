import numpy as np
import argparse
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
import logging
from datetime import datetime
from downstream_modules.accida_classification_utils import BasicConv, resnet50, count_parameters, jigsaw_generator

from pytorch_lightning.callbacks import ProgressBar, GradientAccumulationScheduler, ModelCheckpoint, EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger 

from downstream_modules.downstream_accida_classification import *
from downstream_modules.accida_classification_utils import *

from utils import load_model

device = "cuda:0"

def parse_option() :
    
    parser = argparse.ArgumentParser('argument for downstream task : car-defect classifier')
    parser.add_argument('--dataset', type=str, default='2_class_classification')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_base', type=str, default='resnet50_pmg')
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--classes', type=int, 2)
    parser.add_argument('--LOSS', type=str, default='ce_vanilla', choices=['ce_vanilla', 'ce_label_smooth', 'complement', 'large_margin'])
    parser.add_argument('--epoch', type=int, 20)
    opt = parser.parse_args()
    
    opt.upstream_weight_type_list = ['rotation', 'naive', 'imagenet', 'stanford-car', 'byol', 'car_class']
    
    return opt

def main() :
    
    opt = parse_option()
    
    torch.manual_seed(0)
    np.random.seed(0)

    for upstream_weight_type in opt.upstream_weight_type_list : 
        
        model_name = opt.model_base
        time = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = 'weights/{}/{}'.format(upstream_weight_type,opt.LOSS)
        reg_type = None

        if opt.LOSS == 'ce_vanilla':
            print('==> The loss function is set as: ', opt.LOSS)
            loss = nn.CrossEntropyLoss()

        elif opt.LOSS == 'ce_label_smooth':
            print('==> The loss function is set as: ', opt.LOSS)
            loss = SmoothCrossEntropyLoss()

        elif opt.LOSS == 'large_margin':
            loss = nn.CrossEntropyLoss()
            reg_type = 'large_margin'

        elif opt.LOSS == 'complement':
            loss = ComplementCrossEntropy()

        else:
            print('====> The LOSS IS NOT SET PROPERLY')

        model = load_model(model_name, loss, opt.learning_rate, opt.batch_size, opt.num_workers, pretrain=True, regularizer=reg_type)

        print('The model has {:,} trainable parameters'.format(
            count_parameters(model)))


        neptune_logger = NeptuneLogger(
        api_key = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMDhiZWJiMjctNTk5YS00YzM0LWI4NjAtYmY2NjZlMWYwOTk5In0=',
        project_name = "esther/sandbox"
        )


        bar = ProgressBar()
        # early_stopping = EarlyStopping('val_acc_en', patience=15)

        # Two validation metrics. Let's have two different saver. 
        ckpt_en = ModelCheckpoint(
            dirpath=save_dir, monitor='val_acc_en', mode = 'auto', save_last=True,    filename='{epoch:02d}-{val_acc_en:.4f}_{upstream_weight_type}')

        ckpt_reg = ModelCheckpoint(
            dirpath=save_dir, monitor='val_acc_en', mode = 'auto', save_last=True,    filename='{epoch:02d}-{val_acc:.4f}_{upstream_weight_type}')


        csv_logger = CSVLogger('csv_log/{}'.format(time), name = '{}_model'.format(opt.LOSS))
        tensorboard_logger = TensorBoardLogger('tb_log/{}'.format(time),    name='{}_model'.format(opt.LOSS))

        # use resume_from_checkpoint
        # resume_point = 'weights/20201201_180420/ce_label_smooth/last.ckpt'

        trainer = pl.Trainer(auto_scale_batch_size='power', callbacks=[bar, ckpt_en, ckpt_reg],
                          max_epochs=opt.epoch, gpus=1, precision=16, logger = [neptune_logger, csv_logger])

        print('==> Starting the training process now...')
        trainer.tune(model) 
        trainer.fit(model)

        print('==> starting the testing process now...')
        trainer.test()