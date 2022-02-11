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
import logging
from datetime import datetime
from downstream_modules.accida_classification_utils import BasicConv, resnet50, count_parameters, jigsaw_generator

from pytorch_lightning.callbacks import ProgressBar, GradientAccumulationScheduler, ModelCheckpoint, EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger 

from downstream_modules.downstream_accida_classification import *
from downstream_modules.accida_classification_utils import *

device = "cuda:0"

upstream_weight_type_list = ['imagenet'] # 'rotation', 'naive', 'imagenet', 'stanford-car', 'byol', 'car_class'
dataset = '2_class_classification'
BATCH_SIZE = 2 # this will be automatically tuned and increased to best fit your machine.
NUM_WORKERS = 4
MODEL_BASE = 'resnet50_pmg'
LEARNING_RATE = 0.002   # with weight-decay 
CLASSES = 2
LOSS= 'ce_vanilla' #options: ce_vanilla, ce_label_smooth, complement, large_margin
EPOCH = 20


def load_model(model_name, loss, learning_rate, batch_size, num_workers, regularizer, pretrain=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        ## TODO 1: upstream task에 따라 pretrain weight 달라지게 하기
        ## TODO 2: root 맞춰두기 
        ## TODO 3: 어떤 pth 파일 로드하는지 확인하는 로그 출력해두기
        #for upstream_weight_type in upstream_list :
            
        if upstream_weight_type == 'naive':
            print('>> upstream : naive')
            net = resnet50(pretrained=False)
        
        elif upstream_weight_type == 'car_class' :
            print('>> upstream : car_class classification')
            net = resnet50(pretrained = False)
            
            in_features = net.fc.in_features
            
            upstream_weight_path = '10class_imagenet_acc_0.91796.pth'
            
            net.fc = nn.Sequential(
                nn.Linear(
                    in_features,
                    10 # due to the number of classes at 10_class_classifciation
                )
            )

            net.load_state_dict(torch.load(os.path.join('upstream_artifacts/', upstream_weight_path), 
                                          map_location = device))
            net.to(device)
            print('Successfully Loaded the Weight!') 

        elif upstream_weight_type == 'imagenet':
            print('>> upstream : Imagenet')
            net = resnet50(pretrained=True)

        elif upstream_weight_type == 'stanford-car':
            print('>> upstream : stanford_car')
            upstream_weight_path = 'resnet50_acc91_stanfordcar_pretrain_true.pth'

            if os.path.exists(os.path.join('upstream_artifacts/', upstream_weight_path)):
                pass

            else:
                print('[Notification] Downloading ', upstream_weight_type, ' .pth files...')

                if not os.path.exists('upstream_artifacts'):
                    os.makedirs('upstream_artifacts')

                os.system('gsutil cp gs://socar-data-temp/kp/research_socarvision/artifacts/' + upstream_weight_path + ' ./upstream_artifacts/.')
                #time.sleep(5) # wait until the download is completed

            net = resnet50(pretrained=False)
            in_features = net.fc.in_features


            net.fc = nn.Linear(
                in_features,
                196 # due to the number of classes at stanford-car
            )

            net.load_state_dict(torch.load(os.path.join('upstream_artifacts/', upstream_weight_path)))

            print('Successfully Loaded the Weight!')

        elif upstream_weight_type == 'byol':
            print('>> upstream : byol')
            upstream_weight_path = '2class_resnet50_acc_70_byol_pretrain_true.pth'

            net = resnet50(pretrained=False)

            in_features = net.fc.in_features

            net.load_state_dict(torch.load(os.path.join('upstream_artifacts/', upstream_weight_path)))
            print('Successfully Loaded the Weight!')

        elif upstream_weight_type == 'rotation':
            print('>> upstream : rotnet')
                
            if dataset == '2_class_classification':    
                upstream_weight_path = '2class_resnet50_acc90_rotnet_pretrain_true.pth'


            if os.path.exists(os.path.join('upstream_artifacts/', upstream_weight_path)):
                pass

            else:
                print('[Notification] Downloading ', upstream_weight_type, ' .pth files...')

                if not os.path.exists('upstream_artifacts'):
                    os.makedirs('upstream_artifacts')

                os.system('gsutil cp gs://socar-data-temp/kp/research_socarvision/artifacts/' + upstream_weight_path + ' ../upstream_artifacts/.')
                #time.sleep(5) # wait until the download is completed

            net = resnet50(pretrained=False)
            in_features = net.fc.in_features

            net.fc = nn.Sequential(
                nn.Linear(
                    in_features,
                    4 # due to the number of classes at stanford-car
                )
            )

            net.load_state_dict(torch.load(os.path.join('upstream_artifacts/', upstream_weight_path)))
                    
            print('Successfully Loaded the Weight!')   

        net = PMG(net, loss=loss, feature_size = 512, classes_num = CLASSES, batch_size=batch_size,
                  num_workers=num_workers, lr = learning_rate, reg=regularizer, root='custom') ## this should work right?

    return net


if __name__ == '__main__' :
    torch.manual_seed(0)
    np.random.seed(0)

    for upstream_weight_type in upstream_weight_type_list : 
        
        model_name = MODEL_BASE
        time = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = 'weights/{}/{}'.format(upstream_weight_type,LOSS)
        reg_type = None

        if LOSS == 'ce_vanilla':
            print('==> The loss function is set as: ', LOSS)
            loss = nn.CrossEntropyLoss()

        elif LOSS == 'ce_label_smooth':
            print('==> The loss function is set as: ', LOSS)
            loss = SmoothCrossEntropyLoss()

        elif LOSS == 'large_margin':
            loss = nn.CrossEntropyLoss()
            reg_type = 'large_margin'

        elif LOSS == 'complement':
            loss = ComplementCrossEntropy()

        else:
            print('====> The LOSS IS NOT SET PROPERLY')

        model = load_model(model_name, loss, LEARNING_RATE, BATCH_SIZE, NUM_WORKERS, pretrain=True, regularizer=reg_type)

        print('The model has {:,} trainable parameters'.format(
            count_parameters(model)))


        """
        ------------------------------------
        Intialize the trainier from the model and the callbacks
        # Tensorboard logging: tensorboard --logdir lightning_logs
        ------------------------------------
        """
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


        csv_logger = CSVLogger('csv_log/{}'.format(time), name = '{}_model'.format(LOSS))
        tensorboard_logger = TensorBoardLogger('tb_log/{}'.format(time),    name='{}_model'.format(LOSS))

        # use resume_from_checkpoint
        # resume_point = 'weights/20201201_180420/ce_label_smooth/last.ckpt'

        trainer = pl.Trainer(auto_scale_batch_size='power', callbacks=[bar, ckpt_en, ckpt_reg],
                          max_epochs=EPOCH, gpus=1, precision=16, logger = [neptune_logger, csv_logger])

        print('==> Starting the training process now...')
        trainer.tune(model) 
        trainer.fit(model)

        print('==> starting the testing process now...')
        trainer.test()