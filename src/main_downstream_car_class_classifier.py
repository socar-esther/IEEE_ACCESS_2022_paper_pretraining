import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
# For our model
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle

import os
import shutil
import pandas as pd
import numpy as np
from statistics import mean
import copy

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import *

from utils import *
from downstream_modules.downstream_car_class_classification import *

import neptune

device = torch.device("cuda:0")


'''
Experiment Configuration
'''

do_train_downstream_classifier = True

down_n_epochs = 400 # 100 at 10-class
down_batch_size = 128
down_learning_rate = 0.000001 # 0.00001, 0.001, 0.000001
down_weight_decay = 5e-4

# 실험할 dataset 종류
datasets = [
    '2_class_classification_10p',
#     '10_class_classification_5p',
]

upstream_weight_types = [
#     'naive', # done
#     'imagenet', # done
#     'stanford-car', # weight 수정 필요
#     'byol', # weight import 안됨
    'rotation' # done
]

if do_train_downstream_classifier:
    
    print('[Notification] Start Training Downstream Classifier!')
    
    for dataset in datasets:
        for upstream_weight_type in upstream_weight_types:
            
            print(dataset, ', ', upstream_weight_type)
            
            train_downstream_classfier(dataset, upstream_weight_type, down_batch_size, down_n_epochs, down_learning_rate, down_weight_decay, device)
    
