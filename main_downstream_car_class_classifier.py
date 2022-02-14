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
import argparse

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

def parse_option() :
    
    parser = argparse.ArgumentParser('argument for downstream tasks')
    parser.add_argument('--do_train_downstream_classifier', type=bool, default=True)
    parser.add_argument('--down_n_epochs', type=int, default=400)
    parser.add_argument('--down_batch_size', type=int, default=128)
    parser.add_argument('--down_learning_rate', type=float, default=0.000001)
    parser.add_argument('--down_weight_decay', type=float, default=5e-4)
    opt = parser.parse_args()
    
    opt.datasets = [
    '2_class_classification_10p',
    '10_class_classification_5p',
    ]
    
    opt.upstream_weight_types = [
        'naive', 
        'imagenet', 
        'stanford-car', 
        'byol', 
        'rotation' 
    ]
    
    return opt


def main() :
    opt = parse_option()

    if opt.do_train_downstream_classifier:

        print('[Notification] Start Training Downstream Classifier!')

        for dataset in opt.datasets:
            for upstream_weight_type in opt.upstream_weight_types:

                print(dataset, ', ', upstream_weight_type)

                train_downstream_classfier(dataset, upstream_weight_type, opt.down_batch_size, opt.down_n_epochs, opt.down_learning_rate, opt.down_weight_decay, device)
    

if __name__ == '__main__' :
    main()