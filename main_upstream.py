import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import time
import os
import copy
import statistics
import pickle
import PIL.Image as Image

from upstream_modules.self_byol import *
from upstream_modules.self_rotnet import *

device = torch.device("cuda:0")

def parse_option() :
    
    parser = argparse.ArgumentParser('argument for upstream tasks')
    
    parser.add_argument('--self_task_type', type=str, default='byol', help='select the self-supervised task to train', choices=['rotnet', 'byol', 'stanford'])
    parser.add_argument('--exp_task_type', type=str, default='10_class_classification', help='check the type of the task', choices=['2_class_classification', '10_class_classification'])
    parser.add_argument('--dataset_path', type=str, default='../dataset/', help='Dataset dir root')
    parser.add_argument('--do_create_pretext_dataset', type=bool, default=False)
    parser.add_argument('--do_train_self_weight', type=bool, default=True)
    
    parser.add_argument('--self_train_size', type=int, default=500)
    parser.add_argument('--self_test_size', type=int, default=100)
    
    parser.add_argument('--self_batch_size', type=int, default=64)
    parser.add_argument('--self_n_epochs', type=int, default=1000)
    parser.add_argument('--self_weight_decay_level', type=float, default=5e-2)
    parser.add_argument('--self_learning_rate', type=float, default=0.00003)
    
    parser.add_argument('--unlabeled_train_root', type=str, default='../dataset/self_dataset/byol_train_size_500/Training_unlabeled/')
    parser.add_argument('--train_root', type=str, default='../dataset/self_dataset/byol_train_size_500/Training_labeled')
    parser.add_argument('--test_root', type=str, default='../dataset/self_dataset/byol_train_size_500/Test_labeled')
    parser.add_argument('--num_epochs', type=int, default=1000)
    
    opt = parser.parse_args()
    
    return opt


def main() : 
    opt = parse_option()

    if opt.do_create_pretext_dataset:

        if opt.self_task_type == 'rotnet':
            create_rotnet_set(opt.dataset_path, opt.self_train_size, opt.self_test_size, opt.exp_task_type)

        elif opt.self_task_type == 'byol':
            create_byol_set(opt.dataset_path, opt.self_train_size, opt.self_test_size, opt.exp_task_type)

    if opt.do_train_self_weight:

        if opt.self_task_type == 'rotnet':
            train_rotnet_weight(opt.dataset_path, opt.self_train_size, opt.self_test_size, opt.exp_task_type,\
                                opt.self_batch_size, opt.self_n_epochs, opt.self_learning_rate, opt.self_weight_decay_level,\
                                device)

        elif opt.self_task_type == 'byol':
            train_byol_weight(opt.unlabeled_train_root, opt.train_root, opt.test_root, opt.num_epochs)
            
            
if __name__ == '__main__' :
    main()