import os
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
# from create_self_dataset import *

device = torch.device("cuda:0")

# self_task_type = 'rotnet'
self_task_type = 'byol' # [rotnet, byol]
exp_task_type = '10_class_classification' # 2_class_classification ...

dataset_path = '../dataset/'

do_create_pretext_dataset = False
do_train_self_weight = True

self_train_size = 500
self_test_size = 100

self_batch_size = 64
self_n_epochs = 1000
self_weight_decay_level = 5e-2
self_learning_rate = 0.00003 # 0.01

# for byol
unlabeled_train_root = '../dataset/self_dataset/byol_train_size_500/Training_unlabeled/'
train_root = '../dataset/self_dataset/byol_train_size_500/Training_labeled'
test_root = '../dataset/self_dataset/byol_train_size_500/Test_labeled'
num_epochs = 1000

if do_create_pretext_dataset:
    
    if self_task_type == 'rotnet':
        create_rotnet_set(dataset_path, self_train_size, self_test_size, exp_task_type)

    elif self_task_type == 'byol':
        create_byol_set(dataset_path, self_train_size, self_test_size, exp_task_type)

if do_train_self_weight:
    
    if self_task_type == 'rotnet':
        train_rotnet_weight(dataset_path, self_train_size, self_test_size, exp_task_type,\
                            self_batch_size, self_n_epochs, self_learning_rate, self_weight_decay_level,\
                            device)

    elif self_task_type == 'byol':
        train_byol_weight(unlabeled_train_root, train_root, test_root, num_epochs)