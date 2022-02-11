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

from utils import *
from rotate_datasets import *
from train import *

## conf
device = torch.device("cuda:2")

do_create_rotated_dataset = True
do_train_rotation_task = True

def run() :
    
    # train 500, test 100 at 2_class_classification
    # train 100, test 20 at 10_class_classification
    train_dir, test_dir = create_self_supervised_set(base_dir='../../../dataset/temp_10_class_classification', \
                        data_name='10_class_classification', self_train_num_per_class=500, \
                        self_test_num_per_class=10, do=do_create_rotated_dataset)
    
    train_self_supervised_weight(train_dir, test_dir, target_data_name='10_class_classification', self_train_num_per_class=500, self_model='resnet50', device=device, self_n_epochs=70, self_batch_size=16, self_learning_rate=0.01, self_weight_decay_level = 5e-3)


if __name__ == '__main__' :
    run()