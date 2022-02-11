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
import neptune

from utils import * 

def create_self_supervised_set(base_dir, data_name, self_train_num_per_class, self_test_num_per_class, do):
    
    print('=============================================')
    print('> Start Creating Dataset for Self Supervision Task...')
    print('=============================================')
    
    candidate_train_filenames = list()
    candidate_test_filenames = list()
    
    # 각 Class 별로 파일 이름 모으고
    # Training Set에서만 추출할 것임
    target_src_path = base_dir
    print(os.listdir(target_src_path))
    
    for class_name in sorted(os.listdir(target_src_path)):
        
        class_path = os.path.join(target_src_path, class_name)
        class_test_filenames = sorted(os.listdir(class_path))[: self_test_num_per_class]
        class_train_filenames = sorted(os.listdir(class_path))[self_test_num_per_class : self_test_num_per_class+self_train_num_per_class]
        
        for class_filename in class_test_filenames:
            class_filepath = os.path.join(class_path, class_filename)
            candidate_test_filenames.append(class_filepath)
        
        for class_filename in class_train_filenames:
            class_filepath = os.path.join(class_path, class_filename)
            candidate_train_filenames.append(class_filepath)
            
    print('Train for Self: ', len(candidate_train_filenames))            
    print('Test for Self: ', len(candidate_test_filenames))
    
    print('>> Start Making a Rotated Set...')
    save_base = os.path.join('dataset/', 'rotated_set', data_name, 'train_size_' + str(self_train_num_per_class))
    
    print('> Start Rotating Training Set...')
    train_save_dir = os.path.join(save_base, 'Training')
    
    if not os.path.exists(train_save_dir):
        os.makedirs(train_save_dir)
        
    if do:
        rotate_and_save(candidate_train_filenames, train_save_dir)
        
    print('> Start Rotating Test Set...')
    test_save_dir = os.path.join(save_base, 'Test')
    
    if not os.path.exists(test_save_dir):
        os.makedirs(test_save_dir)
        
    if do:
        rotate_and_save(candidate_test_filenames, test_save_dir)
    
    return train_save_dir, test_save_dir
        

def rotate_and_save(src_filenames, save_dir):
    rotation_list = [0, 90, 180, 270]
        
    for rotation_degree in rotation_list:
        print('> working on degree '+ str(rotation_degree))
        
        rot_save_dir = os.path.join(save_dir, 'degree_' + str(rotation_degree))
        if not os.path.exists(rot_save_dir):
            os.makedirs(rot_save_dir)

        for src_filename in src_filenames:

            
            img = cv2.imread(src_filename)
            rotated_img = image_rotator(img, rotation_degree)
            
#             print(src_filename)
#             print(rot_save_dir, src_filename.split('/')[-1])
            
            cv2.imwrite(os.path.join(rot_save_dir, src_filename.split('/')[-1]), rotated_img)
    
    
def image_rotator(img, rotate_type):
    
    if rotate_type == 0:
        rotated_img = img
    
    elif rotate_type == 90:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
    elif rotate_type == 180:
        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
        
    elif rotate_type == 270:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    else:
        raise ValueError

    return rotated_img
