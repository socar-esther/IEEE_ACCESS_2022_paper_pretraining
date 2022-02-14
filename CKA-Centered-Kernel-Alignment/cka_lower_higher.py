import pickle
import random
import gc
import os
import numpy as np
import pickle
import gzip
import cca_core
from CKA import linear_CKA, kernel_CKA

# for model
import torch
import torchvision

import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import resnet50
import cv2
import random
import math
import pandas as pd
from PIL import Image

# for register hook
import time
from tqdm import tqdm

# for heatmap
import seaborn as sns
import matplotlib.pyplot as plt

device = "cuda:0" 

# fix seed
manual_seed = 42
print("Random Seed: ", manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)

def test_dataloader():

    transform_test = T.Compose([
        T.Resize((550, 550)),
        T.CenterCrop(448),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    #ImageFile.LOAD_TRUNCATED_IMAGES = True
    testset = torchvision.datasets.ImageFolder(root='../../dataset/Accida_classification/split_dataset_v5_normal_strong_100/Test',transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=4,  batch_size=4) # batch_size = 16

    return testset, testloader



if __name__ == "__main__" :
    # Get dataset
    image_datasets, dataloaders = test_dataloader()
    
    for model_nm_1 in os.listdir('models_activations/socarvision_Upstream/upstream/') :
        if 'ipynb' in model_nm_1 : 
            continue
            
        print('>> check first model name : ', model_nm_1)
        net1 = resnet50(pretrained = False)
        net1_dir = os.path.join('models_activations/socarvision_Upstream/upstream/', model_nm_1)
        if "stanford" in model_nm_1 :
            upstream_nm = 'stanford'
            in_features = net1.fc.in_features
            net1.fc = nn.Linear(
                in_features, 
                196
            )
        elif "rot" in model_nm_1 :
            upstream_nm = 'rotnet'
            in_features = net1.fc.in_features
            net1.fc = nn.Linear(
                in_features, 
                4
            )
        elif "byol" in model_nm_1 :
            upstream_nm = 'byol'
            in_features = net1.fc.in_features
            
        elif "naive" in model_nm_1 :
            upstream_nm = 'naive'
        
        elif 'imagenet' in model_nm_1 :
            upstream_nm = 'imagenet'

        net1.load_state_dict(torch.load(net1_dir, map_location=device), strict = False)
        
        net1 = net1.to(device)
        net1 = net1.eval()
        
        # register hook to the net1
        count = 0        
        local_count = 0  
        global_count = 0 
        feature_count = 0 
        
        layer_name_list = list()

        def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach()
            return hook

        for layer in net1.modules() :

            if global_count == 1 : 
#                 layer_name_list.append(layer)
#                 layer.register_forward_hook(get_features(str(feature_count)))
#                 feature_count += 1
                local_count += 1
        
            if isinstance(layer, torch.nn.modules.conv.Conv2d) :
                count += 1
                if count != 4 and count != 14 and count !=27 and count != 46 :
                    if local_count > 43 : 
                        #print('>> check layer name : ', layer)
                        layer_name_list.append(layer)
        
                        layer.register_forward_hook(get_features(str(feature_count)))
                        feature_count += 1
                    local_count += 1
            global_count += 1
        #print('>> check layer number : ', len(layer_name_list))

        start_time = time.time()
        features_list = list() 

        for i in range(len(layer_name_list)) :
            #print(f'Staring with {i}th layer, total 50 layers')

            preds_all_list = list()
            feats_all_list = list()

            for idx, (img, label) in tqdm(enumerate(dataloaders)) :
                img = img.to(device)
                label = label.to(device)
    
                features = dict()
                preds = net1(img)
    
                preds_all_list.append(preds.detach().cpu().numpy())
                feats_all_list.append(features[str(i)].cpu().numpy()) 
    
            features = np.empty((50,feats_all_list[0].shape[1] ,feats_all_list[0].shape[2] ,feats_all_list[0].shape[3]), dtype='float32')
    
            #print('>> check array shape : ', feats_all_list[0].shape)
    
            count = 0
            for j in range(len(feats_all_list)) :
                for k in range(feats_all_list[j].shape[0]) :
                    #temp = feats_all_list[j][k].astype(np.float16)
                    features[count] = feats_all_list[j][k]
                    count += 1
    
            features_list.append(features)
            #print('- feats layer shape :', features.shape, f'in layer {i}')
    
        #print('-- durting time : ', time.time() - start_time)
        
        # residual memory
        gc.collect()

        for model_nm_2 in os.listdir('models_activations/socarvision_Upstream/upstream/') : 
            if 'ipynb' in model_nm_2 :
                continue
            
            net2 = resnet50(pretrained = False)
            net2_dir = os.path.join('models_activations/socarvision_Upstream/upstream/', model_nm_2)
            
            if "stanford" in model_nm_2 :
                downstream_nm = 'stanford'
                in_features = net2.fc.in_features
                net2.fc = nn.Linear(
                    in_features, 
                    196 #2
                )
            elif "rot" in model_nm_2 :
                downstream_nm = 'rotnet'
                in_features = net2.fc.in_features
                net2.fc = nn.Linear(
                    in_features, 
                    4
                )
            elif "byol" in model_nm_2 :
                downstream_nm = 'byol'
#                 in_features = net2.fc.in_features
#                 net2.fc = nn.Linear(
#                     in_features,
#                     2
#                 )
                
            elif "naive" in model_nm_2 :
                downstream_nm = 'naive'
                
            elif "imagenet" in model_nm_2 :
                downstream_nm = 'imagenet'
                
            # 판별
            if upstream_nm != downstream_nm :
                continue 
            else :
                print('>> check second model name : ', net2_dir)
                net2.load_state_dict(torch.load(net2_dir, map_location=device), strict = False)
            
                net2 = net2.to(device)
                net2 = net2.eval()
            
            
                # register hook to the net2
                count = 0        
                local_count = 0 
                global_count = 0 
                feature_count = 0
            
                layer_name_list_2 = list() 

                def get_features(name):
                    def hook(model, input, output):
                        features[name] = output.detach()
                    return hook

                for layer in net2.modules() :
 
                    if global_count == 1 : 
#                         layer_name_list_2.append(layer)
#                         layer.register_forward_hook(get_features(str(feature_count)))
                        local_count +=1 
#                         feature_count += 1
    
                    if isinstance(layer, torch.nn.modules.conv.Conv2d) :
                        count += 1
                        if count != 4 and count != 14 and count !=27 and count != 46 :
                            if local_count > 43 : 
                                #print('>> check layer name : ', layer)
                                layer_name_list_2.append(layer)
        
                                layer.register_forward_hook(get_features(str(feature_count)))
                                feature_count += 1
                            local_count += 1
                    global_count += 1
        
                #print('>> check layer number : ', len(layer_name_list_2))

                start_time = time.time()
                features_list_2 = list() 

                for i in range(len(layer_name_list_2)) :
                    #print(f'Staring with {i}th layer, total 50 layers')

                    preds_all_list = list()
                    feats_all_list = list()

                    for idx, (img, label) in tqdm(enumerate(dataloaders)) :
                        img = img.to(device)
                        label = label.to(device)
    
                        features = dict()
                        preds = net2(img)
    
                        preds_all_list.append(preds.detach().cpu().numpy())
                        feats_all_list.append(features[str(i)].cpu().numpy()) 
    
                    features = np.empty((50,feats_all_list[0].shape[1] ,feats_all_list[0].shape[2] ,feats_all_list[0].shape[3]), dtype='float32')
    
                    #print('>> check array shape : ', feats_all_list[0].shape)
    
                    count = 0
                    for j in range(len(feats_all_list)) :
                        for k in range(feats_all_list[j].shape[0]) :
                            #temp = feats_all_list[j][k].astype(np.float16)
                            features[count] = feats_all_list[j][k]
                            count += 1
    
                    features_list_2.append(features)
                    #print('- feats layer shape :', features.shape, f'in layer {i}')

                #print('-- durting time : ', time.time() - start_time)
                cka_all_scores = np.zeros((6, 6), dtype = 'float32')
                start_time = time.time()

                for i in range(6) :
                    #print(f'Staring with {i}th layer, total layer 6 ... ')
    
                    shape = features_list[i].shape
                    activationA = np.reshape(features_list[i], newshape = (shape[0], np.prod(shape[1:])))
        
                    for j in range(6) :
                        #print(f'-- Intermediate with {j}th, total 52')
        
                        shape = features_list_2[j].shape
                        activationB = np.reshape(features_list_2[j], newshape = (shape[0], np.prod(shape[1:])))
        
                        # linear cka
                        cka = kernel_CKA(activationA, activationB)
                        cka_all_scores[i][j] = cka

                #print('Duration time : ', time.time() - start_time)
            
                print('>> check mean : ', np.mean(cka_all_scores))
                print('>> check std :', np.std(cka_all_scores))
                print('>> check all cka scores :', cka_all_scores)
                
                with open(f'higher_lower_outputs/higher_upstream/{model_nm_1}_{model_nm_2}.pkl', 'wb') as f : 
                    pickle.dump(cka_all_scores, f)
                
