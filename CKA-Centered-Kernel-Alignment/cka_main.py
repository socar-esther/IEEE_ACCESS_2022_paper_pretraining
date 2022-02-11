# for CKA
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

# seed 고정 (학습때와 동일하게)
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
    # 사용할 데이터셋
    image_datasets, dataloaders = test_dataloader()
    
    # 현재 있는 upstream weight 전부 사용해서 cka찍어보기
    for model_nm_1 in os.listdir('models_activations/socarvision_Upstream/upstream/') :
        if 'ipynb' in model_nm_1 : 
            continue
            
        print('>> check first model name : ', model_nm_1)
        net1 = resnet50(pretrained = False)
        net1_dir = os.path.join('models_activations/socarvision_Upstream/upstream/', model_nm_1)
        if "stanford" in model_nm_1 :
            in_features = net1.fc.in_features
            net1.fc = nn.Linear(
                in_features, 
                196
            )
        elif "rot" in model_nm_1 :
            in_features = net1.fc.in_features
            net1.fc = nn.Linear(
                in_features, 
                4
            )
        elif "byol" in model_nm_1 :
            in_features = net1.fc.in_features

        net1.load_state_dict(torch.load(net1_dir, map_location=device), strict = False)
        
        net1 = net1.to(device)
        net1 = net1.eval()
        
        # net1에 register hook 등록
        count = 0        # 예외 레이어를 지정하기 위한 변수
        local_count = 0  # feature을 리스트에 저장하기 위한 변수
        global_count = 0 # 전체 레이어수를 세기 위한 변수 (for max pooling)

        layer_name_list = list()

        def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach()
            return hook

        for layer in net1.modules() :
            # conv2d인 레이어에 대해서만 layer output 가져오는 형식
    
            # 두번째에 있는 max pooling도 가져오기
            if global_count == 1 : 
                layer_name_list.append(layer)
                layer.register_forward_hook(get_features(str(local_count)))
                local_count += 1
        
            # 여기서 제외할 4개 레이어 지정
            if isinstance(layer, torch.nn.modules.conv.Conv2d) :
                count += 1
                if count != 4 and count != 14 and count !=27 and count != 46 :
                    #print('>> check layer name : ', layer)
                    layer_name_list.append(layer)
        
                    # 등록된 hook가 있는 모든 레이어에 대해서 저장하는 방식을 사용하되 레이어 이름별로 저장
                    layer.register_forward_hook(get_features(str(local_count)))
                    local_count += 1
            global_count += 1
        print('>> check layer number : ', len(layer_name_list))
        
        # 첫번째 모델의 conv layer output 값 전부 리스트(=features_list)에 저장
        start_time = time.time()
        features_list = list() 

        for i in range(len(layer_name_list)) :
            print(f'Staring with {i}th layer, total 50 layers')

            preds_all_list = list()
            feats_all_list = list()

            for idx, (img, label) in tqdm(enumerate(dataloaders)) :
                # Accida test와 동일한 데이터 사용
                img = img.to(device)
                label = label.to(device)
    
                # feature 저장할 딕셔너리 지정
                features = dict()
                preds = net1(img)
    
                preds_all_list.append(preds.detach().cpu().numpy())
                feats_all_list.append(features[str(i)].cpu().numpy()) 
    
            # layer output값을 뽑아낸 다음 shape에 따라 지정
            features = np.empty((50,feats_all_list[0].shape[1] ,feats_all_list[0].shape[2] ,feats_all_list[0].shape[3]), dtype='float32')
    
            print('>> check array shape : ', feats_all_list[0].shape)
    
            count = 0
            for j in range(len(feats_all_list)) :
                for k in range(feats_all_list[j].shape[0]) :
                    #temp = feats_all_list[j][k].astype(np.float16)
                    features[count] = feats_all_list[j][k]
                    count += 1
    
            features_list.append(features)
            print('- feats layer shape :', features.shape, f'in layer {i}')
    
        print('-- durting time : ', time.time() - start_time)
        # 각 레이어별로 CKA input값으로 들어가는 값은 feature_list_{layer_num}으로 저장되고 있음 (할당하고 저장하는 방식사용)
        
        # residual memory 정리
        gc.collect()

        for model_nm_2 in os.listdir('models_activations/socarvision_Upstream/upstream/') : 
            if 'ipynb' in model_nm_2 :
                continue
            print('>> check second model name : ', model_nm_2)
            net2 = resnet50(pretrained = False)
            net2_dir = os.path.join('models_activations/socarvision_Upstream/upstream/', model_nm_2)
            
            if "stanford" in model_nm_2 :
                in_features = net2.fc.in_features
                net2.fc = nn.Linear(
                    in_features, 
                    196
                )
            elif "rot" in model_nm_2 :
                in_features = net2.fc.in_features
                net2.fc = nn.Linear(
                    in_features, 
                    4
                )
            elif "byol" in model_nm_2 :
                in_features = net2.fc.in_features
                
            net2.load_state_dict(torch.load(net2_dir, map_location=device), strict = False)
            
            net2 = net2.to(device)
            net2 = net2.eval()
            
            
            # net2에 대해 register hook 등록
            count = 0        # 예외 레이어를 지정하기 위한 변수
            local_count = 0  # feature을 리스트에 저장하기 위한 변수
            global_count = 0 # 전체 레이어수를 세기 위한 변수 (for max pooling)

            layer_name_list_2 = list() 

            def get_features(name):
                def hook(model, input, output):
                    features[name] = output.detach()
                return hook

            for layer in net2.modules() :
                # conv2d인 레이어에 대해서만 layer output 가져오는 형식
    
                # 두번째에 있는 Max pooling도 가져오기
                if global_count == 1 : 
                    layer_name_list_2.append(layer)
                    layer.register_forward_hook(get_features(str(local_count)))
                    local_count +=1 
    
                # 여기서 제외할 4개의 레이어 지정 
                if isinstance(layer, torch.nn.modules.conv.Conv2d) :
                    count += 1
                    if count != 4 and count != 14 and count !=27 and count != 46 :
                        #print('>> check layer name : ', layer)
                        layer_name_list_2.append(layer)
        
                        # 등록된 hook가 있는 모든 레이어에 대해서 저장하는 방식을 사용하되 레이어 이름별로 저장
                        layer.register_forward_hook(get_features(str(local_count)))
                        local_count += 1
                global_count += 1
        
            print('>> check layer number : ', len(layer_name_list_2))
            
            # 두번째 모델의 conv layer output 값 전부 리스트(=features_list)에 저장
            start_time = time.time()
            features_list_2 = list() 

            for i in range(len(layer_name_list_2)) :
                print(f'Staring with {i}th layer, total 50 layers')

                preds_all_list = list()
                feats_all_list = list()

                for idx, (img, label) in tqdm(enumerate(dataloaders)) :
                    # Accida test와 동일한 데이터 사용
                    img = img.to(device)
                    label = label.to(device)
    
                    # feature 저장할 딕셔너리 지정
                    features = dict()
                    preds = net2(img)
    
                    preds_all_list.append(preds.detach().cpu().numpy())
                    feats_all_list.append(features[str(i)].cpu().numpy()) 
    
                # layer output값을 뽑아낸 다음 shape에 따라 지정
                features = np.empty((50,feats_all_list[0].shape[1] ,feats_all_list[0].shape[2] ,feats_all_list[0].shape[3]), dtype='float32')
    
                print('>> check array shape : ', feats_all_list[0].shape)
    
                count = 0
                for j in range(len(feats_all_list)) :
                    for k in range(feats_all_list[j].shape[0]) :
                        #temp = feats_all_list[j][k].astype(np.float16)
                        features[count] = feats_all_list[j][k]
                        count += 1
    
                features_list_2.append(features)
                print('- feats layer shape :', features.shape, f'in layer {i}')

            print('-- durting time : ', time.time() - start_time)
            # 각 레이어별로 CKA input값으로 들어가는 값은 feature_list_{layer_num}으로 저장되고 있음 (할당하고 저장하는 방식사용)
            # 받아온 두 모델의 50개의 값 기반으로 score 찍기
            cka_all_scores = np.zeros((50, 50), dtype = 'float32')
            start_time = time.time()

            for i in range(50) :
                print(f'Staring with {i}th layer, total layer 50 ... ')
    
                shape = features_list[i].shape
                activationA = np.reshape(features_list[i], newshape = (shape[0], np.prod(shape[1:])))
        
                for j in range(50) :
                    #print(f'-- Intermediate with {j}th, total 52')
        
                    shape = features_list_2[j].shape
                    activationB = np.reshape(features_list_2[j], newshape = (shape[0], np.prod(shape[1:])))
        
                    # linear cka값 연산
                    cka = kernel_CKA(activationA, activationB)
        
                    # all_score에 값 저장
                    cka_all_scores[i][j] = cka

            print('Duration time : ', time.time() - start_time)
            
            # 히트맵으로 결과 찍어보기 
            plt.figure(figsize = (12, 10))
            ax = sns.heatmap(cka_all_scores)
            plt.show()
            
            plt.savefig(f'outputs/{model_nm_1}_{model_nm_2}_CKA.png')