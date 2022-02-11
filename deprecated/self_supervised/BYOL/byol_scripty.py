# !pip install byol-pytorch
import torch
from byol_pytorch import BYOL
from torchvision import models
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor()
])

resnet = models.resnet50(pretrained=False)

learner = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

# exp_name = 'dataset/temp_10_class_classification'
exp_name = 'temp_ray_v_stonic'

def get_dataset() :
    # unlabeled set 
    
    # 100, 20 at 10_class_classification
    # 500, 100 at 2_class_classification
    
    train_dir, test_dir = create_self_supervised_set(base_dir = '../../../dataset/' + exp_name, target_data_name = exp_name, self_train_num_per_class = 500, self_test_num_per_class = 100)
    train_dataset = datasets.ImageFolder(root=train_dir, transform = transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print('Complete get datasets ... ')
    return train_loader 

dataloaders = get_dataset()

for i in range(70):
    print('>> check epoch :', i)
    
    for idx, (img, label) in enumerate(dataloaders) :
        if idx % 10 == 0 :
            print('>> check idx : ', idx)
        loss = learner(img)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
    
    # save your improved network
    torch.save(resnet.state_dict(), 'byol_model.pt')
