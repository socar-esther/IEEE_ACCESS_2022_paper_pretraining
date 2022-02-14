# kdd 2022 paper 1675
Implementation of "Discovering the Effectiveness of Pre-Training for Image Recognition in a Large-scale Car-sharing Platform"

## Datasets
- We open our domain benchmark set in [URL](https://socar-kp.github.io/sofar_image_dataset/)

## How to run
- Here is the sample command line to run the upstream tasks
```shell
$ python main_upstream.py --self_task_type byol \
                          --exp_task_type 10_class_classification \
                          --dataset_path ../dataset/ \
                          --do_create_pretext_dataset True \
                          --do_train_self_weight True \
                          --self_train_size 500 \
                          --self_test_size 100 \ 
                          --self_weight_decay_level 5e-2 \
                          --self_learning_rate 1e-5 \
                          --self_batch_size 64 
                          
```
- Sample command line to run the downstream task
```shell
$ python main_downstream_car_class_classifier.py --do_train_downstream_classifier True \
                                                 --down_n_epochs 400 \
                                                 --down_batch_size 128 \
                                                 --down_learning_rate 1e-5 \
                                                 --down_weight_decay 5e-4 
```
- Sample command line to run the layer-wise CKA modules
```shell
$ python CKA-Centered-Kernel-Alignment/cka_main.py  # layerwise CKA 
$ python CKA-Centered-Kernel-Alignment/cka_lower_layer_higher.py # lower & higher layer's CKA
```
