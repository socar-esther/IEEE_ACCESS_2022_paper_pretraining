# kdd 2022 paper 1675
Implementation of the submission to KDD'22: "Discovering the Effectiveness of Pre-Training for Image Recognition in a Large-scale Car-sharing Platform"

## Paper Abstract

Recent progress of deep learning has empowered various intelligent transportation applications, especially in car-sharing platforms. While the traditional operations of the car-sharing service highly relied on human engagements in fleet management, modern car-sharing platforms let users upload car images before and after their use to inspect the cars without a physical visit. To automate the aforementioned inspection task, prior approaches utilized deep neural networks. They commonly employed pre-training, a de-facto technique to establish an effective model under the limited number of labeled datasets. As candidate practitioners who deal with car images would presumably get suffered from the lack of a labeled dataset, we analyzed a sophisticated analogy into the effectiveness of pre-training is important. However, prior studies primarily shed a little spotlight on the effectiveness of pre-training. Motivated by the aforementioned lack of analysis, our study proposes a series of analyses to unveil the effectiveness of various pre-training methods in image recognition tasks at the car-sharing platform. We set two real-world image recognition tasks in the car-sharing platform in a live service, established them under the many-shot and few-shot problem settings, and scrutinized which pre-training method accomplishes the most effective performance in which setting. Furthermore, we analyzed how does the pre-training and fine-tuning convey different knowledge to the neural networks for a precise understanding. We highly expect candidate practitioners can utilize the proposed takeaways to solve their real-world image recognition problem in various settings.


## Prepare Dataset

Please visit the Github Repository (https://github.com/socar-kp/sofar_image_dataset) to check sample images utilized in this paper or acquire a full access to the dataset.

If you aim to reproduce the study, we recommend you to submit a request form to the dataset in the aforementioned Github Repository.

In case of any problems or inquiries, please raise the Issue.


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
