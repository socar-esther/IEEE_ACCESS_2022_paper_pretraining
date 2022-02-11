import os
from shutil import copyfile
import cv2


def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))

            
def create_self_supervised_set(base_dir, target_data_name, self_train_num_per_class, self_test_num_per_class):
    ## 하나의 파일로 합치는 작업
#     dataroot = '../../../dataset'
    dataroot = base_dir
    target_names = os.listdir(base_dir)
    print('>> check class : ', os.listdir(base_dir))
    
    file_train_names_list = list()
    file_test_names_list = list()
    
    for target_name in target_names : 
#         target_path = os.path.join(dataroot, target_name)
        target_path = os.path.join(base_dir, target_name)
        target_train_filenames = sorted(os.listdir(target_path))[:self_train_num_per_class]
        target_test_filenames = sorted(os.listdir(target_path))[self_train_num_per_class : self_train_num_per_class+self_test_num_per_class]
    
        
        for target_filename in target_train_filenames : 
            target_filepath = os.path.join(target_path, target_filename)
            file_train_names_list.append(target_filepath)
        for test_filename in target_test_filenames : 
            target_filepath = os.path.join(target_path, test_filename)
            file_test_names_list.append(target_filepath)

    # 저장할 파일 생성
    save_base = os.path.join('./dataset', target_data_name, 'train_size_' + str(self_train_num_per_class))
    
    train_save_dir_out = os.path.join(save_base, 'train/')
    train_save_dir = os.path.join(save_base, 'train/images/')
    
    if not os.path.exists(train_save_dir) :
        os.makedirs(train_save_dir)
        
    # train 파일 저장
    for src_filename in file_train_names_list:
        
        try:
            img = cv2.imread(src_filename)
            cv2.imwrite(os.path.join(train_save_dir, src_filename.split('/')[-1]), img)
        except Exception as e:
            print(src_filename)
            print(e)

    
    # 저장할 파일 생성
    test_save_dir_out = os.path.join(save_base, 'test/')
    test_save_dir = os.path.join(save_base, 'test/images/')
    
    if not os.path.exists(test_save_dir) :
        os.makedirs(test_save_dir)
    # test 파일 저장
    for src_filename_test in file_test_names_list:
        try:
            img_test = cv2.imread(src_filename_test)
            cv2.imwrite(os.path.join(test_save_dir, src_filename_test.split('/')[-1]), img_test)
        except Exception as e:
            print(src_filename)
            print(e)
    
    return train_save_dir_out, test_save_dir_out
    
    

        
    

            
