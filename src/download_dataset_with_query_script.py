import pandas as pd
import numpy as np
import os
import urllib.request

# 차종 종류 파악
car_info_types = pd.read_csv('./car_info_types.csv')

# 필터링 걸고자 하는 차량 id가 있는 경우
filtering = True 
if filtering :
#     class_ids = [1, 65] # 2 class: ay v. stonic
#     class_ids = [1, 65, 136, 76, 145, 184, 151, 106, 73, 70] # 10 class
    class_ids = [75, 87, 68, 78, 26, 49, 27, 51, 67, 81] # 20 class (추가한 부분만 기재)
    
    
else :
    class_ids = car_info_types['class_id'].values
    


'''
input; target_class_id (int)
output; query (string)
'''

def create_query(target_class_id):

    query = f"""
        WITH car_info AS (
            SELECT *
            FROM `socar-data.tianjin_replica.car_info`
            WHERE (class_id = {target_class_id})
        )

        SELECT
        csi.*,
        CONCAT(csi.upload_host, csi.filename) AS image_url, 
        car_info.class_id,
        car_info.car_name

        FROM `socar-data.tianjin_replica.car_cs_info` AS csi

        LEFT JOIN car_info
        ON csi.car_id = car_info.id

        WHERE car_name is not null
        
        LIMIT 1500
    """
    
    return query

save_base = '../dataset/'

nums = list()

for class_id in class_ids :
    
    class_id = class_id
    car_name = car_info_types[car_info_types['class_id']==class_id]['car_name'].values[0]
    num_data = car_info_types[car_info_types['class_id']==class_id]['number_at_1000'].values[0]
    
    if num_data == 1000:
        query = create_query(class_id)

        # download the dataset with query
        df = pd.read_gbq(query=query, project_id='socar-data', dialect='standard')
        
        filenames = df['image_url'].values
        
        save_path = os.path.join(save_base, str(class_id) + '_' + car_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for filename in filenames:
            try:
                save_filepath = os.path.join(save_path, filename.split('/')[-1])
                urllib.request.urlretrieve(filename, save_filepath)
                
            except Exception as e:
                pass
            
    else:
        pass