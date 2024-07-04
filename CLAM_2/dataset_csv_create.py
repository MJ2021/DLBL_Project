import os
import pandas as pd
import random

top = '/home/Drivessd2tb/dlbl_3'

data_record = {'full_path' : [], 'slide_id' : [], 'case_id' : [], 'slide_name' : [], 'label' : []}

for root, directories, files in os.walk(top, topdown=False):
    for name in files:
        # print(f' name is {name}')
        data_record['slide_name'].append(name)
        data_record['full_path'].append(os.path.join(root, name))
        filename = name.split(".")
        data_record['slide_id'].append(filename[0])
        filename2 = filename[0].split("__")
        data_record['case_id'].append(filename2[0])
        data_record['label'].append(random.choice(['A', 'B', 'C']))



df = pd.DataFrame(data_record)
df.to_csv('dlbl_data_new_3.csv', sep = ',', index = False)