import os
import pandas as pd
from config import CSV_FILE_PATH, DATASET_PATH, SUS_PATH, NORMAL_PATH


df = pd.read_csv(CSV_FILE_PATH)

dataset_path = DATASET_PATH
sus_path = SUS_PATH
normal_path = NORMAL_PATH

def get_label(image_name, sus_path, normal_path):
    if image_name in os.listdir(sus_path):
        return 'Suspicious'
    elif image_name in os.listdir(normal_path):
        return 'Normal'
    else:
        return None 

df['label'] = df['image_name'].apply(lambda x: get_label(x, sus_path, normal_path))
df.to_csv(os.path.join(dataset_path, 'dataset.csv'), index=False)