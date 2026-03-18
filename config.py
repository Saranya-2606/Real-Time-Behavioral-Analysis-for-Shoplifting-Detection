import os

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Data directory
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Dataset directory
DATASET_PATH = os.path.join(DATA_DIR, 'dataset_path')
SUS_PATH = os.path.join(DATASET_PATH, 'Suspicious')
NORMAL_PATH = os.path.join(DATASET_PATH, 'Normal')
RAW_IMAGES_DIR = os.path.join(DATA_DIR, 'raw_images')

os.makedirs(SUS_PATH, exist_ok=True)
os.makedirs(NORMAL_PATH, exist_ok=True)
os.makedirs(RAW_IMAGES_DIR, exist_ok=True)

# Common paths
CSV_FILE_PATH = os.path.join(PROJECT_ROOT, 'nkeypoint.csv')
DATASET_CSV_PATH = os.path.join(DATASET_PATH, 'dataset.csv')
