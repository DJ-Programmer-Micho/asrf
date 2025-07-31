import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.config_loader import load_config
import pandas as pd

config = load_config()

csv_split_train = pd.read_csv(config["split_data"]["split_train_path"])
csv_split_test = pd.read_csv(config["split_data"]["split_test_path"])
csv_split_val = pd.read_csv(config["split_data"]["split_val_path"])

train_info = {
    "head": csv_split_train.head(),
    "dtypes": csv_split_train.dtypes,
    "null_counts": csv_split_train.isnull().sum()
}

validation_info = {
    "head": csv_split_val.head(),
    "dtypes": csv_split_val.dtypes,
    "null_counts": csv_split_val.isnull().sum()
}

test_info = {
    "head": csv_split_test.head(),
    "dtypes": csv_split_test.dtypes,
    "null_counts": csv_split_test.isnull().sum()
}

print(train_info, validation_info, test_info)