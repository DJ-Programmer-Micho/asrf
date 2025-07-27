"""
Prepare the Kurdish Sorani ASR dataset for training with Hugging Face.
"""

import pandas as pd
from datasets import Dataset, Audio, DatasetDict
from transformers import Wav2Vec2Processor
from pathlib import Path
import argparse
from normalizer import KurdishSoraniNormalizer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.config_loader import load_config


def load_and_prepare_split(csv_path, sampling_rate, normalizer=None):
    df = pd.read_csv(csv_path)

    if normalizer:
        df['transcript'] = df['transcript'].apply(normalizer.normalize_text)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("path", Audio(sampling_rate=sampling_rate))
    return dataset


def prepare_datasets(config):
    config = load_config()
    
    csv_split_train = config["split_data"]["split_train_path"]
    csv_split_test = config["split_data"]["split_test_path"]
    csv_split_val = config["split_data"]["split_val_path"]
    normalizer = KurdishSoraniNormalizer()

    train_ds = load_and_prepare_split(
        csv_split_train,
        config['dataset']['sampling_rate'],
        normalizer
    )
    val_ds = load_and_prepare_split(
        csv_split_val,
        config['dataset']['sampling_rate'],
        normalizer
    )
    test_ds = load_and_prepare_split(
        csv_split_test,
        config['dataset']['sampling_rate'],
        normalizer
    )

    return DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })


if __name__ == "__main__":
    import json
    with open("./config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    datasets = prepare_datasets(config)
    print(datasets)
