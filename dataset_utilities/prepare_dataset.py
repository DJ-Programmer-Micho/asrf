"""
Prepare the Kurdish Sorani ASR dataset for training with Hugging Face.
"""

import pandas as pd
from datasets import Dataset, Audio, DatasetDict
from transformers import Wav2Vec2Processor
from pathlib import Path
import argparse
from normalizer import KurdishSoraniNormalizer


def load_and_prepare_split(csv_path, sampling_rate, normalizer=None):
    df = pd.read_csv(csv_path)

    if normalizer:
        df['transcript'] = df['transcript'].apply(normalizer.normalize_text)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("path", Audio(sampling_rate=sampling_rate))
    return dataset


def prepare_datasets(config):
    normalizer = KurdishSoraniNormalizer()

    train_ds = load_and_prepare_split(
        config['dataset']['preprocessed_data_path'] + "/train.csv",
        config['dataset']['sampling_rate'],
        normalizer
    )
    val_ds = load_and_prepare_split(
        config['dataset']['preprocessed_data_path'] + "/validation.csv",
        config['dataset']['sampling_rate'],
        normalizer
    )
    test_ds = load_and_prepare_split(
        config['dataset']['preprocessed_data_path'] + "/test.csv",
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
