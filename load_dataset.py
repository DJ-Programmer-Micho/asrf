import pandas as pd
import torchaudio
from datasets import Dataset, DatasetDict, Features, Value
from transformers import Wav2Vec2Processor
import os

def load_processed_dataset(config: dict, processor: Wav2Vec2Processor) -> DatasetDict:
    audio_col = config["dataset"]["audio_column"]
    text_col = config["dataset"]["transcript_column"]
    sampling_rate = config["dataset"]["sampling_rate"]

    def load_split(path):
        df = pd.read_csv(path)
        # Keep only the columns we care about
        df = df[[audio_col, text_col]]
        features = Features({
            audio_col: Value("string"),
            text_col: Value("string")
        })
        return Dataset.from_pandas(df, features=features)

    def prepare(example):
        audio_path = example[audio_col].lstrip("/")
        waveform, sr = torchaudio.load(os.path.join(config["dataset"]["audio_path"], audio_path))

        if sr != sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sampling_rate)
            sr = sampling_rate

        input_values = processor(waveform.squeeze().numpy(), sampling_rate=sr).input_values[0]
        with processor.as_target_processor():
            labels = processor(example[text_col]).input_ids
        return {
            "input_values": input_values,
            "labels": labels
        }

    return DatasetDict({
        "train": load_split(config["split_data"]["split_train_path"]).map(prepare),
        "validation": load_split(config["split_data"]["split_val_path"]).map(prepare),
        "test": load_split(config["split_data"]["split_test_path"]).map(prepare)
    })
