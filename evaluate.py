"""
Evaluate fine-tuned Kurdish Sorani ASR model on test set.
"""

import json
from dataset_utilities.prepare_dataset import prepare_datasets
from model_utilities.load_model import load_model_and_processor
from datasets import load_metric
import numpy as np
import torch


def compute_wer(predictions, references):
    metric = load_metric("wer")
    return metric.compute(predictions=predictions, references=references)


if __name__ == "__main__":
    from model_utilities.tokenizer_setup import load_processor

    # Load config
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load fine-tuned model and processor
    model, processor = load_model_and_processor({
        "model": {
            "name": config["model"]["output_dir"],
            "cache_dir": config["model"]["output_dir"]
        }
    })

    model.eval()

    # Load dataset
    datasets = prepare_datasets(config)
    test_set = datasets["test"]

    predictions = []
    references = []

    for sample in test_set:
        with torch.no_grad():
            inputs = processor(sample["path"]["array"], return_tensors="pt", sampling_rate=16000)
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred_text = processor.batch_decode(pred_ids)[0]
            predictions.append(pred_text.strip())

        references.append(sample["transcript"].strip())

    wer = compute_wer(predictions, references)
    print(f"\n✅ Word Error Rate (WER): {wer:.3f}")
