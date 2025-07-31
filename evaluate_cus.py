"""
Evaluate fine-tuned Kurdish Sorani ASR model on test set.
"""

import json
from dataset_utilities.prepare_dataset import prepare_datasets
from model_utilities.load_model import load_model_and_processor
from model_utilities.tokenizer_setup import load_processor
import evaluate
import torch


def compute_wer(predictions, references):
    metric = evaluate.load("wer")
    return metric.compute(predictions=predictions, references=references)


if __name__ == "__main__":
    # Load configuration
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load tokenizer/processor
    processor = load_processor(
        vocab_path=config["dataset"]["vocab_path"],
        sampling_rate=config["dataset"]["sampling_rate"]
    )

    # Load fine-tuned model
    model, _ = load_model_and_processor({
        "model": {
            "name": config["model"]["output_dir"],
            "cache_dir": config["model"]["output_dir"]
        }
    })

    model.eval()

    # Prepare test dataset
    datasets = prepare_datasets(config)
    test_set = datasets["test"]

    predictions = []
    references = []

    for sample in test_set:
        with torch.no_grad():
            inputs = processor(
                sample[config["dataset"]["audio_column"]]["array"],
                return_tensors="pt",
                sampling_rate=config["dataset"]["sampling_rate"]
            )
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred_text = processor.batch_decode(pred_ids)[0].strip()
            predictions.append(pred_text)

        references.append(sample[config["dataset"]["transcript_column"]].strip())

    # Compute and show WER
    wer = compute_wer(predictions, references)
    print(f"\n✅ Word Error Rate (WER): {wer:.3f}")
