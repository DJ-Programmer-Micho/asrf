"""
Train Wav2Vec2 model on Kurdish Sorani ASR dataset using Hugging Face Trainer.
"""

import json
from dataset_utilities.prepare_dataset import prepare_datasets
from model_utilities.tokenizer_setup import load_processor
from model_utilities.load_model import load_model_and_processor
from model_utilities.trainer_setup import load_training_arguments, load_data_collator
from transformers import Trainer
from datasets import load_metric


def compute_metrics(pred):
    wer_metric = load_metric("wer")
    pred_ids = pred.predictions.argmax(-1)
    pred_str = processor.batch_decode(pred_ids)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


if __name__ == "__main__":
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load processor
    processor = load_processor(
        config["dataset"]["vocab_path"],
        config["dataset"]["sampling_rate"]
    )

    # Load model
    model, _ = load_model_and_processor(config)

    # Prepare datasets
    datasets = prepare_datasets(config)

    # Tokenize datasets
    def tokenize(batch):
        audio = batch["path"]
        inputs = processor(audio["array"], sampling_rate=config["dataset"]["sampling_rate"])
        with processor.as_target_processor():
            labels = processor(batch["transcript"]).input_ids
        inputs["labels"] = labels
        return inputs

    tokenized_datasets = datasets.map(tokenize, remove_columns=datasets["train"].column_names)

    # Load training args and collator
    training_args = load_training_arguments(config)
    data_collator = load_data_collator(processor, config)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=processor,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(config["model"]["output_dir"])
    processor.save_pretrained(config["model"]["output_dir"])
