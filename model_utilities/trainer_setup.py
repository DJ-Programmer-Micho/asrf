"""
Setup Hugging Face Trainer components for Kurdish Sorani ASR.
Loads TrainingArguments and CTC data collator.
"""
from transformers import TrainingArguments
from data_collator import DataCollatorCTCWithPadding 

import os


def load_training_arguments(config: dict) -> TrainingArguments:
    """
    Load Hugging Face TrainingArguments from config.

    Args:
        config: Dictionary from config.json

    Returns:
        TrainingArguments object
    """
    args = config["training"]
    output_dir = config["model"]["output_dir"]

    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=args.get("evaluation_strategy", "steps"),
        learning_rate=args["learning_rate"],
        per_device_train_batch_size=args["per_device_train_batch_size"],
        per_device_eval_batch_size=args["per_device_eval_batch_size"],
        gradient_accumulation_steps=args.get("gradient_accumulation_steps", 1),
        num_train_epochs=args["num_train_epochs"],
        warmup_steps=args.get("warmup_steps", 0),
        logging_steps=args.get("logging_steps", 100),
        save_steps=args.get("save_steps", 1000),
        eval_steps=args.get("eval_steps", 1000),
        save_total_limit=args.get("save_total_limit", 3),
        fp16=args.get("fp16", True),
        load_best_model_at_end=args.get("load_best_model_at_end", True),
        metric_for_best_model=args.get("metric_for_best_model", "wer"),
        greater_is_better=args.get("greater_is_better", False),
        dataloader_num_workers=args.get("dataloader_num_workers", 4),
        group_by_length=args.get("group_by_length", True),
        remove_unused_columns=args.get("remove_unused_columns", False),
        report_to=["wandb"] if "wandb" in config and config["wandb"].get("project") else []
    )


def load_data_collator(processor, config: dict):
    """
    Create data collator for CTC with dynamic padding.

    Args:
        processor: Wav2Vec2Processor
        config: Dictionary from config.json

    Returns:
        DataCollatorCTCWithPadding object
    """
    collator_args = config.get("data_collator", {})
    return DataCollatorCTCWithPadding(
        processor=processor,
        padding=collator_args.get("padding", True),
        pad_to_multiple_of=collator_args.get("pad_to_multiple_of", None)
    )


if __name__ == "__main__":
    import json
    from model_utilities.tokenizer_setup import load_processor

    config_path = "./config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError("config.json not found")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    processor = load_processor(
        config["dataset"]["vocab_path"],
        config["dataset"]["sampling_rate"]
    )

    training_args = load_training_arguments(config)
    collator = load_data_collator(processor, config)

    print("TrainingArguments and DataCollator loaded successfully.")
