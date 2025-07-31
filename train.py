from datasets import load_dataset, DatasetDict, Audio
from transformers import Trainer
from model_utilities.trainer_setup import load_training_arguments, load_data_collator
from model_utilities.load_model import load_model_and_processor
from compute_metrics import get_compute_metrics_fn
from load_dataset import load_processed_dataset
import json

if __name__ == "__main__":
    config_path = "./config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print("🔄 Loading model and processor...")
    model, processor = load_model_and_processor(config)

    print("📊 Loading datasets...")
    dataset = load_processed_dataset(config, processor)

    print("⚙️ Loading training arguments and data collator...")
    training_args = load_training_arguments(config)
    data_collator = load_data_collator(processor, config)

    print("🚀 Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor,
        data_collator=data_collator,
        # compute_metrics=get_compute_metrics_fn
        compute_metrics=get_compute_metrics_fn(processor)
    )

    trainer.train()
    trainer.save_model(config["model"]["output_dir"])
    processor.save_pretrained(config["model"]["output_dir"])
    print("✅ Training complete. Model and processor saved.")