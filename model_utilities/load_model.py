"""
Load Wav2Vec2 model and processor for training or inference.
"""

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os

def load_model_and_processor(config: dict):
    pretrained_path = config["model"]["pretrained_name_or_path"]
    processor_path = config["model"]["model_process"]
    cache_dir = config["model"].get("cache_dir", None)

    # Load pretrained model weights
    model = Wav2Vec2ForCTC.from_pretrained(
        pretrained_path,
        cache_dir=cache_dir,
        ignore_mismatched_sizes=True  # helps in case classifier layer doesn't match
    )
    model.gradient_checkpointing_enable()
    # Load custom processor (tokenizer + feature extractor)
    processor = Wav2Vec2Processor.from_pretrained(processor_path)

    print(f"✅ Model loaded from: {pretrained_path}")
    print(f"✅ Processor loaded from: {processor_path}")
    return model, processor


if __name__ == "__main__":
    import json

    
    config_path = "../config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError("config.json not found")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model, processor = load_model_and_processor(config)
