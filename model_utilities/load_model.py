"""
Load Wav2Vec2 model and processor for training or inference.
"""

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os


def load_model_and_processor(config: dict):
    """
    Load pretrained Wav2Vec2 model and processor from config.
    
    Args:
        config: Dictionary from config.json
    
    Returns:
        model: Wav2Vec2ForCTC
        processor: Wav2Vec2Processor
    """
    model_name_or_path = config["model"]["name"]
    processor_path = config["model"]["cache_dir"]

    # Load model (can be 'facebook/wav2vec2-large-xlsr-53' or path to local model)
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name_or_path,
        cache_dir=processor_path,
        ignore_mismatched_sizes=True  # useful if tokenizer mismatch
    )

    # Load processor (tokenizer + feature extractor)
    processor = Wav2Vec2Processor.from_pretrained(processor_path)

    print(f"Model loaded from: {model_name_or_path}")
    print(f"Processor loaded from: {processor_path}")
    return model, processor


if __name__ == "__main__":
    import json

    config_path = "./config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError("config.json not found")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model, processor = load_model_and_processor(config)
