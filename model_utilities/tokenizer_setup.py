"""
Load the custom Kurdish Sorani tokenizer and build Wav2Vec2Processor.
"""

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import os


def load_processor(vocab_path: str, sampling_rate: int = 16000, cache_dir: str = None) -> Wav2Vec2Processor:
    """
    Load tokenizer and feature extractor into a Wav2Vec2Processor.
    
    Args:
        vocab_path: Path to vocab.json
        sampling_rate: Audio sampling rate (default 16000)
        cache_dir: Optional cache directory for Hugging Face files
    
    Returns:
        processor: Wav2Vec2Processor ready for training or inference
    """
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        do_lower_case=False
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=sampling_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    print(f"Loaded tokenizer + feature extractor from: {vocab_path}")
    return processor


if __name__ == "__main__":
    import json

    # Load from config.json
    config_path = "./config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError("config.json not found")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    vocab_path = config["dataset"]["vocab_path"]
    sampling_rate = config["dataset"].get("sampling_rate", 16000)

    processor = load_processor(vocab_path, sampling_rate)

    # Save processor to use later
    save_path = config["model"]["cache_dir"]
    processor.save_pretrained(save_path)
    print(f"Processor saved to {save_path}")
