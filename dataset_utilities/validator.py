# dataset_utilities/validator.py

"""
Validate Kurdish Sorani ASR dataset for training readiness.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd
from normalizer import KurdishSoraniNormalizer
from utils.config_loader import load_config


def validate_metadata(csv_path=None, vocab_path=None):
    df = pd.read_csv(csv_path)
    normalizer = KurdishSoraniNormalizer()

    issues = {
        "missing_audio": [],
        "empty_transcript": [],
        "invalid_chars": [],
    }

    vocab = None
    if vocab_path and Path(vocab_path).exists():
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

    for _, row in df.iterrows():
        audio_path = row["path"]
        transcript = row["transcript"]

        if not Path(audio_path).exists():
            issues["missing_audio"].append(audio_path)

        if not isinstance(transcript, str) or len(transcript.strip()) == 0:
            issues["empty_transcript"].append(audio_path)

        stats = normalizer.validate_text(transcript)
        if not stats["is_valid"]:
            issues["invalid_chars"].append({
                "path": audio_path,
                "chars": stats["invalid_chars"]
            })

    print("Validation Summary:")
    print(f"- Missing audio files: {len(issues['missing_audio'])}")
    print(f"- Empty transcripts: {len(issues['empty_transcript'])}")
    print(f"- Transcripts with invalid characters: {len(issues['invalid_chars'])}")

    return issues


if __name__ == "__main__":
    import argparse
    config = load_config()
    csv_path = config["dataset"]["preprocessed_data_path_csv"]
    vocab_path = config["dataset"]["vocab_path"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=csv_path)
    parser.add_argument("--vocab", type=str, default=vocab_path)
    args = parser.parse_args()
    issues = validate_metadata(args.csv, args.vocab)

    # ✅ Show details of transcripts with invalid characters
    print(json.dumps(issues["invalid_chars"], indent=2, ensure_ascii=False))
    validate_metadata(args.csv, args.vocab)
