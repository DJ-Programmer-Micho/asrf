import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from dataset_utilities.normalizer import normalize_transcript_file
import os
from utils.config_loader import load_config

config = load_config()
csv_in_dir_path = config["dataset"]["preprocessed_data_path"]
csv_in_path = config["dataset"]["original_data_path_csv"]
csv_out_path = config["dataset"]["preprocessed_data_path_csv"]
vocab_path = config["dataset"]["vocab_path"]
# Ensure the output directory exists
os.makedirs(csv_in_dir_path, exist_ok=True)

# Normalize the transcripts
normalize_transcript_file(
    input_file=csv_in_path,
    output_file=csv_out_path
)

print(f"✅ Normalization complete! Saved to {csv_out_path}")
