from dataset_utilities.normalizer import normalize_transcript_file
import os

# Ensure the output directory exists
os.makedirs("dataset/preprocessed_data", exist_ok=True)

# Normalize the transcripts
normalize_transcript_file(
    input_file="dataset/original_data/metadata.csv",
    output_file="dataset/preprocessed_data/metadata.csv"
)

print("✅ Normalization complete! Saved to dataset/preprocessed_data/metadata.csv")
