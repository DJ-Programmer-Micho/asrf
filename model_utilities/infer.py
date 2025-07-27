"""
Run inference using the fine-tuned Kurdish Sorani ASR model.
"""

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pathlib import Path


def transcribe(audio_path: str, model_dir: str, sampling_rate: int = 16000) -> str:
    """
    Transcribe a single WAV file using fine-tuned model.

    Args:
        audio_path: Path to .wav file
        model_dir: Path to fine-tuned model directory
        sampling_rate: Target audio sampling rate (default 16000)

    Returns:
        Transcribed text
    """
    # Load model and processor
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    model.eval()

    # Load and resample audio
    speech_array, sr = torchaudio.load(audio_path)
    if sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
        speech_array = resampler(speech_array)

    # Mono channel
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)

    input_values = processor(
        speech_array.squeeze().numpy(),
        return_tensors="pt",
        sampling_rate=sampling_rate
    ).input_values

    # Predict
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

    return transcription.strip()


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Transcribe a WAV file using fine-tuned Kurdish ASR model.")
    parser.add_argument("--audio", required=True, help="Path to input .wav file")
    parser.add_argument("--model_dir", default="./model/finetuned", help="Path to fine-tuned model directory")
    parser.add_argument("--sr", type=int, default=16000, help="Target sampling rate (default: 16000)")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    if not Path(args.model_dir).exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    print("Transcribing...")
    result = transcribe(args.audio, args.model_dir, sampling_rate=args.sr)
    print(f"\n🗣️  Transcription:\n{result}")
