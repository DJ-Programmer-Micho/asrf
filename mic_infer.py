import torch
import librosa
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# === Config ===
model_path = "./model/finetuned"
file_path = "./rec/sample0075.wav"
sampling_rate = 16000

# === Load model and processor ===
print("🔄 Loading processor and model...")
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)
model.eval()

# === Load and normalize audio ===
print(f"📂 Loading audio file: {file_path}")
audio, rate = librosa.load(file_path, sr=sampling_rate)
audio = audio / max(abs(audio))  # Normalize amplitude

# === Preprocess input ===
inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

# === Inference ===
print("🔍 Running inference...")
with torch.no_grad():
    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

# === Debugging Info ===
probs = F.softmax(logits, dim=-1)
max_prob = probs.max().item()

print("\n📝 Kurdish Transcription:")
print(transcription.strip())
print(f"\n📈 Max confidence: {max_prob:.4f}")


import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Load model
model_path = "./model/finetuned"
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)

# Load audio
audio, sr = librosa.load("./rec/sample0075.wav", sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

# Inference
with torch.no_grad():
    logits = model(**inputs).logits  # Shape: (batch_size, time_steps, vocab_size)

# Check logits
print("Logits shape:", logits.shape)
print("Logits max value:", logits.max().item())
print("Logits min value:", logits.min().item())
print("Sample logits row (first 10 vocab scores of first timestep):")
print(logits[0, 0, :10])
