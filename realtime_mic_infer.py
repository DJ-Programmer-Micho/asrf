import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
from datetime import datetime
import torch
import librosa
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# === CONFIGURATION ===
SAMPLING_RATE = 16000
DURATION = 5  # seconds
OUTPUT_DIR = "./rec"
MODEL_DIR = "./model/finetuned"

# === LOAD MODEL AND PROCESSOR ===
print("🔄 Loading model and processor...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("✅ Model and processor ready.\n")

# === STEP 1: List and Select Device ===
print("🎤 Available input devices:")
devices = sd.query_devices()
input_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]
for i in input_devices:
    print(f"  [{i}] {devices[i]['name']}")

device_id = int(input("🔧 Select input device ID: "))

# === STEP 2: Record ===
print(f"\n🎙️ Recording {DURATION} seconds from device ID {device_id}...")
recording = sd.rec(
    int(DURATION * SAMPLING_RATE),
    samplerate=SAMPLING_RATE,
    channels=1,
    dtype='float32',
    device=device_id
)
sd.wait()
recording = np.squeeze(recording)

# === STEP 3: Save to WAV ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(OUTPUT_DIR, f"audio_{timestamp}.wav")
wav.write(output_path, SAMPLING_RATE, (recording * 32767).astype(np.int16))
print(f"✅ Audio saved to: {output_path}")

# === STEP 4: Transcribe ===
print("\n🔍 Transcribing your speech...")
input_audio, sr = sf.read(output_path)
if sr != SAMPLING_RATE:
    input_audio = librosa.resample(input_audio, orig_sr=sr, target_sr=SAMPLING_RATE)

inputs = processor(input_audio, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(inputs.input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

print(f"\n📝 Kurdish Transcription:\n{transcription}")
