"""
prepare_dataset.py

Creates a SMALL training dataset for quick F5-TTS fine-tuning.

Uses:
1. indian_accent_english (50%)
2. processed_english_emotions_tagged (50%)

Output:
data/indian_tts/
├── wavs/
├── metadata.csv
└── vocab.txt
"""

import io
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset, Audio

OUTPUT_DIR = Path("data/indian_tts")
WAV_DIR = OUTPUT_DIR / "wavs"
METADATA_CSV = OUTPUT_DIR / "metadata.csv"
VOCAB_FILE = OUTPUT_DIR / "vocab.txt"

TARGET_SR = 24000

WAV_DIR.mkdir(parents=True, exist_ok=True)

rows = []
all_chars = set()

EMOTION_MAP = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "anger": "Angry",
    "neutral": "Neutral",
    "surprise": "Surprise",
    "surprised": "Surprise",
    "fear": "Neutral",
    "disgust": "Neutral",
    "default": "Neutral",
}


def clean_text(text):
    if text is None:
        return ""
    return str(text).strip().replace("|", " ").replace("\n", " ")


def save_wav(audio_array, sr, path):
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    audio_array = audio_array.astype(np.float32)

    if sr != TARGET_SR:
        audio_array = librosa.resample(
            audio_array,
            orig_sr=sr,
            target_sr=TARGET_SR
        )

    sf.write(str(path), audio_array, TARGET_SR)


def load_audio(sample):
    audio_info = sample["audio"]

    if "bytes" in audio_info and audio_info["bytes"] is not None:
        return sf.read(io.BytesIO(audio_info["bytes"]))

    if "path" in audio_info and audio_info["path"] is not None:
        return sf.read(audio_info["path"])

    raise ValueError("Audio not found")


# ==========================================================
# DATASET 1
# ==========================================================

print("\nLoading indian_accent_english...")

ds = load_dataset(
    "En1gma02/indian_accent_english",
    split="train"
)

ds = ds.cast_column("audio", Audio(decode=False))

limit = len(ds) // 2

print(f"Using {limit} / {len(ds)} samples")

for i in range(limit):

    sample = ds[i]

    try:
        text = clean_text(
            sample.get("text")
            or sample.get("transcription")
            or ""
        )

        if not text:
            continue

        audio, sr = load_audio(sample)

        fname = f"indian_en_{i:06d}.wav"
        save_wav(audio, sr, WAV_DIR / fname)

        rows.append({
            "filename": fname,
            "text": text,
            "emotion": "Neutral",
            "language": "en-IN",
            "speaker": "indian_en"
        })

        all_chars.update(text)

    except Exception as e:
        print(f"Skipped {i}: {e}")

print("Done with indian_accent_english")


# ==========================================================
# DATASET 2
# ==========================================================

print("\nLoading processed_english_emotions_tagged...")

ds = load_dataset(
    "En1gma02/processed_english_emotions_tagged",
    split="train"
)

ds = ds.cast_column("audio", Audio(decode=False))

limit = len(ds) // 2

print(f"Using {limit} / {len(ds)} samples")

for i in range(limit):

    sample = ds[i]

    try:
        text = clean_text(
            sample.get("text")
            or sample.get("transcription")
            or ""
        )

        if not text:
            continue

        audio, sr = load_audio(sample)

        raw_emotion = (
            sample.get("emotion")
            or sample.get("emotion_tag")
            or sample.get("label")
            or "neutral"
        )

        emotion = EMOTION_MAP.get(
            str(raw_emotion).lower(),
            "Neutral"
        )

        fname = f"emotion_en_{i:06d}.wav"

        save_wav(audio, sr, WAV_DIR / fname)

        rows.append({
            "filename": fname,
            "text": text,
            "emotion": emotion,
            "language": "en",
            "speaker": "emotion_en"
        })

        all_chars.update(text)

    except Exception as e:
        print(f"Skipped {i}: {e}")

print("Done with emotion dataset")


# ==========================================================
# METADATA
# ==========================================================

with open(
    METADATA_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "text",
            "emotion",
            "language",
            "speaker"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


# ==========================================================
# VOCAB
# ==========================================================

vocab = sorted(
    c for c in all_chars
    if c.strip()
)

VOCAB_FILE.write_text(
    "\n".join(vocab),
    encoding="utf-8"
)

print("\n================================")
print("DONE")
print("================================")
print("Samples:", len(rows))
print("Wavs:", len(list(WAV_DIR.glob('*.wav'))))
print("Metadata:", METADATA_CSV)
print("Vocab:", VOCAB_FILE)
print("================================")