"""
main_stream.py
Streaming FastAPI server using fine-tuned F5-TTS (base SWivid fork).
Emotion is injected into the text prompt — no special conditioning needed.

Run:
  PYTORCH_ENABLE_MPS_FALLBACK=1 uvicorn main_stream:app --reload
"""

import re
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from f5_tts.api import F5TTS
from text_preprocessor  import preprocess_text
from audio_postprocessor import postprocess

from emotion_detector import detect_emotion

app = FastAPI()

SAMPLE_RATE = 24000

# Fine-tuned checkpoint
CHECKPOINT = "ckpts/indian_tts/model_last.pt"

# Vocab from your fine-tuned dataset
VOCAB = "data/indian_tts_pinyin/vocab.txt"

# Load model with your checkpoint + vocab
f5tts = F5TTS(
    ckpt_file   = CHECKPOINT,
    vocab_file  = VOCAB,
)

# Reference voice
VOICE_MAP = {
    "indian_english": {
        "ref_audio": "voices/finetuned_new4.wav",
        "ref_text":  "He would not part with the bag for a single moment.",  # ← from Step 2
    },
}

# Emotion prefix map — injected into text so model
# produces appropriate prosody (works without special conditioning)
EMOTION_PREFIX = {
    "Happy":    "Joyfully: ",
    "Sad":      "Sadly: ",
    "Angry":    "Angrily: ",
    "Surprise": "Surprisingly: ",
    "Neutral":  "",
}

# Load model once at startup
print("Loading F5-TTS model...")
f5tts = F5TTS(ckpt_file=CHECKPOINT)
print("✓ Model loaded")


class TTSRequest(BaseModel):
    text:         str
    lang_code:    str = "en"
    voice:        str = "indian_english"
    emotion:      str = "auto"   # "auto" | "Happy" | "Sad" | "Angry" | "Neutral" | "Surprise"


def split_phrases(text: str) -> list[str]:
    """Split on sentence + clause boundaries for low-latency streaming."""
    phrases = re.split(r'(?<=[.!?,;:])\s+', text.strip())
    return [p.strip() for p in phrases if p.strip()]



def generate_chunks(request: TTSRequest):
    voice   = VOICE_MAP[request.voice]
    phrases = split_phrases(request.text)

    for phrase in phrases:
        # Layer 1 — text preprocessing
        emotion      = detect_emotion(phrase) if request.emotion == "auto" \
                       else request.emotion
        prefix       = EMOTION_PREFIX.get(emotion, "")
        clean_phrase = preprocess_text(prefix + phrase)

        print(f"  [{emotion}] {clean_phrase}")

        # Layer 2 — inference with quality hyperparameters
        wav, sr, _ = f5tts.infer(
            ref_file           = voice["ref_audio"],
            ref_text           = voice["ref_text"],
            gen_text           = clean_phrase,
            seed               = -1,
            nfe_step           = 32,
            cfg_strength       = 2.0,
            sway_sampling_coef = -1.0,
            speed              = 0.95,
        )

        if hasattr(wav, "numpy"):
            wav = wav.numpy()

        # Layer 3 — audio post-processing
        wav = postprocess(wav)

        # Convert to int16 PCM bytes
        audio_int16 = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
        raw_bytes   = audio_int16.tobytes()
        yield len(raw_bytes).to_bytes(4, byteorder="little") + raw_bytes



@app.post("/tts-stream")
def synthesize_stream(request: TTSRequest):
    if request.voice not in VOICE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{request.voice}'. Available: {list(VOICE_MAP)}"
        )
    return StreamingResponse(
        generate_chunks(request),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(SAMPLE_RATE)},
    )


@app.get("/voices")
def list_voices():
    return list(VOICE_MAP.keys())


@app.get("/emotions")
def list_emotions():
    return ["auto", "Happy", "Sad", "Angry", "Neutral", "Surprise"]