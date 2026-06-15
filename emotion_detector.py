"""
emotion_detector.py
Detects emotion label from text using zero-shot classification.
Loaded once, reused per request.
"""

from transformers import pipeline

EMOTIONS = ["Happy", "Sad", "Angry", "Neutral", "Surprise"]

# Uses a lightweight zero-shot model — no GPU needed
_classifier = pipeline(
    "zero-shot-classification",
    model="cross-encoder/nli-MiniLM2-L6-H768",
    device=-1,   # CPU is fine for classification
)

def detect_emotion(text: str) -> str:
    """
    Returns one of: Happy, Sad, Angry, Neutral, Surprise
    Falls back to Neutral on short/ambiguous text.
    """
    if len(text.split()) < 3:
        return "Neutral"

    result = _classifier(text, candidate_labels=EMOTIONS)
    top    = result["labels"][0]
    score  = result["scores"][0]

    # Only use detected emotion if confidence is reasonable
    return top if score > 0.40 else "Neutral"


if __name__ == "__main__":
    tests = [
        "I just got promoted! This is the best day of my life!",
        "My dog passed away yesterday. I miss him so much.",
        "How dare you lie to me again!",
        "The meeting is scheduled for 3pm tomorrow."
    ]
    for t in tests:
        print(f"{detect_emotion(t):10s} ← {t}")

