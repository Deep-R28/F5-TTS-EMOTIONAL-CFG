"""
audio_postprocessor.py
Post-processes raw F5-TTS output for broadcast-quality audio.
Pipeline: noise gate → EQ → compression → normalisation → smooth edges
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, sosfilt


SAMPLE_RATE = 24000


def butter_filter(audio: np.ndarray, cutoff, btype: str, order=5) -> np.ndarray:
    """Generic Butterworth filter."""
    nyq = SAMPLE_RATE / 2
    if isinstance(cutoff, (list, tuple)):
        norm = [c / nyq for c in cutoff]
    else:
        norm = cutoff / nyq
    sos = butter(order, norm, btype=btype, output='sos')
    return sosfilt(sos, audio)


def noise_gate(audio: np.ndarray, threshold_db: float = -45.0) -> np.ndarray:
    """Silence sections below threshold — removes model breathing artifacts."""
    threshold  = 10 ** (threshold_db / 20)
    frame_size = int(SAMPLE_RATE * 0.02)   # 20ms frames
    output     = audio.copy()
    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i + frame_size]
        rms   = np.sqrt(np.mean(frame ** 2))
        if rms < threshold:
            output[i:i + frame_size] *= 0.0
    return output


def eq_voice(audio: np.ndarray) -> np.ndarray:
    """
    3-band EQ optimised for Indian English voice:
    - High-pass  at 80Hz   → removes low rumble
    - Presence boost 2-5kHz → clarity and crispness
    - High shelf cut >8kHz  → reduce harshness
    """
    # High-pass — remove sub-bass rumble
    audio = butter_filter(audio, 80, 'high')

    # Presence boost — bandpass 2kHz-5kHz, add 30% of it back
    presence = butter_filter(audio, [2000, 5000], 'band')
    audio    = audio + 0.3 * presence

    # High shelf cut — reduce harshness above 8kHz
    high = butter_filter(audio, 8000, 'high')
    audio = audio - 0.2 * high

    return audio


def compress(audio: np.ndarray,
             threshold_db: float = -18.0,
             ratio: float = 3.0,
             attack_ms: float = 5.0,
             release_ms: float = 50.0) -> np.ndarray:
    """
    Dynamic range compressor — evens out loud/quiet parts.
    Makes speech more consistent like a professional recording.
    """
    threshold  = 10 ** (threshold_db / 20)
    attack     = int(SAMPLE_RATE * attack_ms / 1000)
    release    = int(SAMPLE_RATE * release_ms / 1000)
    gain       = np.ones(len(audio))
    envelope   = 0.0

    for i in range(len(audio)):
        level    = abs(audio[i])
        if level > envelope:
            envelope = envelope + (level - envelope) / max(attack, 1)
        else:
            envelope = envelope + (level - envelope) / max(release, 1)

        if envelope > threshold:
            target_gain = threshold + (envelope - threshold) / ratio
            gain[i]     = target_gain / max(envelope, 1e-8)
        else:
            gain[i] = 1.0

    return audio * gain


def normalise(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Normalise peak level to target dB."""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    target_linear = 10 ** (target_db / 20)
    return audio * (target_linear / peak)


def smooth_edges(audio: np.ndarray, fade_ms: float = 10.0) -> np.ndarray:
    """Fade in/out at start and end — removes clicks and pops."""
    fade_samples = int(SAMPLE_RATE * fade_ms / 1000)
    fade_in      = np.linspace(0, 1, fade_samples)
    fade_out     = np.linspace(1, 0, fade_samples)
    audio[:fade_samples]  *= fade_in
    audio[-fade_samples:] *= fade_out
    return audio


def add_silence_padding(audio: np.ndarray, ms: float = 80.0) -> np.ndarray:
    """Add small silence gap between phrases — improves naturalness."""
    pad = np.zeros(int(SAMPLE_RATE * ms / 1000))
    return np.concatenate([audio, pad])


def postprocess(audio: np.ndarray) -> np.ndarray:
    """
    Full post-processing pipeline.
    Input:  raw float32 numpy array from F5-TTS
    Output: processed float32 numpy array ready for playback
    """
    audio = audio.astype(np.float32)
    audio = noise_gate(audio)           # 1. remove artifacts
    audio = eq_voice(audio)             # 2. voice EQ
    audio = compress(audio)             # 3. even out dynamics
    audio = normalise(audio)            # 4. consistent volume
    audio = smooth_edges(audio)         # 5. remove clicks
    audio = add_silence_padding(audio)  # 6. natural phrase gap
    audio = np.clip(audio, -1.0, 1.0)  # 7. final clip guard
    return audio


if __name__ == "__main__":
    import soundfile as sf

    # Test on a WAV file
    audio, sr = sf.read("voices/ref_indian_en.wav")
    processed = postprocess(audio)
    sf.write("test_postprocessed.wav", processed, sr)
    print("✓ Saved test_postprocessed.wav")