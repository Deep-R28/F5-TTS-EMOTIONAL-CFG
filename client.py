"""
client.py
Streams audio from server and plays on speaker via PyAudio FIFO queue.

Run: python client.py
"""

import queue
import threading
import requests
import pyaudio
import numpy as np

SERVER_URL  = "http://127.0.0.1:8000/tts-stream"
SAMPLE_RATE = 24000
CHANNELS    = 1
FORMAT      = pyaudio.paInt16
CHUNK_SIZE  = 1024

audio_queue = queue.Queue()
SENTINEL    = None


def player_thread():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=SAMPLE_RATE, output=True,
                    frames_per_buffer=CHUNK_SIZE)
    print("[player] Ready.")

    while True:
        chunk = audio_queue.get()
        if chunk is SENTINEL:
            print("[player] Stream complete.")
            break
        for i in range(0, len(chunk), CHUNK_SIZE * 2):
            stream.write(chunk[i: i + CHUNK_SIZE * 2])
        audio_queue.task_done()

    stream.stop_stream()
    stream.close()
    p.terminate()


def receive_and_queue(payload: dict):
    print(f"[receiver] voice    : {payload['voice']}")
    print(f"[receiver] lang     : {payload['lang_code']}")
    print(f"[receiver] emotion  : {payload['emotion']}")
    print(f"[receiver] text     : {payload['text']}\n")

    with requests.post(SERVER_URL, json=payload, stream=True) as resp:
        resp.raise_for_status()
        chunk_index = 0
        while True:
            size_bytes = resp.raw.read(4)
            if not size_bytes or len(size_bytes) < 4:
                break
            chunk_size = int.from_bytes(size_bytes, byteorder="little")
            pcm = resp.raw.read(chunk_size)
            if not pcm:
                break
            chunk_index += 1
            ms = int((len(pcm) / 2 / SAMPLE_RATE) * 1000)
            print(f"[receiver] chunk {chunk_index} → {ms}ms — queued")
            audio_queue.put(pcm)

    audio_queue.put(SENTINEL)
    print(f"\n[receiver] {chunk_index} chunks queued.")



    # ── Change these to test different scenarios ──
if __name__ == "__main__":

    payload = {
    "text": "Hello. This is a test.",   # ← very short, one phrase only
    "lang_code": "en",
    "voice": "indian_english",
    "emotion": "auto"
}

    t = threading.Thread(target=player_thread, daemon=True)
    t.start()

    receive_and_queue(payload)
    t.join()
    print("\n[done] Playback complete.")