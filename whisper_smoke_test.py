import numpy as np
import sounddevice as sd
import whisper_timestamped as whisper

MODEL = "small"
LANGUAGE = "fi"   # Finnish input
DURATION_SEC = 7
SR = 16000

def record():
    print("Recording...")
    audio = sd.rec(int(DURATION_SEC * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return audio.reshape(-1).astype("float32")

def main():
    model = whisper.load_model(MODEL, device="cpu")
    audio = record()
    audio = whisper.pad_or_trim(audio)

    result = whisper.transcribe(
        model=model,
        audio=audio,
        language=LANGUAGE,
        task="translate",  # Finnish -> English
        vad=True,
        condition_on_previous_text=False,
        verbose=False,
    )
    print("TEXT:", result.get("text", "").strip())

if __name__ == "__main__":
    main()