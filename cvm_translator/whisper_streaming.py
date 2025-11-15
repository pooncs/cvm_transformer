import whisper
import numpy as np
import sounddevice as sd
import threading
import queue
import time


class WhisperStreamer:
    def __init__(self, model_size="base", sample_rate=16000, chunk_duration=0.3):
        self.model = whisper.load_model(model_size)
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_duration * sample_rate)
        self.audio_q = queue.Queue()
        self.text_q = queue.Queue()
        self.running = False

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_q.put(indata.copy())

    def vad(self, chunk):
        rms = np.sqrt(np.mean(chunk ** 2))
        return rms > 0.01

    def transcribe_chunk(self, chunk):
        audio = chunk.astype(np.float32)
        result = self.model.transcribe(audio, language="ko", temperature=0.0, beam_size=1, fp16=False)
        text = result["text"].strip()
        return {"text": text, "confidence": 0.95 if text else 0.0}

    def run(self):
        self.running = True
        with sd.InputStream(callback=self.audio_callback, channels=1, samplerate=self.sample_rate):
            while self.running:
                try:
                    chunk = self.audio_q.get(timeout=0.3)
                except queue.Empty:
                    continue
                if not self.vad(chunk):
                    continue
                out = self.transcribe_chunk(chunk)
                if out["text"]:
                    self.text_q.put(out)

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def get(self, block=True, timeout=None):
        return self.text_q.get(block=block, timeout=timeout)


if __name__ == "__main__":
    ws = WhisperStreamer()
    ws.start()
    print("Listening... press Ctrl+C to stop")
    try:
        while True:
            print(ws.get())
    except KeyboardInterrupt:
        ws.stop()