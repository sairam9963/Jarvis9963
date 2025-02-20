import sounddevice as sd
import queue
import vosk
from gtts import gTTS
import os

q = queue.Queue()
def callback(indata, frames, time, status):
    q.put(bytes(indata))

def listen():
    model = vosk.Model("model")  # Download from Vosk models page
    recognizer = vosk.KaldiRecognizer(model, 16000)
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                return recognizer.Result()

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("mpg321 response.mp3")
