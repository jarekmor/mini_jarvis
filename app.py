import pyaudio
import wave
import os

import sys
sys.stdout.reconfigure(encoding='utf-8')

import openai
from elevenlabs import Voice, VoiceSettings
from elevenlabs import set_api_key

from dotenv import load_dotenv
load_dotenv()

# Set OpenAI and Elevemlabs API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
set_api_key(os.getenv("ELEVEN_API_KEY"))

# Parameters for recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "temp_audio.wav"
RECORD_SECONDS = 5

audio = pyaudio.PyAudio()

# Start recording
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
print("Recording...")

frames = []
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Finished recording.")
stream.stop_stream()
stream.close()
audio.terminate()

with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

with open(WAVE_OUTPUT_FILENAME, 'rb') as f:
    file=open(WAVE_OUTPUT_FILENAME, 'rb')
    transcript = openai.Audio.transcribe(model="whisper-1", file=file)
    print(transcript)

# Generate OpenAI response
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "Jesteś pomocnym asystentem."},
    {"role": "user", "content": "Odpowiedz na następujące pytanie: " + transcript["text"] + "?"}
  ],
  max_tokens=100,
  temperature=0.7,
  stop=[".", "("]
)
resp_text = completion.choices[0].message["content"] + "."
print(resp_text)

# Generate Bella voice :-)
from elevenlabs import generate, stream
audio_stream = generate(
  text=resp_text,
  voice="Bella",
  model="eleven_multilingual_v2",
  stream=True
)
stream(audio_stream)
