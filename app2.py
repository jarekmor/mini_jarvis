import os
import wave
import pyaudio
import webrtcvad
import openai
from dotenv import load_dotenv
from elevenlabs import set_api_key, generate, stream

# Set encoding for standard output
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# Set API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
set_api_key(os.getenv("ELEVEN_API_KEY"))

# Recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 32000
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "temp_audio.wav"
RECORD_SECONDS = 10

# Check if the audio contains human voice
def contains_human_voice(wav_filename, threshold=0.05):
    vad = webrtcvad.Vad(3)
    voice_frames = 0
    total_frames = 0

    with wave.open(wav_filename, 'rb') as wf:
        sample_rate = wf.getframerate()
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("Sample rate must be one of: 8000, 16000, 32000 or 48000.")
        num_samples = int(sample_rate * 0.02)
        samples = wf.readframes(num_samples)
        while len(samples) == num_samples * 2:
            is_speech = vad.is_speech(samples, sample_rate)
            if is_speech:
                voice_frames += 1
            total_frames += 1
            samples = wf.readframes(num_samples)
    voice_percentage = voice_frames / total_frames
    return voice_percentage >= threshold

# Record audio from microphone and return transcript
def record_from_mic():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording...")
    frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS))]
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    with open(WAVE_OUTPUT_FILENAME, 'rb') as file:
        transcript = openai.Audio.transcribe(model="whisper-1", file=file)
        print(transcript["text"], "\n\n")
    return transcript

# Generate voice response and stream it to the user
def generate_voice_response(transcript):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Jesteś pomocnym asystentem."},
            {"role": "user", "content": "Odpowiedz zwięźle i krótko na następujące pytanie: " + transcript["text"] + "?"}
        ],
        max_tokens=200,
        temperature=0.7,
        stop=["."]
    )
    resp_text = completion.choices[0].message["content"] + "."
    print(resp_text)
    
    words = resp_text.split()
    chunks = []
    while words:
        chunk = words[:5]
        words = words[5:]
        chunks.append(' '.join(chunk))
        
    def text_stream():
        for chunk in chunks:
            yield chunk

    audio_stream = generate(
        text=text_stream(),
        voice="Bella",
        model="eleven_multilingual_v2",
        stream=True
    )
    stream(audio_stream)

# Main block
if __name__ == "__main__":
    
    while True:
        transcript = record_from_mic()
        if contains_human_voice(WAVE_OUTPUT_FILENAME, 0.20):
            print("At least 20% of the recording contains human voice!")
            generate_voice_response(transcript)
            # break
        else:
            print("No human voice detected in the recording. Starting recording again...")
            continue
