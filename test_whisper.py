import whisper
import pyaudio
import numpy as np
import wave
import time

# Load Whisper model
model = whisper.load_model("tiny")

# Audio settings
CHUNK = 1024  # Buffer size
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Mono
RATE = 16000  # Whisper works best at 16kHz
RECORD_SECONDS = 10  # Fixed recording duration

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

# WAV file setup
output_filename = "live_audio.wav"
wf = wave.open(output_filename, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)

frames = []

# Start recording
print("Recording started... Speak into the microphone.")
start_time = time.time()

while time.time() - start_time < RECORD_SECONDS:
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)
    wf.writeframes(data)  # Save to WAV file

# Stop recording
wf.close()
stream.stop_stream()
stream.close()
audio.terminate()
print("Recording completed! Saved as", output_filename)

# Convert audio to numpy array for transcription
audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0

# Transcribe using Whisper
print("\nProcessing transcription...\n")
result = model.transcribe(audio_data, fp16=False)
print("You said:", result["text"], "\n")