import os
import tempfile
import streamlit as st
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile

model = whisper.load_model("base")

def record_audio(duration=5, fs=44100):
    st.info(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return audio, fs

def save_audio_to_wav(audio, fs):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    scipy.io.wavfile.write(temp_file.name, fs, audio)
    return temp_file.name

def listen():
    audio, fs = record_audio()
    wav_path = save_audio_to_wav(audio, fs)
    st.success("Recording complete. Transcribing...")
    result = model.transcribe(wav_path)
    os.remove(wav_path)
    return result["text"]