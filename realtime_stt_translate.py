# realtime_stt_translate.py
# Streamlit app: Real-time multilingual speech recognition with on-the-fly English translation
# Tech: streamlit, streamlit-webrtc, faster-whisper (CTranslate2), scipy
# Works locally (no paid APIs). Supports dozens of languages via Whisper. 
# It chunks microphone audio every ~2s and streams translated text to the UI.

import os
import threading
import queue
from collections import deque

import av
import numpy as np
from scipy.signal import resample_poly
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, AudioProcessorBase

# Whisper model
from faster_whisper import WhisperModel

################################################################################
# ----------------------- App Config & Globals --------------------------------#
################################################################################

st.set_page_config(page_title="Real-time Speech → English", page_icon="🎙️", layout="centered")

# WebRTC NAT traversal (public Google STUN)
RTC_CFG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
})

# Thread-safe queues for audio chunks and recognition results
AUDIO_CHUNK_QUEUE: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
RESULT_QUEUE: "queue.Queue[tuple]" = queue.Queue()

# Session state defaults
if "running" not in st.session_state:
    st.session_state.running = False
if "worker" not in st.session_state:
    st.session_state.worker = None
if "transcript" not in st.session_state:
    st.session_state.transcript = []
if "detected_lang" not in st.session_state:
    st.session_state.detected_lang = None
if "detected_lang_prob" not in st.session_state:
    st.session_state.detected_lang_prob = None

################################################################################
# ----------------------- Audio Processor (WebRTC) ----------------------------#
################################################################################

class MicAudioProcessor(AudioProcessorBase):
    """
    Receives raw audio frames from the browser (usually 48 kHz stereo),
    downsamples to 16 kHz mono, and emits ~2s chunks with 0.5s overlap to
    AUDIO_CHUNK_QUEUE for transcription.
    """

    def __init__(self):
        self.input_sample_rate = 48000  # most browsers send 48 kHz
        self.target_sample_rate = 16000
        self.buffer = np.array([], dtype=np.float32)
        self.chunk_sec = 2.0
        self.overlap_sec = 0.5
        self.lock = threading.Lock()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert frame to numpy (shape: channels x samples), average to mono
        pcm = frame.to_ndarray()
        if pcm.ndim == 2:  # (channels, samples)
            pcm = pcm.mean(axis=0)
        else:
            pcm = pcm.astype(np.float32)

        # Normalize if int16
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32) / 32768.0

        # Ensure float32
        pcm = pcm.astype(np.float32)

        # Resample to 16k using polyphase (robust & efficient)
        sr_in = int(frame.sample_rate) if getattr(frame, "sample_rate", None) else self.input_sample_rate
        if sr_in != self.target_sample_rate:
            # Use rational resampling
            pcm = resample_poly(pcm, self.target_sample_rate, sr_in).astype(np.float32)

        with self.lock:
            self.buffer = np.concatenate([self.buffer, pcm])
            needed = int(self.chunk_sec * self.target_sample_rate)
            if self.buffer.size >= needed:
                # Prepare chunk with overlap retention
                keep = int(self.overlap_sec * self.target_sample_rate)
                chunk = self.buffer[:needed].copy()
                # Keep tail for overlap
                self.buffer = self.buffer[needed - keep:]

                # Non-blocking put; drop if queue is full to avoid latency buildup
                try:
                    AUDIO_CHUNK_QUEUE.put_nowait(chunk)
                except queue.Full:
                    pass

        # We don't modify the outgoing audio stream
        return frame

################################################################################
# ----------------------- Whisper Worker Thread -------------------------------#
################################################################################

def start_worker(model_size: str, device: str, compute_type: str):
    """Background worker that consumes audio chunks and pushes transcripts."""
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    lang_emitted = False

    while True:
        audio = AUDIO_CHUNK_QUEUE.get()
        if audio is None:
            break

        # Transcribe-translate to English
        segments, info = model.transcribe(
            audio=audio,
            language=None,          # auto-detect source language
            task="translate",      # always output English
            beam_size=5,
            vad_filter=True,
            no_speech_threshold=0.6,
        )

        # Emit detected language once
        if not lang_emitted and info is not None and getattr(info, "language", None):
            RESULT_QUEUE.put(("lang", info.language, getattr(info, "language_probability", None)))
            lang_emitted = True

        text_out = "".join(seg.text for seg in segments) if segments else ""
        if text_out.strip():
            RESULT_QUEUE.put(("text", text_out.strip()))

    RESULT_QUEUE.put(("done",))

################################################################################
# ----------------------- UI --------------------------------------------------#
################################################################################

st.title("🎙️ Real-time Multilingual Speech → English")
st.caption("Speak in any supported language. We transcribe & translate to English on the fly.")

with st.sidebar:
    st.subheader("Settings")
    model_size = st.selectbox("Whisper model", ["tiny", "base", "small", "medium"], index=2, help="Bigger = better but slower")
    device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0, help="'auto' picks CUDA if available")
    compute_type = st.selectbox(
        "Compute precision",
        ["auto", "int8", "int8_float16", "float16", "float32"],
        index=1,
        help="Lower precision is faster on CPU; float16 on GPU is great if supported",
    )
    st.markdown("---")
    st.write("**Tip:** Start with 'small' on CPU or 'medium' on GPU for good accuracy.")

# WebRTC streamer - receive mic audio only
ctx = webrtc_streamer(
    key="stt-translate",
    mode=WebRtcMode.RECVONLY,
    audio_processor_factory=MicAudioProcessor,
    rtc_configuration=RTC_CFG,
    media_stream_constraints={"audio": True, "video": False},
)

# Controls
col1, col2, col3 = st.columns([1,1,1])
with col1:
    clear_btn = st.button("Clear Transcript")
with col2:
    save_btn = st.button("Save to file")
with col3:
    stop_btn = st.button("Stop")

if clear_btn:
    st.session_state.transcript = []
    st.session_state.detected_lang = None
    st.session_state.detected_lang_prob = None

# Start worker when playing
if ctx.state.playing and not st.session_state.running:
    st.session_state.running = True
    st.session_state.worker = threading.Thread(
        target=start_worker,
        kwargs={
            "model_size": model_size,
            "device": None if device == "auto" else device,
            "compute_type": None if compute_type == "auto" else compute_type,
        },
        daemon=True,
    )
    st.session_state.worker.start()

# Poll results queue each rerun
try:
    while True:
        item = RESULT_QUEUE.get_nowait()
        if item[0] == "lang":
            st.session_state.detected_lang = item[1]
            st.session_state.detected_lang_prob = item[2]
        elif item[0] == "text":
            st.session_state.transcript.append(item[1])
        elif item[0] == "done":
            st.session_state.running = False
        else:
            pass
except queue.Empty:
    pass

# Stop
if stop_btn and st.session_state.running:
    # Signal worker to stop
    try:
        AUDIO_CHUNK_QUEUE.put_nowait(None)
    except queue.Full:
        pass
    st.session_state.running = False

# Display detected language
if st.session_state.detected_lang:
    prob = st.session_state.detected_lang_prob
    if prob is not None:
        st.info(f"Detected language: **{st.session_state.detected_lang}** (p≈{prob:.2f})")
    else:
        st.info(f"Detected language: **{st.session_state.detected_lang}**")

# Live transcript display
st.subheader("Live English Transcript")
if len(st.session_state.transcript) == 0:
    st.write("_Start speaking..._")
else:
    st.write("\n\n".join(st.session_state.transcript))

# Save transcript to a local text file
if save_btn and st.session_state.transcript:
    text = "\n".join(st.session_state.transcript)
    os.makedirs("./exports", exist_ok=True)
    path = "./exports/transcript_en.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    st.success(f"Saved to {path}")

# Footer note
st.caption(
    "Audio is processed on your machine if you run locally. If deployed, audio is sent to the server hosting this app for processing."
)
