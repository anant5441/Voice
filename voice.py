import streamlit as st
import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
from pydub.utils import which
import pyaudio
import whisper
import google.generativeai as genai
from dotenv import load_dotenv
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up Streamlit page
st.set_page_config(page_title="Medical Voice Assistant", page_icon="🩺")
st.title("Medical Voice Assistant")
st.write("Record your symptoms and get medical analysis")

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Check FFmpeg
ffmpeg_path = which("ffmpeg")
if not ffmpeg_path:
    st.error("FFmpeg not found! Please install it and add it to your PATH.")
else:
    AudioSegment.converter = ffmpeg_path
    logging.info(f"FFmpeg found at: {ffmpeg_path}")

def record_audio(file_path, timeout=20, phrase_time_limit=30):
    """Record audio from microphone"""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            st.info("Start speaking now...")
            
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            st.success("Recording complete.")
            
            # Convert and save audio
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            return True
            
    except Exception as e:
        st.error(f"Recording error: {str(e)}")
        return False

def transcribe_with_whisper(audio_filepath):
    """Transcribe audio using Whisper"""
    try:
        with st.spinner("Loading Whisper model..."):
            model = whisper.load_model("medium")
        
        with st.spinner("Transcribing audio..."):
            result = model.transcribe(audio_filepath, task="transcribe")
            st.info(f"Detected Language: {result['language']}")
            return result["text"]
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def analyze_transcript_with_gemini(transcript):
    """Analyze transcript with Gemini"""
    try:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
        genai.configure(api_key=GOOGLE_API_KEY)
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""Analyze this patient transcript:
{transcript}

Provide:
1. Summary
2. Symptoms/concerns
3. Recommended specialist
4. Urgency level
5. Recommendations
6. First-aid advice if urgent"""
        
        with st.spinner("Analyzing with Gemini..."):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None

# Streamlit UI
audio_path = "patient_recording.mp3"

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Recording", disabled=st.session_state.recording):
        st.session_state.recording = True
        st.session_state.processed = False
        if record_audio(audio_path, phrase_time_limit=30):
            st.session_state.recording = False

with col2:
    if st.button("Analyze Recording", disabled=st.session_state.recording or st.session_state.processed):
        with st.spinner("Processing..."):
            transcript = transcribe_with_whisper(audio_path)
            if transcript:
                st.subheader("Transcript")
                st.write(transcript)
                
                analysis = analyze_transcript_with_gemini(transcript)
                if analysis:
                    st.subheader("Medical Analysis")
                    st.write(analysis)
                    st.session_state.processed = True

# Instructions
st.sidebar.markdown("""
## Instructions
1. Click **Start Recording** and speak about your symptoms
2. Click **Analyze Recording** when finished
3. View your transcript and medical analysis

Note: Recording will automatically stop after 30 seconds of silence or when you stop speaking.
""")

# Add some styling
st.markdown("""
<style>
    .stButton button {
        width: 100%;
        padding: 10px;
        font-weight: bold;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)