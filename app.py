import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
from PIL import Image
import wave
from audiorecorder import audiorecorder
from io import BytesIO

# Load the model once
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model('emotionrecognition.h5')

# Extract MFCC features
def extract_mfcc(wav_file, target_duration=3):
    y, sr = librosa.load(wav_file)
    
    # Ensure the audio is 3 seconds long
    target_samples = target_duration * sr
    
    if len(y) > target_samples:
        y = y[:target_samples]  # Truncate if audio is longer than 3 seconds
    elif len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))  # Pad if audio is shorter than 3 seconds
    
    # Extract MFCC features
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

# Predict emotion from audio
def predict(model, wav_file):
    emotions = {
        0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
        4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
    }

    mfcc = extract_mfcc(wav_file)
    mfcc = np.reshape(mfcc, (1, 40, 1))
    predictions = model.predict(mfcc)

    predicted_index = np.argmax(predictions[0])
    predicted_emotion = emotions[predicted_index]
    return predicted_emotion

# Sidebar Styling & Navigation
def sidebar_ui():
    st.sidebar.markdown(
        """
        <style>
        .sidebar-content { display: flex; flex-direction: column; align-items: center; }
        .element-container:nth-child(1) img { display: block; margin-left: auto; margin-right: auto; }
        .block-container { padding-top: 2rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.image("logo.png", width=130)
    st.sidebar.markdown("<h3 style='text-align: center;'>MoodMatrix: Speech<br>Analysis System</h3>", unsafe_allow_html=True)
    page = st.sidebar.radio("", ["Analyze", "Project Details", "About Us"], index=0)
    st.sidebar.markdown("---")

    # Footer Icons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.markdown("[![GitHub](https://img.icons8.com/ios-filled/25/ffffff/github.png)](https://github.com)", unsafe_allow_html=True)
    with col2:
        st.markdown("[![External](https://img.icons8.com/ios-glyphs/25/ffffff/external-link.png)](https://yourprojectlink.com)", unsafe_allow_html=True)

    return page

# Pages
# Modify the analyze_page() function to handle file upload properly
def analyze_page():
    model = load_model()
    st.subheader("🎤 Analyze your speech for the most comprehensive emotion, sentiment and thematic analysis.")

    emoji_map = {
        'neutral': '😐', 'calm': '😌', 'happy': '😄', 'sad': '😢',
        'angry': '😡', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'
    }

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎙️ Record Audio")
        audio = audiorecorder("Click to Record", "Recording...")

        if len(audio) > 0:
            buffer = BytesIO()
            audio.export(buffer, format="wav")
            st.audio(buffer.getvalue(), format="audio/wav")

            audio.export("recorded_audio.wav", format="wav")

            with st.spinner("Analyzing emotion..."):
                emotion = predict(model, "recorded_audio.wav")
                st.success(f"**Detected Emotion:** {emoji_map[emotion]} {emotion.capitalize()}")

    with col2:
        st.markdown("#### 📁 Upload Audio File")
        uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")

            with st.spinner("Analyzing emotion..."):
                emotion = predict(model, uploaded_file)
                st.success(f"**Detected Emotion:** {emoji_map[emotion]} {emotion.capitalize()}")

def project_details_page():
    st.subheader("Project Details")
    st.markdown("""
    This project uses deep learning (LSTM) to classify emotions from speech based on MFCC audio features.

    **Technologies:**
    - TensorFlow/Keras
    - Librosa for audio processing
    - Streamlit for UI

    **Model Input:**
    - `.wav` files
    - Extracted MFCCs (40 coefficients)

    **Emotions Covered:**
    Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
    """)

def about_us_page():
    st.subheader("About Us")
    st.markdown("""
    We are a team of passionate engineers building AI tools for real-world applications.

    **Contact:** moodmatrix@example.com  
    **GitHub:** [github.com/moodmatrix](https://github.com)  
    """)

# Main App
def main():
    st.set_page_config(page_title="MoodMatrix", page_icon="🎙️", layout="centered")
    
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 4rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    page = sidebar_ui()

    if page == "Analyze":
        analyze_page()
    elif page == "Project Details":
        project_details_page()
    elif page == "About Us":
        about_us_page()

if __name__ == "__main__":
     main()
