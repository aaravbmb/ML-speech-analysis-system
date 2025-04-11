import streamlit as st
import numpy as np    
import tensorflow as tf
import librosa
from PIL import Image

# Load the model once
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model('emotionrecognition.h5')

# Extract MFCC features
def extract_mfcc(wav_file):
    y, sr = librosa.load(wav_file)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

# Predict emotion from audio
def predict(model, wav_file):
    emotions = {
        1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
        5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
    }
    mfcc = extract_mfcc(wav_file)
    mfcc = np.reshape(mfcc, (1, 40, 1))
    predictions = model.predict(mfcc)
    return emotions[np.argmax(predictions[0]) + 1]

# Sidebar Styling & Navigation
def sidebar_ui():
    st.sidebar.markdown(
        "<style>"
        ".sidebar-content { display: flex; flex-direction: column; align-items: center; }"
        "</style>",
        unsafe_allow_html=True
    )

    # Sidebar Image
    st.sidebar.image("logo.png", width=130)

    # Title
    st.sidebar.markdown("<h3 style='text-align: center;'>MoodMatrix: Speech<br>Analysis System</h3>", unsafe_allow_html=True)

    # Navigation Tabs
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
def analyze_page():
    model = load_model()
    st.subheader("Upload an Audio File")
    uploaded_file = st.file_uploader("Choose a WAV file", type="wav")
    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        emotion = predict(model, uploaded_file)
        st.success(f"Detected Emotion: **{emotion}**")

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
    page = sidebar_ui()

    if page == "Analyze":
        analyze_page()
    elif page == "Project Details":
        project_details_page()
    elif page == "About Us":
        about_us_page()

if __name__ == "__main__":
    main()
