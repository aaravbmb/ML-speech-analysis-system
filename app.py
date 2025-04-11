import streamlit as st
import numpy as np
import tensorflow as tf
import urllib.request
import soundfile as sf
import librosa

# Cached model loader
@st.cache_resource(show_spinner=False)
def load_model():
    model = tf.keras.models.load_model('emotionrecognition.h5')
    return model

# Extract MFCC features from audio file
def extract_mfcc(wav_file):
    y, sr = sf.read(wav_file)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

# Predict emotion from audio
def predict(model, wav_file):
    emotions = {
        1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
        5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
    }
    test_point = extract_mfcc(wav_file)
    test_point = np.reshape(test_point, newshape=(1, 40, 1))
    predictions = model.predict(test_point)
    predicted_emotion = emotions[np.argmax(predictions[0]) + 1]
    return predicted_emotion

# Emotion Recognition App
def emotion_recognition_app():
    st.header("🎙️ Speech Emotion Recognition")
    with st.spinner('Loading model...'):
        model = load_model()
    st.success('Model loaded successfully!')

    file_to_be_uploaded = st.file_uploader("Choose a .wav audio file", type="wav")

    if file_to_be_uploaded:
        st.audio(file_to_be_uploaded, format='audio/wav')
        with st.spinner('Analyzing emotion...'):
            emotion = predict(model, file_to_be_uploaded)
        st.success(f'Emotion of the audio is **{emotion}** 🎧')

# About Page
def about_page():
    st.header("📚 About the Project")
    st.markdown("""
    This project is a **Speech Emotion Recognition System** built using deep learning.  
    It uses **MFCC (Mel-frequency cepstral coefficients)** features extracted from `.wav` audio files  
    and classifies emotions like *happy, sad, angry, neutral*, etc.  
      
    **Technologies used**:
    - TensorFlow / Keras for the LSTM model
    - Librosa for feature extraction
    - Streamlit for web deployment
    """)

# Team Page
def team_page():
    st.header("👨‍💻 Meet the Team")
    st.markdown("""
    - **Aarav Bamba** — Product Analyst & Developer  
    - **OpenAI GPT-4** — Assistant and Copilot  
    - Special thanks to [Chiluveri Sanjay](https://github.com/chiluveri-sanjay) for the base repo.  
    """)

# Main App Navigation
def main():
    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.radio("Go to", ["Emotion Recognition", "About the Project", "Meet the Team"])

    if page == "Emotion Recognition":
        emotion_recognition_app()
    elif page == "About the Project":
        about_page()
    elif page == "Meet the Team":
        team_page()

if __name__ == "__main__":
    main()
