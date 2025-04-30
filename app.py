import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
from PIL import Image
from audiorecorder import audiorecorder
from io import BytesIO
from sklearn.preprocessing import StandardScaler
import joblib  # Import joblib to load the scaler

# Load the model once
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model('emotionrecognition.h5')

# Extract MFCC features from audio data
def extract_features(audio_data, sr):
    # Extract MFCC features from audio data
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    
    # Take the mean of the MFCCs for each coefficient across the frames
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # Return the features as a flattened array
    return mfccs_mean.reshape(1, -1)

# Predict emotion from audio

def predict(model, audio_file_path, scaler):
    # Load the audio file
    audio_data, sr = librosa.load(audio_file_path, sr=None)

    # Extract features from the audio file
    features = extract_features(audio_data, sr)

    # Scale the features using the loaded scaler
    features_scaled = scaler.transform(features)

    # ✅ Reshape features to match the model input
    features_scaled = features_scaled.reshape(1, 40, 1)

    # Predict the emotion using the trained model
    emotion = model.predict(features_scaled)

    # Map emotion index to label
    emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    
    predicted_emotion = emotion_labels[np.argmax(emotion)]

    return predicted_emotion
    
def sidebar_ui():
    st.markdown(
        """
        <style>
        [data-baseweb="radio"] {
            margin: auto;
        }

        [data-baseweb="radio"] .st-au{
            display:none;
        }
        [data-baseweb="radio"] > div {
            display: flex;
            flex-direction: row;
            justify-content: center;
            padding: 10px;
            width:200px;
            gap: 8px;
            background: rgba(200,200,200,0.1);
            border-radius: 5px;
            margin-top: 20px;
        }

        [data-baseweb="radio"] label {
            background-color: #f0f2f6;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 500;
            font-family: 'Helvetica', sans-serif;
            cursor: pointer;
            border: 1px solid transparent;
            transition: 0.2s ease-in-out;
        }

        [data-baseweb="radio"] label[data-selected="true"] {
            background-color: rgba(200,200,200,0.5);
            color: white;
            border-radius: 6px;
            border: 1px solid rgba(200,200,200,0.1);
        }

        .sidebar-img img {
            width: 100px;
            display: block;
            margin: auto;
            margin-top: 60px;
            margin-bottom: 15px;
        }

        .sidebar-title {
            font-weight: bold;
            font-family: Helvetica;
            text-align: center;
            font-size: 20px;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.sidebar.markdown(
        """
        <div class="sidebar-img">
            <img src="https://i.imgur.com/8HCrHAQ.png"/>
        </div>
        <div class='sidebar-title'>MoodMatrix: Speech Analysis System</div>
        """,
        unsafe_allow_html=True
    )

    page = st.sidebar.radio("", ["Analyze", "Project Details", "About Us"], index=0)

    return page

#Pages
import os
import tempfile
import speech_recognition as sr
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit as st
from audiorecorder import audiorecorder
import joblib  # Import joblib to load the scaler

# Function to extract text from the recorded audio
def extract_text_from_audio(audio_data):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_data) as source:
        audio_data = recognizer.record(source)  # Record the audio

    try:
        # Use Google's speech recognition to transcribe the audio
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError:
        return "Speech recognition service is unavailable"

# Function to generate a word cloud image with smaller size
def generate_word_cloud(text):
    wordcloud = WordCloud(width=600, height=300, background_color="white", max_words=150).generate(text)
    return wordcloud

# Load the model and scaler
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model('emotionrecognition.h5')

@st.cache_resource(show_spinner=False)
def load_scaler():
    return joblib.load('scaler.pkl')  # Load the scaler here

def analyze_page():
    model = load_model()
    scaler = load_scaler()  # Load the scaler here
    
    st.subheader("🎤 Analyze your speech for the most comprehensive emotion, sentiment and thematic analysis.")

    emoji_map = {
        'neutral': '😐', 'calm': '😌', 'happy': '😄', 'sad': '😢',
        'angry': '😡', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'
    }

    col1, col2 = st.columns(2, border=True)

    wordcloud = None  # Initialize wordcloud variable
    detected_emotion = ""

    # Handle Audio Recording in col1
    with col1:
        st.markdown("#### 🎙️ Record Audio")
        audio = audiorecorder("Click to Record", "Recording...")

        if len(audio) > 0:
            buffer = BytesIO()
            audio.export(buffer, format="wav")
            st.audio(buffer.getvalue(), format="audio/wav")

            # Save the recorded audio
            audio.export("recorded_audio.wav", format="wav")

            # Extract text from recorded audio and generate word cloud
            text = extract_text_from_audio("recorded_audio.wav")
            wordcloud = generate_word_cloud(text)

            # Emotion analysis
            with st.spinner("Analyzing emotion..."):
                emotion = predict(model, "recorded_audio.wav", scaler)  # Pass the scaler here
                detected_emotion = f"**Detected Emotion:** {emoji_map[emotion]} {emotion.capitalize()}"

    # Handle File Upload in col2
    with col2:
        st.markdown("#### 📁 Upload Audio File")
        uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")

            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            # Extract text from uploaded file and generate word cloud
            text = extract_text_from_audio(temp_file_path)
            wordcloud = generate_word_cloud(text)

            # Emotion analysis
            with st.spinner("Analyzing emotion..."):
                emotion = predict(model, temp_file_path, scaler)  # Pass the scaler here
                detected_emotion = f"**Detected Emotion:** {emoji_map[emotion]} {emotion.capitalize()}"

    if detected_emotion:
        st.subheader("Emotion Detection 🎉")
        st.success(detected_emotion)

    if wordcloud:
        st.subheader("📝 Word Cloud from Audio")
        plt.figure(figsize=(6, 3)) 
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)


def project_details_page():
    st.subheader("Robust accuracy, powered by the best in class libraries and latest machine learning model. 💪")
    st.markdown('<a href="https://docs.google.com/document/d/e/2PACX-1vSqZGsDtJFUJPcTCzbpMihUgqeXAeWjtGLsMgRta_tadII2Ez5ZXMYZ6vJVWpzb2K59A4c0P4F9e12p/pub" target="_blank">Click Here to view the Full Report</a>', unsafe_allow_html=True)

    col1, col2 = st.columns([0.6,0.4],border=True)

    with col1:
        st.markdown("<h3>Accuracy: <span style='color:lightgreen;'>86.63%</span></h3>",unsafe_allow_html=True)
        st.caption("as compared to industry standard (~84%)")
    
    with col2:
        st.text("Precision: ~87%")
        st.text("Recall: ~87%")
        st.text("F1 Score: ~87%")
    
    col3 , col4 = st.columns([0.7,0.3],border=True)

    with col3:
        imgwidth = 300
        st.subheader("Performance Metrics")
        st.caption("Training and Validation Accuracy")
        st.image("learningcurve/accuracy.png")
        st.caption("Training and Validation Loss")
        st.image("learningcurve/loss.png")
        st.caption("Confusion Matrix")
        st.image("learningcurve/confusionmatrix.png")
    
    with col4:
        st.markdown("**Model Used**")
        st.markdown("LSTM")
        st.caption("Unlike RNNs, LLMs (like Transformers) process entire input sequences simultaneously, enabling better context understanding, scalability, and performance in complex language tasks.")
        st.divider()
        st.markdown("**Audio Processing**")
        st.markdown("Librosa")
        st.caption("Librosa is widely used to extract features like MFCCs (Mel-frequency cepstral coefficients), which represent the short-term power spectrum of sound. MFCCs capture vocal characteristics that vary with emotions, making them ideal input features for training emotion recognition models.")
        st.divider()
        st.markdown("**Audio Transcription**")
        st.markdown("Recognizer - Google Speech Recognizer")
        st.caption("A recognizer converts spoken audio into text. We use Google Speech Recognizer to transcribe audio, then generate a word cloud from the transcribed text.")
        st.divider()
        st.markdown("**Deployment**")
        st.markdown("Streamlit")
        st.caption("Streamlit is a Python framework for building interactive web apps quickly, often used for data science and machine learning demos with minimal code.")

def about_us_page():
    st.subheader("Creating projects with passion, creativity and business mindset. 🚀")
    st.markdown(
        """<style>
            *{
                padding:0px;     
            }
            .cards{
                margin:5px;
                background-color: #d9d9d9;
                box-shadow: -10px -10px 30px 0px rgba(255,255,255,0.2), 10px 10px 30px 0px rgba(0, 0, 0, 0.5);
                width: 500px;
                height:275px;
                border-radius:20px;
            }
            .wrapper {
                display: flex;
                flex-direction: column;
                align-items: center; /* center rows horizontally */
                gap: 20px; /* space between top and bottom */
                justify-content: center; /* center the whole wrapper vertically */
            }
            .top-row {
                display: flex;
                gap: 20px; /* space between top cards */
                justify-content: center;
            }
            .bottom-row {
                display: flex;
                justify-content: center;
            }
            .card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }

    .profile-img {
        width: 150px;
        height: 150px;
        object-fit: cover;
    }

    .title-box {
        margin-left: 5px;
        flex-grow: 1;
        font-size:20px;
    }

    .card-title {
        margin: 0;
        font-weight: bold;
        color: #000;
    }

    .card-subtitle {
        margin: 0;
        font-size: 16px;
        color: grey;
    }

    .linkedin-icon {
        width: 50px;
        height: 50px;
        border-radius: 10px;
        background-color: rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        text-decoration: none;
        margin-top:20px;
        margin-right:20px;
    }

    .linkedin-icon img {
        width: 25px;
        filter: brightness(0) invert(1);
    }

    .card-body {
        margin-top: 5px;
        margin-left:20px;
        padding-right:30px;
        padding-left:15px;
        font-size: 16px;
        color: #000;
    }
            </style>"""
        ,unsafe_allow_html=True
    )
    st.markdown("""<div class='wrapper'>
                    <div class='top-row'>
                        <div class='cards'>
                            <div class="card-header">
            <div style="display: flex; align-items: center;">
                <img class="profile-img" src="https://i.imgur.com/a03oGBu.png" alt="Profile Photo">
                <div class="title-box">
                    <p class="card-title">Devangi Bedi</p>
                    <p class="card-subtitle">Consultant at Great Learning</p>
                </div>
            </div>
            <a class="linkedin-icon" href="https://www.linkedin.com/in/devangi-bedi" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn">
            </a>
        </div>
        <div class="card-body">
            Passionate about leveraging technology to solve real world problems. She is working as a consultant at Great Learning with prior experience in product management & cloud computing.
        </div>
                        </div>
                        <div class='cards'><div class="card-header">
            <div style="display: flex; align-items: center;">
                <img class="profile-img" src="https://i.imgur.com/5uUvNYt.png" alt="Profile Photo">
                <div class="title-box">
                    <p class="card-title">Akshita Panwar</p>
                    <p class="card-subtitle">Machine Learning Engineer</p>
                </div>
            </div>
            <a class="linkedin-icon" href="https://www.linkedin.com/in/devangi-bedi" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn">
            </a>
        </div>
        <div class="card-body">
            She is driven by the potential of AI and ML to create innovative solutions. As an aspiring engineer, she is dedicated to exploring deep learning technologies to build impactful solutions.
        </div>
                </div>
                    </div>
                    <div class='bottom-row'>
                        <div class='cards'><div class="card-header">
            <div style="display: flex; align-items: center;">
                <img class="profile-img" src="https://i.imgur.com/zCsG4yg.png" alt="Profile Photo">
                <div class="title-box">
                    <p class="card-title">Aarav Bamba</p>
                    <p class="card-subtitle">Product Analyst at XEBO.ai</p>
                </div>
            </div>
            <a class="linkedin-icon" href="https://www.linkedin.com/in/devangi-bedi" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn">
            </a>
        </div>
        <div class="card-body">
            Leveraging Machine Learning and latest AI tools to spearhead AI-based solutions. Proficient with product management, ensuring that all technical solutions provide value to customers.
        </div>
                </div>
                    </div>
                </div>""",unsafe_allow_html=True)

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
