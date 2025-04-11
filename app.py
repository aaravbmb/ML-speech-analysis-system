import streamlit as st
import numpy as np
import tensorflow as tf
import urllib.request
import soundfile as sf
import librosa  # to extract speech features

# Sidebar selection
def main():
    selected_box = st.sidebar.selectbox(
        'Choose an option..',
        ('Emotion Recognition', 'View Source Code')
    )

    if selected_box == 'Emotion Recognition':
        st.sidebar.success('Try uploading an audio file to test.')
        application()

    if selected_box == 'View Source Code':
        st.code(get_file_content_as_string("app.py"))

# Cached GitHub source code loader
@st.cache_data(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/chiluveri-sanjay/Emotion-recognition/main/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

# Cached model loader
@st.cache_resource(show_spinner=False)
def load_model():
    model = tf.keras.models.load_model('emotionrecognition.h5')
    return model

# Main application logic
def application():
    with st.spinner('Loading model...'):
        model = load_model()
    st.success('Model loaded successfully!')

    file_to_be_uploaded = st.file_uploader("Choose a .wav audio file", type="wav")

    if file_to_be_uploaded:
        st.audio(file_to_be_uploaded, format='audio/wav')
        with st.spinner('Analyzing emotion...'):
            emotion = predict(model, file_to_be_uploaded)
        st.success(f'Emotion of the audio is **{emotion}** 🎧')

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

if __name__ == "__main__":
    main()
