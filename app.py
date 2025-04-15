import streamlit as st
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load Models & Scalers
emotion_model = joblib.load("cnn_emotion_model.pkl")
emotion_scaler = joblib.load("cnn_emotion_scaler.pkl")
gender_model = joblib.load("cnn_gender_model.pkl")
gender_scaler = joblib.load("cnn_gender_scaler.pkl")

emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
gender_labels = ['female', 'male']

def extract_mfcc(file, max_pad_len=174):
    audio, sr = librosa.load(file, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc.flatten().reshape(1, -1)

st.title("🎙️ Speech Emotion & Gender Recognition")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    features = extract_mfcc(uploaded_file)
    
    # Emotion prediction
    emotion_input = emotion_scaler.transform(features)
    emotion_pred = emotion_model.predict(emotion_input)
    emotion_result = emotion_labels[np.argmax(emotion_pred)]

    # Gender prediction
    gender_input = gender_scaler.transform(features)
    gender_pred = gender_model.predict(gender_input)
    gender_result = gender_labels[np.argmax(gender_pred)]

    st.subheader("🧠 Predictions")
    st.write(f"**Emotion:** {emotion_result}")
    st.write(f"**Gender:** {gender_result}")
