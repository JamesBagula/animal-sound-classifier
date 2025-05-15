import streamlit as st
import pandas as pd
import numpy as np
import librosa
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from PIL import Image
import tempfile

# Must be first Streamlit command
st.set_page_config(page_title="Animal Sound Classifier", layout="centered")

# Load dataset
df = pd.read_csv("dataset.csv")

# Extract features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration > 11:
            st.warning("Audio must be 11 seconds or less.")
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Prepare model
X, y = [], []
for _, row in df.iterrows():
    path = row["file_path"].strip()
    if os.path.exists(path):
        features = extract_features(path)
        if features is not None:
            X.append(features)
            y.append(row["label"])
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
accuracy = accuracy_score(y_test, clf.predict(X_test))

# UI
st.title("Animal Sound Classifier")
st.markdown(f"**Supported animals**: {', '.join(sorted(set(y)))}")

# Section 1: Download sample animal sounds
st.subheader("Download Sample Animal Sounds to Test")
sample_animals = ['cat', 'dog', 'cow', 'elephant', 'tiger']
for animal in sample_animals:
    path = f"sample_sounds/{animal}.wav"
    if os.path.exists(path):
        with open(path, "rb") as file:
            st.download_button(f"Download {animal.capitalize()} Sound", file.read(), file_name=f"{animal}.wav")

# Section 2: Upload your sound
st.subheader("Upload Your Animal Sound (.wav)")
uploaded_file = st.file_uploader("Upload a .wav sound file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    features = extract_features(tmp_path)
    if features is not None:
        probabilities = clf.predict_proba([features])[0]
        prediction = clf.predict([features])[0]
        max_prob = np.max(probabilities)

        if max_prob < 0.6:
            st.error(f"Prediction: UNKNOWN | Confidence: {max_prob:.2f}")
            st.text("No match found.")
        else:
            st.success(f"Prediction: {prediction.upper()} | Model Accuracy: {accuracy * 100:.2f}%")

            image_path = f"images/{prediction.lower()}.jpg"
            if os.path.exists(image_path):
                st.image(Image.open(image_path).resize((200, 200)), caption=prediction.capitalize())
            else:
                st.text("Image not found.")

            # Show bar chart
            st.subheader("Prediction Probabilities")
            fig, ax = plt.subplots()
            colors = {
                'cat': 'green', 'dog': 'orange', 'cow': 'brown',
                'elephant': 'gray', 'tiger': 'red'
            }
            color_list = [colors.get(label.lower(), 'skyblue') for label in clf.classes_]
            ax.bar(clf.classes_, probabilities, color=color_list)
            ax.set_ylim([0, 1])
            ax.set_ylabel("Confidence")
            st.pyplot(fig)

            # Closest match
            st.subheader("Play Closest Match from Dataset")
            distances = cdist([features], X, metric='euclidean')
            closest_index = np.argmin(distances)
            match_path = df.iloc[closest_index]["file_path"].strip()
            if os.path.exists(match_path):
                with open(match_path, "rb") as f:
                    st.audio(f.read(), format="audio/wav")
            else:
                st.warning("Closest match file not found.")