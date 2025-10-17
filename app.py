import numpy as np
import re
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# --- Flask Configuration ---
# Tell Flask where our HTML and static asset folders are
app = Flask(__name__,
            template_folder='HTML',
            static_folder='static')

# --- Load Model & Tokenizer ---
try:
    model = load_model('emotion_model.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("✅ Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading assets: {e}")
    model, tokenizer = None, None

# --- Constants & Preprocessing ---
MAX_LENGTH = 100
emotion_labels = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}
emotion_videos = {
    'Sadness': 'sadness.mp4', 'Joy': 'joy.mp4', 'Love': 'love.mp4',
    'Anger': 'anger.mp4', 'Fear': 'fear.mp4', 'Surprise': 'surprise.mp4'
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# --- Page Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

# --- Prediction Logic ---

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not tokenizer:
        return render_template('index.html', error="Model not loaded.")

    user_text = request.form['text']
    if not user_text.strip():
        return render_template('index.html', error="Please enter some text.")

    cleaned_text = clean_text(user_text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')
    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_emotion = emotion_labels.get(predicted_class, "Unknown")
    predicted_video = emotion_videos.get(predicted_emotion, None)

    return render_template('index.html',
                           text=user_text,
                           prediction=predicted_emotion,
                           video_file=predicted_video)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)