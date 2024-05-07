from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from pydub import AudioSegment
import librosa
import os
import tempfile
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Load the trained model
model = joblib.load('knn_model.pkl')

def convert_oga_to_wav(oga_file_path):
    try:
        # Load the OGA file
        audio = AudioSegment.from_file(oga_file_path, format="oga")
        # Convert to WAV and return new file path using a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            audio.export(tmp_wav.name, format="wav")
            logging.info(f"Converted {oga_file_path} to {tmp_wav.name}")
            return tmp_wav.name
    except Exception as e:
        logging.error(f"Failed to convert audio: {str(e)}")
        return None

def extract_features(audio_file):
    try:
        # Load the audio file
        y, sr = librosa.load(audio_file, sr=None)
        # Extract features, e.g., MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Aggregating multiple statistics
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        logging.error(f"Error in feature extraction: {str(e)}")
        return None

def predict_anomaly(features):
    try:
        # Predict using the loaded KNN model
        prediction = model.predict([features])
        return prediction[0]
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return False

@app.route('/')
def index():
    return render_template('improveUi.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # Get the audio file from the request
        audio_file = request.files['audio_data']
        audio_path = "./temp_audio.oga"
        audio_file.save(audio_path)
        logging.info("Audio file saved")

        # Convert OGA to WAV
        wav_path = convert_oga_to_wav(audio_path)
        if not wav_path:
            return jsonify(result="Error converting audio file.")

        # Extract features from audio file
        feature_vector = extract_features(wav_path)
        if feature_vector is None:
            return jsonify(result="Error extracting features.")

        # Make prediction using the loaded model
        prediction = predict_anomaly(feature_vector)
        result = "Anomaly detected!" if prediction else "No anomaly detected."

        # Return result as JSON
        return jsonify(result=result)
    except Exception as e:
        logging.error(f"Unhandled exception: {str(e)}")
        return jsonify(result=f"Error processing audio: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
