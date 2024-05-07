from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import wave

app = Flask(__name__)

# Load the trained model
model = joblib.load('knn_model.pkl')

def process_audio_data(audio_file):
    # Save the audio file as .wav
    with wave.open(audio_file.filename + ".wav", 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # Sample width in bytes
        wav_file.setframerate(44100)  # Sample rate in Hz
        wav_file.writeframes(audio_file.read())

    # Placeholder for actual feature extraction logic from the .wav file
    feature_vector = np.random.randn(1, 20)
    return feature_vector

def predict_anomaly(features):
    prediction = model.predict(features)
    return prediction[0]  # Assuming the model returns an array

@app.route('/')
def index():
    return render_template('improveUi.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    # Get the audio file from the request
    audio_file = request.files['audio_data']
    
    # Process the audio file to generate feature vector
    feature_vector = process_audio_data(audio_file)

    # Make prediction using the loaded model
    prediction = predict_anomaly(feature_vector)

    result = "Anomaly detected!" if prediction else "No anomaly detected."

    # Return result as JSON
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
