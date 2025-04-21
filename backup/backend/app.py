from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
model = load_model('plant_disease_model.h5')

# Class names
CLASS_NAMES = ['Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust']

# Home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')


# Image upload and prediction
@app.route('/detection/upload/', methods=['POST'])
def predict():
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = image.reshape(1, 256, 256, 3)

    prediction = model.predict(image)
    result = CLASS_NAMES[np.argmax(prediction)]
    return jsonify({'result': f"{result.split('-')[0]} leaf with {result.split('-')[1]}"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
