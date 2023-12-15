from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model('Models/age_gender_model.h5')

# Mapping of labels for gender
gender_dict = {0: 'Male', 1: 'Female'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['image']
        
        # Process the image
        features = []
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = np.array(img)
        features.append(img)
        features = np.array(features).reshape(len(features), 128, 128, 1)
        features = features / 255.0

        # Make prediction
        pred = model.predict(features)
        print(pred)
        pred_gender = gender_dict[round(pred[0][0][0])]
        pred_age = round(pred[1][0][0])

        # Prepare the result
        result = {'gender': pred_gender, 'age': pred_age}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
