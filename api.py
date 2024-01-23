import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)


model = tf.keras.models.load_model('Fire_detection.h5')


def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return image

@app.route('/detect_fire', methods=['POST'])
def detect_fire():
    # Get the uploaded image 
    image_file = request.files['image']

   
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)

  
    predictions = model.predict(processed_image)
    predicted_label = np.argmax(predictions[0])

    if predicted_label == 0:
        return jsonify({'fire_detected': True})
    else:
        return jsonify({'fire_detected': False})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)
