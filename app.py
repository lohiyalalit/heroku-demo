
# coding=utf-8
import os
import numpy as np
from PIL import Image,ImageOps

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template

# Define a flask app
app = Flask(__name__,template_folder='templates')

# Model saved with Keras model.save()
MODEL_PATH = 'keras_model.h5'
upload_folder = 'static/pics'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()

def predict(image_location,model):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_location)
    size = (224,224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    # run the inference
    prediction = model.predict(data)
    prediction = np.argmax(prediction)
    if prediction == 0:
        result = "Cat"
    else:
        result = "Dog"
    return result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
        if request.method == "POST":
            image_file = request.files['file']
            if image_file:
                image_location = os.path.join(upload_folder, image_file.filename)
                image_file.save(image_location)
                preds = predict(image_location,model)
                return preds
        return None


if __name__ == '__main__':
    app.run(debug=True)

