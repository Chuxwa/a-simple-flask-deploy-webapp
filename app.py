import os
import sys
import base64

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil, np_to_base64

# our model
from web_models.web import init, predict_model
import torch

# Declare a flask app
app = Flask(__name__)

device = '9'
# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# model = MobileNetV2(weights='imagenet')

# print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
# MODEL_PATH = 'models/your_model.h5'
# model = load_model(MODEL_PATH)

# Load your own trained model
model = init(device = device)
# model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img, model):
    img = img.resize((1024, 1024))
    img = image.img_to_array(img)
    img = np.transpose(img,(2,0,1))

    # Preprocessing the image
    im0 = predict_model(model,img,device = device)
    # im0 = np.transpose(im0,(1,2,0))

    # preds = predict(model,x)
    return im0

@app.route('/result.jpg', methods=['GET'])
def result():
    img = open('result.jpg','rb')
    result = img.read()
    return Response(result, mimetype="image/jpeg")


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        im0 = model_predict(img, model)
        result = np_to_base64(im0)
        # im0 = image.array_to_img(im0)
        # image.save_img("result.jpg",im0)
        # Serialize the result, you can add additional fields
        return jsonify(result=result)
        
        # return render_template("index.html", image=result)
        # return None

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5001), app)
    http_server.serve_forever()
