import flask
from flask import Flask, redirect, url_for, request, render_template
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import base64
import io 

MODELS_PATH = './models/'
OUTPUT_PATH = './output/'

BASE_MODEL = 'SRWNNbase.h5'

srwnnModelPaht = MODELS_PATH +  BASE_MODEL

app = Flask(__name__)

def generate(imageInput, modelPath):
    generator = tf.keras.models.load_model(modelPath)

    imageInput = Image.open(imageInput.stream)
    arrayInput = np.array(imageInput)

    input = tf.cast(arrayInput, tf.float32)[...,:3]
    input = (input/127.5) - 1
    image = tf.expand_dims(input, axis = 0)
    
    genOutput = generator(image, training =  False) 

    return genOutput[0, ...]

def getModelPath(modelConfig):
    if modelConfig == '0000':
        print('model path: ', srwnnModelPaht)
        return srwnnModelPaht
    if modelConfig != '0000': 
        return srwnnModelPaht 

@app.route('/')
def index():
    return "Index Page"

@app.route('/generate', methods=['POST'])
def gen():
    data = {"success": False}
    if request.files.get("image"):
        image = request.files['image']
        
        payload = request.form.to_dict()
        modelConfig = payload['model']
        print('model config: ', modelConfig)

        modelPathStr = getModelPath(modelConfig)

        generatedImageArray = generate(image, modelPathStr)
        generatedImage = Image.fromarray(np.uint8(((generatedImageArray+1)/2)*255), 'RGB')
        buffer = io.BytesIO()
        generatedImage.save(buffer,format="png")
        imageBuffer = buffer.getvalue()                     
        
        encodedImage = base64.b64encode(imageBuffer)
        img_str = encodedImage

        data["success"] = True

    return flask.jsonify({'msg': data, 'img': str(img_str) })
    


if __name__ == '__main__':
	app.run(host='0.0.0.0',debug=True)
