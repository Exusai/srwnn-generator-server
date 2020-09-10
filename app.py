import flask
from flask import Flask, redirect, url_for, request, render_template
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import base64
import io 

MODELS_PATH = './models/'
BASE_MODEL = 'SRWNNbase.h5'
srwnnModelPaht = MODELS_PATH +  BASE_MODEL

denoise1ModelPaht = MODELS_PATH +  'SRWNNdeNoise1.h5'
denoise2ModelPaht = MODELS_PATH +  'SRWNNdeNoise2.h5'
denoise3ModelPaht = MODELS_PATH +  'SRWNNdeNoise3.h5'
deblur1ModelPaht = MODELS_PATH +  'SRWNNdeBlur1.h5'
deblur2ModelPaht = MODELS_PATH +  'SRWNNdeBlur1.h5'
deblur3ModelPaht = MODELS_PATH +  'SRWNNdeBlur1.h5'

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
    if modelConfig == '0000' return srwnnModelPaht
    if modelConfig == '0100' return denoise1ModelPaht #change to actual model for images
    if modelConfig == '0010' return denoise1ModelPaht
    if modelConfig == '0020' return denoise2ModelPaht
    if modelConfig == '0030' return denoise3ModelPaht
    if modelConfig == '0001' return deblur1ModelPaht
    if modelConfig == '0002' return deblur2ModelPaht
    if modelConfig == '0003' return deblur3ModelPaht
    else return srwnnModelPaht 

@app.route('/')
def index():
    return "Super.Resolution.Waifu.Neural.Network"

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
