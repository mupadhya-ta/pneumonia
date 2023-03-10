import os
from flask_cors import CORS
from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
import tflite as tf

app = Flask(__name__)
CORS(app)
@app.route("/pneumoniahome")
def home():
    return render_template('home.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')


@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    pred=None
    if request.method == 'POST':
        try:
            img = Image.open(request.files['image']).convert('L')
            img.save("uploads/image.jpg")
            img_path = os.path.join(os.path.dirname(__file__), 'uploads/image.jpg')
            os.path.isfile(img_path)
            img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
            img = tf.keras.utils.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            model = tf.keras.models.load_model("models/pneumonia.h5")
            pred = np.argmax(model.predict(img))
            rendered_page= render_template('predict.html', pred=pred) 
        except:
            message = "Please upload an image"
            
            rendered_page= render_template('pneumonia_predict.html', message=message)

    return rendered_page;

if __name__ == '__main__':
    app.run(debug = True)
