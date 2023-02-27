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
   
            file = request.files['image']
    
            # Save the file to disk
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
        
            
            img = tf.keras.utils.load_img(file_path, target_size=(128, 128))
            img = tf.keras.utils.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            model = tf.keras.models.load_model("models\pneumonia.h5")
            prediction = model.predict(img)
            predicted_label = np.argmax(prediction)
            class_names={0: 'NORMAL', 1: 'PNEUMONIA'}
            predicted_class = class_names[predicted_label]
            rendered_page= render_template('predict.html', pred=predicted_label) 
            return rendered_page

if __name__ == '__main__':
    app.run(debug = True)
