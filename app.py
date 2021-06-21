from tensorflow.keras.preprocessing import image

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from Base64 import Base64Utility
# TensorFlow and tf.keras
from tensorflow.keras.preprocessing import image
# Some utilites
import numpy as np
import os
import sys
import argparse
from Classifier import Classifier
# Declare a flask app
app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from ResNet50 import ResNet50
from InceptionResNetV2 import InceptionResNetV2
from EfficientNet import EfficientNet

CONST_LABELS_DICT = {
    'action': 0, 'adventure': 1, 'advertising': 2, 'background': 3, 'ballad': 4, 'calm': 5, 'children': 6,
    'christmas': 7, 'commercial': 8, 'cool': 9, 'corporate': 10, 'dark': 11, 'deep': 12,
    'documentary': 13, 'drama': 14, 'dramatic': 15, 'dream': 16, 'emotional': 17, 'energetic': 18,
    'epic': 19, 'fast': 20, 'film': 21, 'fun': 22, 'funny': 23, 'game': 24, 'groovy': 25, 'happy': 26,
    'heavy': 27, 'holiday': 28, 'hopeful': 29, 'inspiring': 30, 'love': 31, 'meditative': 32,
    'melancholic': 33, 'melodic': 34, 'motivational': 35, 'movie': 36, 'nature': 37, 'party': 38,
    'positive': 39, 'powerful': 40, 'relaxing': 41, 'retro': 42, 'romantic': 43, 'sad': 44, 'sexy': 45,
    'slow': 46, 'soft': 47, 'soundscape': 48, 'space': 49, 'sport': 50, 'summer': 51, 'trailer': 52,
    'travel': 53, 'upbeat': 54, 'uplifting': 55
}

class Model(Classifier):

    def __init__(self, model_name, train_bool):
        __models_dict = {
                'ResNet50': ResNet50(),
                'EfficientNet': EfficientNet(),
                'InceptionResNetV2': InceptionResNetV2()
        }
        self._train_bool = train_bool
        if model_name in list(__models_dict.keys()):
            self.model = __models_dict.get(model_name)

            if self._train_bool:
                self.model.create_model()
                self.model = self.model.train()
        else:
            print('invalid model name was declared')

        # Load your own trained model
        if self._train_bool is False:
            self.MODEL_PATH = os.path.join(os.getcwd(), 'models', model_name)
            self.model = self.model.load_model(self.MODEL_PATH)
            print(self.MODEL_PATH)
        print('Model loaded. Start serving... Check http://127.0.0.1:5003/')

    def predict(self, img):
        img = img.resize((7409, 96))
        image_array = np.array(img)[:, :, 0:3]
        x = image.img_to_array(image_array)
        x = np.expand_dims(x, axis=0)
        return self.model.predict(x)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        base = Base64Utility()
        # Get the image from post request
        img = base.base64_to_pil(request.json)
        # Save the image to ./uploads
        # img.save("./uploads/image.png")
        # Make prediction
        preds = model.predict(img)
        print(preds)
        print("8888 ",preds.argsort())
        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))  # Max probability
        predicted_class_indices = np.argmax(preds, axis=1)
        print('predicted_class_indices', predicted_class_indices)
        print("sd", preds.argsort()[0])

        labels = dict((v, k) for k, v in CONST_LABELS_DICT.items())
        predictions = [labels[k] for k in predicted_class_indices]

        for name, num in labels.items():
            if name == predicted_class_indices:
                print(name)
                print(preds[0])
                print(predicted_class_indices, type(predicted_class_indices))
                result = str(f"{num}={preds[0][name] * 100:.1f}%")

        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=result)

    return None


if __name__ == '__main__':
    # Serve the app with gevent
    __model_names = ['ResNet50', 'InceptionResNetV2', 'EfficientNet']
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="please choose the index of the model which you require to use ['ResNet50','InceptionResNetV2', 'EfficientNet']. The models ordered from the best to the worst.")
    args = parser.parse_args()
    if len(sys.argv) < 2:
        print('You have specified a less number of arguments')
        sys.exit()
    if len(sys.argv) > 3:
        print('You have specified an extra number of arguments')
        sys.exit()
    args.model = int(args.model)
    if args.model > 2 or args.model < 0:
        print("You choose incorrect index for required model. Please try again and choose the index from 0 to 2 included")
        sys.exit()
    model = Model(__model_names[args.model], False)

    http_server = WSGIServer(('0.0.0.0', 5003), app)
    http_server.serve_forever()