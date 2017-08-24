import requests
import json
import random
import os
import sys
import pickle
import io
import cv2

import numpy as np
import pandas as pd
import tensorflow as tf

from flask import Flask, render_template, request, redirect, send_from_directory, jsonify, Response, g, make_response
from flask_restful import Resource, Api
import pymongo
from datetime import datetime
from functools import wraps, update_wrapper

from os import listdir
from os.path import isfile, join
from werkzeug import secure_filename

from keras.models import load_model


sys.path.insert(0, '../src')
from helpers import *

uri = os.environ['MONGODB_URI']

app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'PNG', 'JPG'])
app.config['MAX_CONTENT_PATH'] = 4000000

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3

# helper function
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# index page
@app.route('/')
def index():
    # global logged_In
    # if logged_In != True:
    #     return redirect("/login")
    return render_template('index.html')

#results page
@app.route('/results', methods = ["GET", "POST"])
def results():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            # flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            cropped, score, class_ = crop_image('../{}'.format(file_path), detection_graph)

            if class_ != 16.:
                return render_template('tryagain.html')
            if score < 0.5:
                return render_template('tryagain.html')
            if cropped is None:
                return render_template('tryagain.html')

            cv2.imwrite('static/img/cropped/{}'.format(filename), cropped)
            upload_cropped = 'static/img/cropped/{}'.format(filename)

            processed = preprocess_image(cropped)

            pred = model.predict(processed).flatten()
            top_keys, top_perc = return_top_n(pred)

            if top_perc[0] <= 0.2:
                return render_template('speciesnotfound.html')

            cursor1 = web_data.find_one({'id':int(top_keys[0])})
            cursor2 = web_data.find_one({'id':int(top_keys[1])})
            cursor3 = web_data.find_one({'id':int(top_keys[2])})

            mask = random.sample(range(1,9), 4)

            lst1 = cursor1['paths'].split(',')
            lst2 = cursor2['paths'].split(',')
            lst3 = cursor3['paths'].split(',')

            images1 = {'b1p0': img_root+lst1[mask[0]][2:-1], 'b1p1':img_root+lst1[mask[1]][2:-1],'b1p2':img_root+lst1[mask[2]][2:-1],'b1p3':img_root+lst1[mask[3]][2:-1]}

            images2 = {'b2p0': img_root+lst2[mask[0]][2:-1], 'b2p1':img_root+lst2[mask[1]][2:-1],'b2p2':img_root+lst2[mask[2]][2:-1],'b2p3':img_root+lst2[mask[3]][2:-1]}

            images3 = {'b3p0': img_root+lst3[mask[0]][2:-1], 'b3p1':img_root+lst3[mask[1]][2:-1],'b3p2':img_root+lst3[mask[2]][2:-1],'b3p3':img_root+lst3[mask[3]][2:-1]}

            def conf_fxn(score):
                if score <= 0.02:
                    return 'Shoot, this definitely is not your bird'
                elif score <= 0.3:
                    return "This could be your bird but it's not likely"
                elif score <= 0.5:
                    return 'We think this might be your bird'
                elif score <= 0.8:
                    return "It's very likely this is your bird"
                elif score <= 0.97:
                    return "We're pretty confident this is your bird"
                else:
                    return "This is definitely your bird"

            likelihoods = {'conf1': conf_fxn(top_perc[0]), 'perc1': round(top_perc[0]*100, 2), 'conf2': conf_fxn(top_perc[1]), 'perc2': round(top_perc[1]*100, 2), 'conf3': conf_fxn(top_perc[2]), 'perc3': round(top_perc[2]*100, 2),}

            return render_template('results.html', bird1 = cursor1, bird2 = cursor2, bird3 = cursor3, pics1 = images1, pics2 = images2, pics3 = images3, how_conf = likelihoods, preview_img = upload_cropped)

# try again page
@app.route('/tryagain')
def tryagain():
    return render_template('tryagain.html')

# not found page
@app.route('/speciesnotfound')
def speciesnotfound():
    return render_template('speciesnotfound.html')



if __name__ == '__main__':
    img_root = 'https://s3.amazonaws.com/lazybirder/s3/'
    client = pymongo.MongoClient(uri)
    db = client.get_default_database()
    web_data = db['web_data']

    b_dict = load_dict()
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = '/usr/local/lib/python3.5/dist-packages/tensorflow/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'

    print('\n##### Loading detection_graph into memory #####')
    # Load a (frozen) Tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    print('\n##### Loading frozen model into memory #####')
    # Load a (frozen) Tensorflow model into memory
    model = load_model('../data/vgg16-top200-single-SGD.h5')

    app.run(host='0.0.0.0', port=80, threaded=True)
