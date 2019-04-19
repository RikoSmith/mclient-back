import os.path
import os
import threading
import datetime
from datetime import timedelta
import base64
import json
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras import regularizers
import os
import pandas as pd
from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from functools import wraps
from keras.layers import Input, Flatten, Dropout  # , Activation
from flask_cors import CORS, cross_origin
from random import randint


# from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Activation, Dense,  Embedding


app = Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_ECHO"] = False
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True
app.config["SQLALCHEMY_DATABASE_URI"] = "postgres://clrjmonqrginde:284171df33b7d58673e9b9f3b572523ce59e2560dcf462efdb8f3052bc2f07c5@ec2-54-246-86-167.eu-west-1.compute.amazonaws.com:5432/damcpa0auiprid"
app.config["SECRET_KEY"] = "this is my secret key 1245215"

db = SQLAlchemy(app)


# ------------------DATABASE --------------------------------------


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(150), unique=True)
    uname = db.Column(db.String(50), unique=True)
    name = db.Column(db.String(50))
    lname = db.Column(db.String(50))
    quote = db.Column(db.String(300))
    sex = db.Column(db.String(1))
    country = db.Column(db.String(150))
    password = db.Column(db.String(80))
    weight = db.Column(db.Integer())


class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(200))
    complete = db.Column(db.Boolean)
    user_id = db.Column(db.Integer)


class Slinks(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    link = db.Column(db.String(300))
    site = db.Column(db.String(100))


class Fdata(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(150))
    weight = db.Column(db.Integer)
    mood = db.Column(db.String(200))
    hbeat = db.Column(db.Integer)
    todos = db.Column(db.Integer)
    date = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    feedback = db.Column(db.Boolean, default=True)
    mood_text = db.Column(db.String(200))
    features = db.Column(db.Text())


class Stats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    data_bar = db.Column(db.ARRAY(db.Float))

# ---------------------- HELPER FUNCTIONS ----------------------------


def token_checker(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
            print(request.headers['x-access-token'])

        if not token:
            print(request.headers)
            return jsonify({"ok": False, "message": "Token is missing!"})

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'])
            current_user = User.query.filter_by(
                user_id=data['user_id']).first()
        except:
            return jsonify({"ok": False, "message": "Token is invalid"})

        return f(current_user, *args, **kwargs)

    return decorated


def getPrediction(eng_stress_model, livecnn):
    #predict!
    livepreds = eng_stress_model.predict(livecnn, 
                            batch_size=32, 
                            verbose=1)

    #convert prediction to text
    livepreds1 = livepreds.argmax(axis=1)
    liveabc = livepreds1.astype(int).flatten()
    conv = ['female_not_stressed', 'female_stressed', 'male_not_stressed',
        'male_stressed']

    #array for conversion into binary class (stressed/not_stressed)
    conv2 = ['not_stressed', 'stressed', 'not_stressed', 'stressed']

    #prediction with 4 classes
    livepredictions4 = conv[liveabc[0]]
    #prediction with 2 classes
    livepredictions2 = conv2[liveabc[0]]
    
    return livepredictions2

def extract_feature(file_name, offst=0.1):
    X, sample_rate = librosa.load(
        file_name, res_type='kaiser_fast', offset=offst)
    mfccs = np.mean(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=64).T, axis=0)
    return mfccs
# ---------------------- ROUTES --------------------------------------
@app.route('/login', methods=['POST'])
def login():

    auth = request.authorization
    if not auth or not auth.username or not auth.password:
        return make_response('Verification error', 401, {"WWW-Authentificate": "Login required"})
    user = User.query.filter_by(uname=auth.username).first()

    if not user:
        return make_response('Verification error', 401, {"WWW-Authentificate": "Login required"})

    if check_password_hash(user.password, auth.password):
        token = jwt.encode({'user_id': user.user_id, 'exp': datetime.datetime.utcnow(
        ) + datetime.timedelta(hours=168)}, app.config['SECRET_KEY'])
        print("User logged in:")
        print(user.user_id)
        return jsonify({"ok": True, "token": token.decode('UTF-8')})

    return make_response('Verification error', 401, {"WWW-Authentificate": "Login required"})


@app.route('/user', methods=['POST'])
def signup():
    data = request.get_json()
    hashed_password = generate_password_hash(data['password'], method='sha256')

    new_user = User(user_id=str(uuid.uuid4(
    )), uname=data['uname'], name=data['name'], lname=data['lname'], sex=data['sex'], password=hashed_password)

    if data['quote'] is not None:
        new_user.quote = data['quote']

    if data['country'] is not None:
        new_user.country = data['country']

    if data['weight'] is not None:
        new_user.weight = data['weight']

    db.session.add(new_user)
    db.session.commit()
    return jsonify({"ok": True, "message": "New user created!"})


@app.route('/user', methods=['GET'])
@token_checker
def get_user_data(current_user):
    user = User.query.filter_by(uname=current_user.uname).first()
    user_data = {}
    user_data["id"] = user.id
    user_data["user_id"] = user.user_id
    user_data["uname"] = user.uname
    user_data["name"] = user.name
    user_data["lname"] = user.lname
    user_data["quote"] = user.quote
    user_data["sex"] = user.sex
    user_data["country"] = user.country

    return jsonify({"ok": "true", "user": user_data})

@app.route('/user_change', methods=['POST'])
@token_checker
def change_user_data(current_user):
    data = request.get_json()
    user = User.query.filter_by(uname=current_user.uname).first()
    
    if data['name'] is not None:
        user.name = data['name']
    
    if hasattr(data, 'lname'):
        user.lname = data['lname']

    if data['country'] is not None:
        user.country = data['country']
    
    if data['sex'] is not None:
        user.sex = data['sex']
    
    if data['quote'] is not None:
        user.quote = data['quote']
    
    if data['weight'] is not None:
        user.weight = data['weight']
    

    user_data = {}
    user_data["id"] = user.id
    user_data["user_id"] = user.user_id
    user_data["uname"] = user.uname
    user_data["name"] = user.name
    user_data["lname"] = user.lname
    user_data["quote"] = user.quote
    user_data["sex"] = user.sex
    user_data["country"] = user.country
    user_data["weight"] = user.weight

    db.session.commit()


    return jsonify({"ok": "true", "new_data": user_data})


@app.route('/fdata', methods=['GET'])
@token_checker
def get_user_fdata(current_user):
    print(current_user.user_id)

    date_day_start = datetime.datetime.utcnow().replace(
        hour=0, minute=0, second=0, microsecond=0)

    date_week_start = (datetime.datetime.utcnow() - timedelta(days=6)).replace(
        hour=0, minute=0, second=0, microsecond=0)

    #print(date_week_start)

    udata = Fdata.query.filter_by(
        user_id=current_user.user_id).order_by(Fdata.id.desc()).first()

    udata_day_str = Fdata.query.filter_by(
        user_id=current_user.user_id).filter_by(mood="stressed").filter(date_day_start <= Fdata.date).all()

    udata_day_nstr = Fdata.query.filter_by(
        user_id=current_user.user_id).filter_by(mood="not_stressed").filter(date_day_start <= Fdata.date).all()

    udata_week_str = Fdata.query.filter_by(
        user_id=current_user.user_id).filter_by(mood="stressed").filter(date_week_start <= Fdata.date).all()

    udata_week_nstr = Fdata.query.filter_by(
        user_id=current_user.user_id).filter_by(mood="not_stressed").filter(date_week_start <= Fdata.date).all()

    #print(udata_week_nstr)

    if not udata:
        return jsonify({"ok": "false", "message": "No data to display"})

    user_fdata = {}
    user_fdata["mood"] = udata.mood
    user_fdata["mood_text"] = udata.mood_text
    user_fdata["weight"] = udata.weight
    user_fdata["hbeat"] = udata.hbeat
    user_fdata["udata_day_str"] = len(udata_day_str)
    user_fdata["udata_day_nstr"] = len(udata_day_nstr)
    user_fdata["udata_week_str"] = len(udata_week_str)
    user_fdata["udata_week_nstr"] = len(udata_week_nstr)

    return jsonify({"ok": "true", "fdata": user_fdata})


'''
@app.route('/fdata', methods=['POST'])
@token_checker
def update_user_fdata(current_user):

    data = request.form
    files = request.files
    print(data)
    print(request.files)

    if(files["audio"]):
        print(files["audio"])
        files["audio"].save('tempFiles/' + files["audio"].filename)

        def extract_feature(file_name, offst=0.5):
            X, sample_rate = librosa.load(
                file_name, res_type='kaiser_fast', offset=offst)
            stft = np.abs(librosa.stft(X))
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
            mel = np.mean(librosa.feature.melspectrogram(
                X, sr=sample_rate).T, axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(
                S=stft, sr=sample_rate).T, axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(
                y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            return mfccs, chroma, mel, contrast, tonnetz

        mfccs, chroma, mel, contrast, tonnetz = extract_feature(
            'tempFiles/'+files["audio"].filename, 0)
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])

        live = pd.DataFrame(data=ext_features)
        live = live.stack().to_frame().T
        livecnn = np.expand_dims(live, axis=2)
        # loading json and creating model
        from keras.models import model_from_json

        json_file5 = open(
            'model/2_class_stress_out_of_8_class_En.json', 'r')

        eng2 = json_file5.read()
        json_file5.close()

        eng_stress_model = model_from_json(eng2)

        # load weights into new model
        eng_stress_model.load_weights(
            "model/2_class_stress_out_of_8_class_En.h5")
        print("Loaded model from disk")

        opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
        eng_stress_model.compile(
            loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        livepreds = eng_stress_model.predict(livecnn,
                                             batch_size=32,
                                             verbose=1)

        livepreds1 = livepreds.argmax(axis=1)
        liveabc = livepreds1.astype(int).flatten()
        conv = ['not_stressed', 'stressed']

        livepredictions = conv[liveabc[0]]
        print("Result: " + livepredictions)
        os.remove('tempFiles/' + files["audio"].filename)

    new_fdata = Fdata(user_id=current_user.user_id,
                      mood=data["mood"], hbeat=data["hbeat"], weight=data["weight"])

    db.session.add(new_fdata)
    db.session.commit()
    return jsonify({"ok": "true", "message": "Fdata updated", "new_mood": livepredictions})
'''


'''@app.route('/test_feature_64', methods=['POST'])
@token_checker
def audio_data_feature(current_user):

    data = request.form

    if(data["features"]):
        print(data["features"])
        j_data = json.loads(data["features"])
        mfccs = j_data['fe']
        print(mfccs)
        print(mfccs[0])

        live = pd.DataFrame(data=mfccs)
        live = live.stack().to_frame().T

        livecnn = np.expand_dims(live, axis=2)

        # loading json and creating model with 4 classes
        from keras.models import model_from_json
        json_file5 = open(
            'model/Stress_recog_MFCC_with_gender.json', 'r')
        eng2 = json_file5.read()
        json_file5.close()
        eng_stress_model = model_from_json(eng2)

        # load weights into new model
        eng_stress_model.load_weights(
            "model/Stress_recog_MFCC_with_gender.h5")
        print("Loaded model from disk")

        # compile the model
        opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
        eng_stress_model.compile(
            loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # predict!
        livepreds = eng_stress_model.predict(livecnn,
                                             batch_size=32,
                                             verbose=1)

        # convert prediction to text
        livepreds1 = livepreds.argmax(axis=1)
        liveabc = livepreds1.astype(int).flatten()
        conv = ['female_not_stressed', 'female_stressed', 'male_not_stressed',
                'male_stressed']

        # array for conversion into binary class (stressed/not_stressed)
        conv2 = ['not_stressed', 'stressed', 'not_stressed', 'stressed']

        # prediction with 4 classes
        livepredictions4 = conv[liveabc[0]]
        # prediction with 2 classes
        livepredictions2 = conv2[liveabc[0]]

        print(livepredictions2)

        return jsonify({"ok": "true", "message": "Fdata updated", "new_mood": livepredictions2})
        '''


@app.route('/test_feature', methods=['POST'])
@token_checker
def audio_data_feature(current_user):

    data = request.form

    if(data["features"]):
        print(data["features"])
        j_data = json.loads(data["features"])
        mfccs = j_data['fe']
        print(mfccs)
        print(mfccs[0])

        live = pd.DataFrame(data=mfccs)
        live = live.stack().to_frame().T

        livecnn = np.expand_dims(live, axis=2)

        # loading json and creating model with 4 classes
        from keras.models import model_from_json
        json_file5 = open('model/Stress_recog_MFCC_with_gender.json', 'r')
        eng2 = json_file5.read()
        json_file5.close()
        eng_stress_model = model_from_json(eng2)

        # load weights into new model
        eng_stress_model.load_weights("model/Stress_recog_MFCC_with_gender.h5")
        print("Loaded model from disk")

        # compile the model
        opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
        eng_stress_model.compile(
            loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # predict!
        livepreds = eng_stress_model.predict(livecnn,
                                             batch_size=32,
                                             verbose=1)

        # convert prediction to text
        livepreds1 = livepreds.argmax(axis=1)
        liveabc = livepreds1.astype(int).flatten()
        conv = ['female_not_stressed', 'female_stressed', 'male_not_stressed',
                'male_stressed']

        # array for conversion into binary class (stressed/not_stressed)
        conv2 = ['not_stressed', 'stressed', 'not_stressed', 'stressed']

        # prediction with 4 classes
        livepredictions4 = conv[liveabc[0]]
        # prediction with 2 classes
        livepredictions2 = conv2[liveabc[0]]

        print(livepredictions2)

        s_grad = ["Stressed", "A litle stressed", "Quite stressed",
                  "Moderately stressed", "Slightly stressed"]

        ns_grad = ["Happy", "Moderate", "Stressless", "Normal", "Good"]

        mood_text = ""

        if(livepredictions2 == "stressed"):
            rn = randint(0, len(s_grad)-1)
            mood_text = s_grad[rn]
        else:
            rn = randint(0, len(ns_grad)-1)
            mood_text = ns_grad[rn]

        return jsonify({"ok": "true", "message": "Fdata updated", "new_mood": livepredictions2, "mood_text": mood_text})


@app.route('/test_pre_control', methods=['POST'])
@token_checker
def audio_data(current_user):

    data = request.form

    if(data["audio"]):
        # print(data["audio"])
        # audio_64 = base64.b64decode(data["audio"] + '=' * (-len(data["audio"]) % 4))
        print("Audio: start")
        print(data["audio"][0:100])
        audio_64 = base64.b64decode(data["audio"])
        # os.remove('tempFiles/audio.wav')
        audio_file = open('tempFiles/audio.wav', 'wb')
        audio_file.write(audio_64)
        audio_file.close()

        def extract_feature(file_name, offst=0.5):
            X, sample_rate = librosa.load(
                file_name, res_type='kaiser_fast', offset=offst)
            stft = np.abs(librosa.stft(X))
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
            mel = np.mean(librosa.feature.melspectrogram(
                X, sr=sample_rate).T, axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(
                S=stft, sr=sample_rate).T, axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(
                y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            return mfccs, chroma, mel, contrast, tonnetz

        mfccs, chroma, mel, contrast, tonnetz = extract_feature(
            'tempFiles/audio.wav', 0)
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])

        live = pd.DataFrame(data=ext_features)
        live = live.stack().to_frame().T
        livecnn = np.expand_dims(live, axis=2)
        # loading json and creating model
        from keras.models import model_from_json

        json_file5 = open(
            'model/2_class_stress_out_of_8_class_En.json', 'r')

        eng2 = json_file5.read()
        json_file5.close()

        eng_stress_model = model_from_json(eng2)

        # load weights into new model
        eng_stress_model.load_weights(
            "model/2_class_stress_out_of_8_class_En.h5")
        print("Loaded model from disk")

        opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
        eng_stress_model.compile(
            loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        livepreds = eng_stress_model.predict(livecnn,
                                             batch_size=32,
                                             verbose=1)

        livepreds1 = livepreds.argmax(axis=1)
        liveabc = livepreds1.astype(int).flatten()
        conv = ['not_stressed', 'stressed']

        livepredictions = conv[liveabc[0]]
        print("Result: " + livepredictions)
        # os.remove('tempFiles/audio.wav')
        s_grad = ["Stressed", "A litle stressed", "Quite stressed",
                  "Moderately stressed", "Slightly stressed"]

        ns_grad = ["Happy", "Moderate", "Stressless", "Normal", "Good"]

        mood_text = ""

        if(livepredictions == "stressed"):
            rn = randint(0, len(s_grad)-1)
            mood_text = s_grad[rn]
        else:
            rn = randint(0, len(ns_grad)-1)
            mood_text = ns_grad[rn]

        # ////Add new fdata entry
        new_fdata = Fdata(user_id=current_user.user_id,
                          mood=livepredictions, hbeat=75, weight=99, date=datetime.datetime.utcnow(), mood_text=mood_text)

        db.session.add(new_fdata)
        db.session.commit()

        return jsonify({"ok": "true", "message": "Fdata updated", "new_mood": livepredictions, "fdata_id": new_fdata.id, "mood_text": mood_text})



@app.route('/test', methods=['POST'])
@token_checker
def audio_data_user_tuned(current_user):

    data = request.form
    user = User.query.filter_by(uname=current_user.uname).first()

    if(data["audio"]):
        # print(data["audio"])
        # audio_64 = base64.b64decode(data["audio"] + '=' * (-len(data["audio"]) % 4))
        print("Audio: start")
        print(data["audio"][0:100])
        audio_64 = base64.b64decode(data["audio"])
        # os.remove('tempFiles/audio.wav')
        audio_file = open('tempFiles/audio.wav', 'wb')
        audio_file.write(audio_64)
        audio_file.close()

        mfccs = extract_feature(
            'tempFiles/audio.wav', 0)
        
        #print(mfccs)

        live = pd.DataFrame(data=mfccs)
        live = live.stack().to_frame().T

        livecnn = np.expand_dims(live, axis=2)

        #loading models from disk
        path_to_users_model_folder = 'model#' + user.user_id
        path_to_def_model_folder = 'model'

        if os.path.isdir(path_to_users_model_folder):
            print("exists")
            model_path = path_to_users_model_folder + '/' + user.user_id + '_Stress_recog_64MFCC_with_gender.json'
            weights_path = path_to_users_model_folder + '/' + user.user_id + '_Stress_recog_64MFCC_with_gender.h5'
        else:
            print("not exists")
            model_path = path_to_def_model_folder + '/' + 'Stress_recog_64MFCC_with_gender.json'
            weights_path = path_to_def_model_folder + '/' + 'Stress_recog_64MFCC_with_gender.h5'

        # loading json and creating model with 4 classes
        from keras.models import model_from_json
        json_file5 = open(model_path, 'r')
        eng2 = json_file5.read()
        json_file5.close()
        eng_stress_model = model_from_json(eng2)

        # load weights into new model
        eng_stress_model.load_weights(weights_path)
        print("Loaded model from disk")

        #compile the model
        opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
        eng_stress_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        livepredictions = getPrediction(eng_stress_model, livecnn)
        print("Result: " + livepredictions)
        # os.remove('tempFiles/audio.wav')
        s_grad = ["Stressed", "A litle stressed", "Quite stressed",
                  "Moderately stressed", "Slightly stressed"]

        ns_grad = ["Happy", "Moderate", "Stressless", "Normal", "Good"]

        mood_text = ""

        if(livepredictions == "stressed"):
            rn = randint(0, len(s_grad)-1)
            mood_text = s_grad[rn]
        else:
            rn = randint(0, len(ns_grad)-1)
            mood_text = ns_grad[rn]

        mfccs_str = "" + base64.b64encode(mfccs).decode()
        # ////Add new fdata entry
        user = User.query.filter_by(uname=current_user.uname).first()
        print(user.weight)
        new_fdata = Fdata(user_id=current_user.user_id,
                          mood=livepredictions, hbeat=75, weight=user.weight, date=datetime.datetime.utcnow(), mood_text=mood_text, features=mfccs_str)

        db.session.add(new_fdata)
        db.session.commit()

        return jsonify({"ok": "true", "message": "Fdata updated", "new_mood": livepredictions, "fdata_id": new_fdata.id, "mood_text": mood_text})


@app.route('/test_web', methods=['POST'])
@token_checker
@cross_origin()
def audio_data_web(current_user):

    data = request.form
    files = request.files
    print(data)
    print(request.files)

    if(files["audio"]):
        print(files["audio"])
        files["audio"].save('tempFiles_web/audio.wav')

        def extract_feature(file_name, offst=0.5):
            X, sample_rate = librosa.load(
                file_name, res_type='kaiser_fast', offset=offst)
            stft = np.abs(librosa.stft(X))
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
            mel = np.mean(librosa.feature.melspectrogram(
                X, sr=sample_rate).T, axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(
                S=stft, sr=sample_rate).T, axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(
                y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            return mfccs, chroma, mel, contrast, tonnetz

        mfccs, chroma, mel, contrast, tonnetz = extract_feature(
            'tempFiles_web/audio.wav', 0)
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])

        live = pd.DataFrame(data=ext_features)
        live = live.stack().to_frame().T
        livecnn = np.expand_dims(live, axis=2)
        # loading json and creating model
        from keras.models import model_from_json

        json_file5 = open(
            'model_web/2_class_stress_out_of_8_class_En.json', 'r')

        eng2 = json_file5.read()
        json_file5.close()

        eng_stress_model = model_from_json(eng2)

        # load weights into new model
        eng_stress_model.load_weights(
            "model_web/2_class_stress_out_of_8_class_En.h5")
        print("Loaded model from disk")

        opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
        eng_stress_model.compile(
            loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        livepreds = eng_stress_model.predict(livecnn,
                                             batch_size=32,
                                             verbose=1)

        livepreds1 = livepreds.argmax(axis=1)
        liveabc = livepreds1.astype(int).flatten()
        conv = ['not_stressed', 'stressed']

        livepredictions = conv[liveabc[0]]
        print("Result: " + livepredictions)
        return jsonify({"ok": "true", "message": "Fdata updated", "new_mood": livepredictions})


@app.route('/feedback', methods=['POST'])
@token_checker
@cross_origin()
def feedback_handler(current_user):

    data = request.form
    files = request.files
    # print(data)
    # print(request.files)

    fdata = Fdata.query.filter_by(id=data["lastId"]).first()
    user = User.query.filter_by(uname=current_user.uname).first()

    if (data["feedback"] == 'false'):
        fdata.feedback = False
    else:
        fdata.feedback = True
    db.session.commit()
    # print(fdata.feedback)

    # feedback and user_male are info that is taken from the user!!!!!!!!!!!!!!!!
    liveabc = None
    user_male = (user.sex == 'M')
    liveprediction = fdata.mood

    #mapping male and predicition from data to liveabc
    if user_male:
        if liveprediction == "stressed":
            liveabc = 3
        else:
            liveabc = 2
    else:
        if liveprediction == "stressed":
            liveabc = 1
        else:
            liveabc = 0
    
    #loading models from disk
    path_to_users_model_folder = 'model#' + user.user_id
    path_to_def_model_folder = 'model'

    if os.path.isdir(path_to_users_model_folder):
        print("exists")
        model_path = path_to_users_model_folder + '/' + user.user_id + '_Stress_recog_64MFCC_with_gender.json'
        weights_path = path_to_users_model_folder + '/' + user.user_id + '_Stress_recog_64MFCC_with_gender.h5'
    else:
        print("not exists")
        model_path = path_to_def_model_folder + '/' + 'Stress_recog_64MFCC_with_gender.json'
        weights_path = path_to_def_model_folder + '/' + 'Stress_recog_64MFCC_with_gender.h5'

    # loading json and creating model with 4 classes
    from keras.models import model_from_json
    json_file5 = open(model_path, 'r')
    eng2 = json_file5.read()
    json_file5.close()
    eng_stress_model = model_from_json(eng2)

    # load weights into new model
    eng_stress_model.load_weights(weights_path)
    print("Loaded model from disk")
    
    epNum = 0
    if user_male == True:
        it = (liveabc + 1) % 2 + 2
    else:
        it = (liveabc + 1) % 2
    conv2 = ['not_stressed', 'stressed', 'not_stressed', 'stressed']
    
    check_pred = conv2[it]
    while(True):
        if (data["feedback"] == 'false') and (liveprediction != check_pred) and (epNum <= 3):
            

            for layer in eng_stress_model.layers[:14]:
                layer.trainable=False
            for layer in eng_stress_model.layers[14:]:
                layer.trainable=True

            #compile the model
            opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

            eng_stress_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

            y_train = np.zeros((1, 4))
            y_train[0][it] = 1

            mfccs_str = fdata.features
            mfccs_buff = base64.decodebytes(mfccs_str.encode())
            mfccs = np.frombuffer(mfccs_buff, dtype=np.float64)
            #print(mfccs)


            live = pd.DataFrame(data=mfccs)
            live = live.stack().to_frame().T

            livecnn = np.expand_dims(live, axis=2)

            eng_stress_model.fit(livecnn, y_train, batch_size=32, epochs=1, validation_data=(livecnn, y_train))


            liveprediction = getPrediction(eng_stress_model, livecnn)

            epNum += 1

            print("liveprediction: " + liveprediction)
            print("check_pred: " + check_pred)

        else:
            break
        print(epNum)

    if epNum > 0:
        user_model_path = path_to_users_model_folder + '/' + user.user_id + '_Stress_recog_64MFCC_with_gender.json'
        user_weights_path = path_to_users_model_folder + '/' + user.user_id + '_Stress_recog_64MFCC_with_gender.h5'

        model_name = user.user_id + '_Stress_recog_64MFCC_with_gender.h5'
        save_dir = path_to_users_model_folder
        # Save model and weights
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        model_path = os.path.join(save_dir, model_name)
        eng_stress_model.save(model_path)
        print('Saved trained model at %s ' % model_path)

        import json
        model_json = eng_stress_model.to_json()
        with open(user_model_path, "w") as json_file:
            json_file.write(model_json)

    return jsonify({"ok": "true", "message": "Updated last record and (probably) tuned model: " + data["lastId"]})


@app.route('/fdata10', methods=['GET'])
@token_checker
def get_user_fdata10(current_user):
    print(current_user.user_id)
    udata = Fdata.query.filter_by(
        user_id=current_user.user_id).order_by(Fdata.id.desc()).limit(15).all()

    if not udata:
        return jsonify({"ok": "false", "message": "No data to display"})

    # print(udata)
    fdataList = []
    for fd in udata:
        f = {}
        f["mood"] = fd.mood
        f["date"] = fd.date + timedelta(hours=6)
        f["feedback"] = fd.feedback
        fdataList.append(f)
    # print(fdataList)

    return jsonify({"ok": "true", "fdataList": fdataList})


if __name__ == '__main__':
    app.run(host='0.0.0.0')
