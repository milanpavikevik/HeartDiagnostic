from typing import Optional
from fastapi import FastAPI, Request, Depends, BackgroundTasks,  File, UploadFile
from fastapi.templating import Jinja2Templates
import os
from fastapi.responses import HTMLResponse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
# Audio
import librosa
import librosa.display

import models
from database import SessionLocal, engine
from sqlalchemy.orm import Session
from pydantic import BaseModel
from models import Client
import numpy as np
from tqdm import tqdm
#import itertool
import pandas as pd

app = FastAPI()
models.Base.metadata.create_all(bind=engine)
tempplates = Jinja2Templates(directory="templetes")


def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


@app.get("/")
def read_root(request: Request, db: Session = Depends(get_db)):
    patients = db.query(Client).all()

    return tempplates.TemplateResponse("dashboard.html",{
        "request":request,
        "patients":patients
    })


def extract_features(audio_path, offset):
    y, sr = librosa.load(audio_path, offset=offset, duration=2.5)
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=2048,
                                       hop_length=512,
                                       n_mels=128)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
    return mfccs


def predict_the_data(wavfile):

    content = wavfile.file
    # load model
    model = load_model("C:\\Users\\user\\Desktop\\LokaFiles\\heartbeat_classifier_(test_85%).h5")
    # classification
    classify_file = content
    x_test = []
    Fpred="nema"
    Fconf=0
    x_test.append(extract_features(classify_file, 0.5))
    x_test = np.asarray(x_test)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    pred = model.predict(x_test, verbose=1)
    print(pred)
    pred_class = model.predict_classes(x_test)
    print(pred_class[0])
    if pred_class[0] == 0:
        print("Extrasystole heartbeat")
        print("confidence of prediction:", pred[0][0])
        Fpred = "Extrasystole heartbeat"
        Fconf = round((pred[0][0]*100), 2)
    elif (pred_class[0] == 1):
        print("Murmur heartbeat")
        print("confidence of prediction:", pred[0][1])
        Fpred = "Murmur heartbeat"
        Fconf = round((pred[0][1] * 100), 2)
    else:
        print("Normal heartbeat")
        print("confidence of prediction:", pred[0][2])
        Fpred = "Normal heartbeat"
        Fconf = round((pred[0][2] * 100), 2)

    return Fpred, Fconf


@app.post("/uploadfile/")
async def create_upload_file(request: Request, file: UploadFile = File(...),  db: Session = Depends(get_db)):

    result1, result2 = predict_the_data(file)
    clasa = Client()
    clasa.prediction = result1
    clasa.confidenceLevel = result2
    db.add(clasa)
    db.commit()
    patients = db.query(Client).all()
    return tempplates.TemplateResponse("dashboardPost.html", {
        "request": request,
        "patients": patients
    })
