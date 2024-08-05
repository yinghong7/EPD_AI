# -*- coding: utf-8 -*-
import time
import numpy as np
import tensorflow as tf
import librosa
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from pydub import AudioSegment
import soundfile as sf
import os, sys
import asyncio
import uvicorn
#from librosa import display
#np version 1.25.2
#tf version 2.15.0
#librosa version 0.10.1


from fastapi import FastAPI

app = FastAPI()

#class InputPath(BaseModel):
    #file_path: str

loaded_model = tf.keras.models.load_model("EPD_AI-main\soundmodel.h5")
AudioSegment.converter = os.getcwd()+ "\\ffmpeg.exe"
AudioSegment.ffprobe = os.getcwd()+ "\\ffprobe.exe"

@app.get("/")
async def read_root():
    return {"message": "Prediction server is running"}

@app.post("/predict") #file_path= the file name of the wav file e.g. blade.wav
async def predict(file_path: str):
    # Convert input features to a numpy array
    #print(type(file_path.file_path))
    file_loc="EPD_AI-main/" #Please change this to the folder location with the wav files
    real_file_path =file_loc+file_path
    # return {"type_file_path":type(file_path)}
    wavfile = real_file_path.replace('m4a', 'wav')
    audio, sr = librosa.load(path = wavfile, sr=22050)
    #sound = AudioSegment.from_file(real_file_path, format='m4a')
    #wav_file = sound.export (wavfile, format='wav')
    #audio, sr = librosa.load(path = wav_file, sr=22050)
    #Preproccesing
    audio = librosa.effects.time_stretch(y=audio, rate=len(audio)/sr)
    zcr=librosa.feature.zero_crossing_rate(audio)
    zcritem=zcr.flatten()
    zcritem=zcritem.reshape(1,44)
    cent = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    centitem=cent.flatten()
    centitem=cent.reshape(1,44)
    mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=231,norm=np.inf)
    mels_db = librosa.power_to_db(S=mels, ref=1.0)
    mels_out=mels_db.reshape((1,128, 16, 6))
    features=[mels_out,zcritem,centitem]
    # Perform model inference
    predictions = loaded_model.predict(features)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Map predicted class index to class label
    class_labels = ["Handheld percussive breakers >10kg","Handheld percussive breakers <10kg","Others","Electric Percussive Drill"]
    predicted_label = class_labels[predicted_class]

    # Convert predicted class to Python integer
    predicted_class = int(predicted_class)

    # Create the response JSON
    response_data = {"predicted_class": predicted_class, "predicted_label": predicted_label}

    return predicted_label

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8086) #change as necessary