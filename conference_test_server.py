# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
import argparse
from pydub import AudioSegment
import soundfile as sf
import os, sys, glob
import pickle
import pandas as pd

# AudioSegment.converter = os.getcwd()+ "/ffmpeg.exe"
# AudioSegment.ffprobe = os.getcwd()+ "/ffprobe.exe"

def preprocess(input_file_path):
  audio, sr = librosa.load(path = input_file_path, sr=22050)
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
  cens = librosa.feature.chroma_cens (y = audio, sr = sr)
  cens = cens.reshape (1, 12*44)
  return [mels_out,cens,centitem, zcritem]

def classify(input_file_path):
  #pmeclasses=["Handheld percussive breakers >10kg","Handheld percussive breakers <10kg","Others","Electric Percussive Drill"]
  pmeclasses=["Electric Percussive Drill", "Handheld percussive breakers <10kg","Others","Handheld percussive breakers >10kg"]
  testfeat = preprocess(input_file_path)
  loaded_model = tf.keras.models.load_model('/Users/ying/Downloads/soundmodel_conference_v8.h5')
  # extra_model = pickle.load(open('below and over 10 conference.pkl', 'rb'))
  preds=loaded_model.predict(testfeat)
  indmax = np.argmax(preds[0])
  return pmeclasses[indmax]
  # indmax = np.argmax(preds[0])
  # if pmeclasses[indmax] in ["Handheld percussive breakers >10kg", "Handheld percussive breakers <10kg"]:
  #   preds = extra_model.predict (testfeat [0].reshape(1, 128*16*6))
  #   return preds
  # else: return pmeclasses[indmax]

def main():
    folderpath = os.getcwd() +'/0108_EPD testing'
    result, name = [], []
    for wav_file in glob.glob(os.path.join(folderpath, '*')):
      pred = classify(wav_file)
      if type(pred) != str:
         result.append (pred[0])
      else: result.append (pred)
      name.append (wav_file.split('/')[-1].split('_')[0])
    df = pd.DataFrame (list(zip(result, name)), columns = ['Pred_result', 'True_label'])
    df.to_csv ('result.csv')
      # with open('result.pkl', 'wb') as file: 
      #   pickle.dump(result, file) 
      # with open('name.pkl', 'wb') as file: 
      #   pickle.dump(name, file) 
    

if __name__ == "__main__":
    main ()