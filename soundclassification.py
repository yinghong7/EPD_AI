# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
from librosa import display
#np version 1.25.2
#tf version 2.15.0
#librosa version 0.10.1


def preprocess(input_file_path):
  audio, sr = librosa.load(path = input_file_path, sr=None)
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
  return [mels_out,zcritem,centitem]

#files=["Percussive over 10_wall removal_source_16_10.wav","Handheld less 10_wall removal_source_03_3.wav","Electrical driller_hole making_receiver_05_7.wav","Wall chaser_chanelling_source_06_19_1.wav"]
#audio_p='/content/drive/MyDrive/EPD_DATA/Sound/'+ files[1]

def classify(input_file_path):
  pmeclasses=["Handheld percussive breakers >10kg","Handheld percussive breakers <10kg","Others","Electric Percussive Drill"]
  testfeat=preprocess(audio_p)
  loaded_model = tf.keras.models.load_model('/content/drive/MyDrive/NoiseModel.keras')
  preds=loaded_model.predict(testfeat)
  indmax = np.argmax(preds[0])
  return pmeclasses[indmax]

predic=classify(audio_p)
