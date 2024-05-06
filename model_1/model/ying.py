# cd 
# C:/Users/Ying.Hong/Arup/EPD AI Noise - General/8_Testing audioclips online resources

# python

# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
import argparse
from pydub import AudioSegment
import soundfile as sf
import os, sys, glob
import pandas as pd
import matplotlib.pyplot as plt

AudioSegment.converter = os.getcwd()+ "\\ffmpeg.exe"
AudioSegment.ffprobe = os.getcwd()+ "\\ffprobe.exe"

def wav_transform (input_file_path):
  wavfile = input_file_path.replace ('m4a', 'wav')
  sound = AudioSegment.from_file(input_file_path, format='m4a')
  wav_file = sound.export (wavfile, format='wav')
  return wav_file

def preprocess(input_file_path):
  audio, sr = librosa.load(path = input_file_path, sr=22050)
  audio = librosa.effects.time_stretch(y=audio, rate=len(audio)/sr)
  zcr = librosa.feature.zero_crossing_rate(audio)
  zcritem = zcr.flatten()
  zcritem = zcritem.reshape(1, 44)
  cent = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
  centitem = cent.flatten()
  centitem = cent.reshape(1, 44)
  mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=231,norm=np.inf)
  mels_db = librosa.power_to_db(S=mels, ref=1.0)
  mels_out = mels_db.reshape((1, 128, 16, 6))
  return mels_out, zcritem, centitem

def classify(input_file_path):
  pmeclasses = ["Handheld percussive breakers >10kg","Handheld percussive breakers <10kg","Others","Electric Percussive Drill"]
  testfeat = preprocess(input_file_path)
  loaded_model = tf.keras.models.load_model('soundmodel.h5') #path for AI model
  preds=loaded_model.predict(testfeat)
  indmax = np.argmax(preds[0])
  return pmeclasses[indmax]


def compare_prediction (folder_to_visit):
    melspectrogram, zcr_f, bandwidth, folder, filename, predict_label = [], [], [], [], [], []
    for i in folder_to_visit:
        for file in glob.glob(os.path.join(i, '*.wav')):
            #wav_file = wav_transform (file)
            mels_out, zcritem, centitem = preprocess (file)
            melspectrogram.append (mels_out)
            zcr_f.append (zcritem)
            bandwidth.append (centitem)
            folder.append (i)
            filename.append (file)
            predic = classify(file)
            predict_label.append (predic)
    df = pd.DataFrame(list(zip(folder, filename, predict_label, melspectrogram, zcr_f, bandwidth)), columns =['Folder name', 'File name', 'Predicted label', 'melspectrogram', 'zcr', 'bandwidth'])
    return df


#######################################
class plot:
    def __init__ (self, files):
        self.meldb, self.bandwidth, self.zcr_ls = [], [], []
        for i in files:
            self.audio, self.sr = librosa.load(path = i, sr=22050)
            self.audio = librosa.effects.time_stretch(y = self.audio, rate=len(self.audio)/self.sr)
            self.meldb.append (librosa.power_to_db(S=librosa.feature.melspectrogram(y = self.audio, sr=22050,n_fft=2048, hop_length=231,norm=np.inf), ref=1.0))
            self.bandwidth.append (librosa.feature.spectral_bandwidth(y=self.audio, sr=self.sr).flatten().reshape (1, 44))
            self.zcr_ls.append (librosa.feature.zero_crossing_rate(self.audio).flatten().reshape (1, 44))              
    def mel_plot (self):
        fig, ax = plt.subplots (nrows = 3, ncols = 1)
        img1 = librosa.display.specshow(data=self.meldb[0], sr=self.sr, x_axis='time', y_axis='mel', ax = ax[0])
        ax[0].set(title = 'Original recording')
        img2 = librosa.display.specshow(data=self.meldb[1], sr=self.sr, x_axis='time', y_axis='mel', ax = ax[1])
        ax[1].set(title = 'Built-in app recording')
        #img3 = librosa.display.specshow(data=self.meldb[2], sr=self.sr, x_axis='time', y_axis='mel', ax = ax[2])
        #ax[2].set(title = 'AI Noise App recording')
        plt.show()
    def bandwidth_plot (self):
        length_bin = np.arange (0, 44, dtype = int)
        plt.plot(length_bin.reshape (44,), self.bandwidth[0].reshape (44,), label = 'Original recording')
        plt.plot(length_bin.reshape (44,), self.bandwidth[1].reshape (44,), label = 'Built-in app recording')
        #plt.plot(length_bin.reshape (44,), self.bandwidth[2].reshape (44,), label = 'AI Noise App recording')
        plt.title ('Bandwidth')
        plt.ylim (0, 4000)
        plt.legend ()
        plt.show()
    def zcr_plot (self):
        length_bin = np.arange (0, 44, dtype = int)
        plt.plot(length_bin.reshape (44,), self.zcr_ls[0].reshape (44,), label = 'Original recording')
        plt.plot(length_bin.reshape (44,), self.zcr_ls[1].reshape (44,), label = 'Built-in app recording')
        #plt.plot(length_bin.reshape (44,), self.zcr_ls[2].reshape (44,), label = 'AI Noise App recording')
        plt.title ('ZCR')
        plt.ylim (0, 1)
        plt.legend ()
        plt.show()   
    def scatter_plot (self):
        fig, (ax1, ax2, ax3) = plt.subplots (nrows = 1, ncols = 3)
        a1, b1 = np.polyfit(self.zcr_ls[0].reshape (44,), self.zcr_ls[1].reshape (44,), 1) #fit a line
        a2, b2 = np.polyfit(self.zcr_ls[1].reshape (44,), self.zcr_ls[2].reshape (44,), 1)
        a3, b3 = np.polyfit(self.zcr_ls[0].reshape (44,), self.zcr_ls[2].reshape (44,), 1)
        ax1.scatter (self.zcr_ls[0], self.zcr_ls[1]) #same phones
        ax1.plot (self.zcr_ls[0].reshape(44,), self.zcr_ls[0].reshape (44,)*a1+b1)
        ax1.set_title ('Original recording vs Built-in app recording')
        ax2.scatter (self.zcr_ls[1], self.zcr_ls[2]) #same recording display
        ax2.plot (self.zcr_ls[1].reshape(44,), self.zcr_ls[1].reshape (44,)*a2+b2)
        ax2.set_title ('Built-in app recording vs AI Noise App recording')
        ax3.scatter (self.zcr_ls[0], self.zcr_ls[2]) 
        ax3.plot (self.zcr_ls[0].reshape(44,), self.zcr_ls[0].reshape (44,)*a1+b1)
        ax3.set_title ('Original recording vs AI Noise App recording')
        plt.show()

###############################################
files_01 = [os.getcwd()+'\\Tested audio\\Handheld less 10_wall removal_receiver_01_3 1.wav', 
os.getcwd()+'\\iPhone built-in app recording\\Festival Walk Management Office 37.wav', 
os.getcwd()+'\\iPhone app recording\\sound-20240423040302.wav']
files_02 = [os.getcwd()+'\\Tested audio\\Handheld less 10_wall removal_receiver_01_6.wav', 
os.getcwd()+'\\iPhone built-in app recording\\Festival Walk Management Office 38.wav', 
os.getcwd()+'\\iPhone app recording\\sound-20240423041028.wav']
files_03 = ['\\Tested audio\\Handheld less 10_wall removal_receiver_03_5.wav', 
'\\iPhone built-in app recording\\Festival Walk Management Office 39.wav', 
'\\iPhone app recording\\sound-20240423041206.wav']
files_04 = [os.getcwd()+'\\Tested audio\\Handheld less 10_wall removal_receiver_03_8.wav', 
os.getcwd()+'\\iPhone built-in app recording\\Festival Walk Management Office 40.wav', 
os.getcwd()+'\\iPhone app recording\\sound-20240423041709.wav']
files_05 = [os.getcwd()+'\\Tested audio\\Electrical drill_drilling_source_05_10.wav', 
os.getcwd()+'\\iPhone built-in app recording\\Festival Walk Management Office 54.wav', 
os.getcwd()+'\\iPhone app recording\\sound-20240429030125.wav']
files_06 = [os.getcwd()+'\\Tested audio\\Electrical driller_hole making_receiver_01_7_1.wav', 
os.getcwd()+'\\iPhone built-in app recording\\Festival Walk Management Office 55.wav', 
os.getcwd()+'\\iPhone app recording\\sound-20240429030259.wav']
files_07 = [os.getcwd()+'\\Tested audio\\Percussive over 10_wall removal_receiver_09_10.wav', 
os.getcwd()+'\\iPhone built-in app recording\\Festival Walk Management Office 56.wav', 
os.getcwd()+'\\iPhone app recording\\sound-20240429030409.wav']
files_08 = [os.getcwd()+'\\Tested audio\\Percussive over 10_wall removal_receiver_17_7.wav', 
os.getcwd()+'\\iPhone built-in app recording\\Festival Walk Management Office 57.wav', 
os.getcwd()+'\\iPhone app recording\\sound-20240429030608.wav']
files_09 = [os.getcwd()+'\\Tested audio\\Percussive over 10_wall removal_source_06_8.wav', 
os.getcwd()+'\\iPhone built-in app recording\\Festival Walk Management Office 58.wav', 
os.getcwd()+'\\iPhone app recording\\sound-20240429030741.wav']
files_10 = [os.getcwd()+'\\Tested audio\\Percussive over 10_wall removal_source_13_10.wav', 
os.getcwd()+'\\iPhone built-in app recording\\Festival Walk Management Office 59.wav', 
os.getcwd()+'\\iPhone app recording\\sound-20240429030837.wav']


###################################################
below10kg_youtube = [os.getcwd()+'\\Tested audio\\below10kg2_4.wav', os.getcwd()+'\\Tested audio\\below10kg3_1.wav', os.getcwd()+'\\Tested audio\\below10kg3_2.wav',
os.getcwd()+'\\Tested audio\\below10kg3_3.wav', os.getcwd()+'\\Tested audio\\below10kg3_4.wav']

below10kg_collected = [os.getcwd()+'\\Tested audio\\Handheld less 10_wall removal_receiver_01_3.wav',
os.getcwd()+'\\Tested audio\\Handheld less 10_wall removal_receiver_01_6.wav', os.getcwd()+'\\Tested audio\\Handheld less 10_wall removal_receiver_03_5.wav',
os.getcwd()+'\\Tested audio\\Handheld less 10_wall removal_receiver_03_8.wav']

over10kg_youtube = [os.getcwd()+'\\Tested audio\\over10kg2_1.wav', os.getcwd()+'\\Tested audio\\over10kg3_10.wav', os.getcwd()+'\\Tested audio\\over10kg3_11.wav']
over10kg_collected = [os.getcwd()+'\\Tested audio\\Percussive over 10_wall removal_receiver_09_10.wav', os.getcwd()+'\\Tested audio\\Percussive over 10_wall removal_receiver_17_7.wav',
os.getcwd()+'\\Tested audio\\Percussive over 10_wall removal_source_06_8.wav']

electrical_youtube = [os.getcwd()+'\\Tested audio\\electricdrill3_14.wav', os.getcwd()+'\\Tested audio\\electricdrill3_17.wav',os.getcwd()+'\\Tested audio\\electricdrill3_18.wav',
os.getcwd()+'\\Tested audio\\electricdrill3_20.wav']

electrical_collected = [os.getcwd()+'\\Tested audio\\Electrical drill_drilling_source_05_10.wav',
os.getcwd()+'\\Tested audio\\Electrical driller_hole making_receiver_01_7_1.wav']
###################################################
order_list = ['Original recording', 'App recording @8000sr', 'App recording @24000sr', 'App recording @48000sr', 'App recording @44100sr (Default)']

below10_01 = [os.getcwd()+'\\Tested audio\\Handheld less 10_wall removal_receiver_01_3.wav', os.getcwd()+'\\temp\\sound-20240430024346.wav', 
os.getcwd()+'\\temp\\sound-20240430025708.wav', os.getcwd()+'\\temp\\sound-20240430030910.wav', os.getcwd()+'\\iPhone app recording\\sound-20240423040302.wav']

below10_02 = [os.getcwd()+'\\Tested audio\\Handheld less 10_wall removal_receiver_01_6.wav', os.getcwd()+'\\temp\\sound-20240430024423.wav', 
os.getcwd()+'\\temp\\sound-20240430025809.wav', os.getcwd()+'\\temp\\sound-20240430031013.wav', os.getcwd()+'\\iPhone app recording\\sound-20240423041028.wav']

over10_01 = [os.getcwd()+'\\Tested audio\\Percussive over 10_wall removal_receiver_09_10.wav', os.getcwd()+'\\temp\\sound-20240430025216.wav', 
os.getcwd()+'\\temp\\sound-20240430030332.wav', os.getcwd()+'\\temp\\sound-20240430031533.wav', os.getcwd()+'\\iPhone app recording\\sound-20240429030409.wav']

over10_02 = [os.getcwd()+'\\Tested audio\\Percussive over 10_wall removal_source_13_10.wav', os.getcwd()+'\\temp\\sound-20240430025459.wav', 
os.getcwd()+'\\temp\\sound-20240430030629.wav', os.getcwd()+'\\temp\\sound-20240430031906.wav', os.getcwd()+'\\iPhone app recording\\sound-20240429030837.wav']

class plot:
    def __init__ (self, files, order_list):
        self.order_list = order_list
        self.meldb, self.bandwidth, self.zcr_ls = [], [], []
        for i in files:
            self.audio, self.sr = librosa.load(path = i, sr=22050)
            self.audio = librosa.effects.time_stretch(y = self.audio, rate=len(self.audio)/self.sr)
            self.meldb.append (librosa.power_to_db(S=librosa.feature.melspectrogram(y = self.audio, sr=22050,n_fft=2048, hop_length=231,norm=np.inf), ref=1.0))
            self.bandwidth.append (librosa.feature.spectral_bandwidth(y=self.audio, sr=self.sr).flatten().reshape (1, 44))
            self.zcr_ls.append (librosa.feature.zero_crossing_rate(self.audio).flatten().reshape (1, 44))              
    def mel_plot (self):
        fig, ax = plt.subplots (nrows = 1, ncols = len(self.meldb))
        for i in range (len(self.order_list)):
            img = librosa.display.specshow(data=self.meldb[i], sr=self.sr, x_axis='time', y_axis='mel', ax = ax[i])
            ax[i].set(title = self.order_list[i])
        plt.show()
    def bandwidth_plot (self):
        length_bin = np.arange (0, 44, dtype = int)
        for i in range (len(self.order_list)):
            plt.plot(length_bin.reshape (44,), self.bandwidth[i].reshape (44,), label = self.order_list[i])
            plt.ylim(0, 4000)
        plt.legend ()
        plt.show()
    def zcr_plot (self):
        length_bin = np.arange (0, 44, dtype = int)
        for i in range (len (self.order_list)):
            plt.plot(length_bin.reshape (44,), self.zcr_ls[i].reshape (44,), label = self.order_list[i])
            plt.ylim(0, 1.0)
        plt.legend()
        plt.show()   


def main():
    #folder_to_visit = [os.getcwd()+'\\iPhone built-in app recording', os.getcwd()+'\\Andriod built-in app recording']
    folder_to_visit = [os.getcwd()+'\\Andriod app recording', os.getcwd()+'\\iPhone app recording']
    df = compare_prediction (folder_to_visit)
    df.to_csv ('Audio compare_.csv')
    plot(files_01).scatter_plot()

if __name__ == "__main__":
    main ()


