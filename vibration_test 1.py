import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import librosa 
import keras
from sklearn.preprocessing import Normalizer
import os
from sklearn.externals import joblib
from tensorflow import set_random_seed
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

filename = '/Users/yc/Downloads/Vibration_raw.xlsx'
xls = pd.ExcelFile(filename)
folder = ['1726_CH1', '1726_CH2', '1726_CH3', '2049_CH1', '2049_CH2', '2049_CH3', '2114_CH1', '2114_CH2', '2114_CH3', '2145_CH1', '2145_CH2', '2145_CH3', '2155_CH1', '2155_CH2', '2155_CH3']  
trainfolder = ['1726_CH1', '1726_CH2', '1726_CH3', '2049_CH1', '2049_CH2', '2049_CH3', '2114_CH1', '2114_CH2', '2114_CH3']  
testfolder = ['2145_CH1', '2145_CH2', '2145_CH3', '2155_CH1', '2155_CH2', '2155_CH3']
fs = 10


def read_data (filename, foldername):
  new_df = pd.DataFrame ()  
  for i in foldername:
    df_x = pd.read_excel (filename, i)
    df_x['source'] = i
    new_df = pd.concat ([new_df, df_x])
  return new_df 


def plot (filename, folder): 
  df1 = pd.read_excel (filename, folder) 
  plt.figure(figsize=(6.5,4))
  plt.plot (df1['Time'], df1['Acceleration'])
  plt.ylim (0, 0.02)
  plt.xlabel ('Time (s)')
  plt.ylabel ('Acceleration (m/s^2)')
  plt.show()



def power_spectrum_plot (data, sample_rate, FIG_SIZE):
  fft = np.fft.fft (data)
  f, t, Sxx = scipy.signal.spectrogram(data, sample_rate, scaling = 'spectrum')
  left_spectrum = Sxx[:int(len(Sxx)/2)]
  left_f = f[:int(len(Sxx)/2)]
  plt.figure (figsize = FIG_SIZE)
  plt.plot (left_f, left_spectrum, alpha = 0.4)
  plt.xlabel ('Frequency')
  plt.ylabel ('Magnitude')
  plt.title ('Power Spectrogram')
  plt.show ()



def stft_plot (data, sample_rate, FIG_SIZE, nperseg):
  spectrogram = np.abs (Zxx)
  plt.figure (figsize = (8,4))
  librosa.display.specshow(spectrogram, sr=10, hop_length = nperseg - nperseg//2)
  plt.xlabel("Time")
  plt.ylabel ('Frequency')
  plt.colorbar()
  plt.title("Spectrogram")
  plt.show()
  log_spectrogram = librosa.amplitude_to_db(spectrogram)
  plt.figure(figsize = (8,4))
  librosa.display.specshow(log_spectrogram, sr=10, hop_length = nperseg - nperseg//2)
  plt.xlabel("Time")
  plt.ylabel ('Frequency')
  plt.colorbar(format="%+2.0f dB")
  plt.title ("Spectrogram (dB)")
  plt.show ()


def stft (data, sample_rate, nperseg):
	f, t, spectrogram = scipy.signal.stft(data, sample_rate, nperseg = nperseg)
	log_spectrogram = librosa.amplitude_to_db(spectrogram)
	return spectrogram, log_spectrogram


def power_spectrum (data, sample_rate):
  fft = np.fft.fft (data)
  f, t, Sxx = scipy.signal.spectrogram(data, sample_rate, scaling = 'spectrum')
  return fft, Sxx 


def cwt (data, width):
  cwtmatr = scipy.signal.cwt(data, scipy.signal.ricker, width)
  return cwtmatr
 

vib_data = read_data(xls, folder)
vib_train = read_data(xls, trainfolder)
vib_test = read_data(xls, testfolder)

#Train and test features
fft_train, spectrum_train = power_spectrum(vib_train['Acceleration'].to_numpy(), 10)
fft_test, spectrum_test = power_spectrum(vib_test['Acceleration'].to_numpy(), 10)

stft_train, log_stft_train = stft(vib_train['Acceleration'].to_numpy(), 10, 300)
stft_test, log_stft_test = stft(vib_test['Acceleration'].to_numpy(), 10, 300)

cwt_train = cwt(vib_train['Acceleration'].to_numpy(), np.arange (3,31))
cwt_test = cwt(vib_test['Acceleration'].to_numpy(), np.arange (3,31))

#Pre-processing
scaler = Normalizer()
fft_train = Normalizer.fit_transform (fft_train)
fft_test = Normalizer.transform (fft_test)

spectrum_train = Normalizer.fit_transform (spectrum_train)
spectrum_test = Normalizer.transform (spectrum_test)

stft_train = Normalizer.fit_transform (stft_train)
stft_test = Normalizer.transform (stft_test)

log_stft_train = Normalizer.fit_transform (log_stft_train)
log_stft_test = Normalizer.transform (log_stft_test)

cwt_train = Normalizer.fit_transform (cwt_train)
cwt_test = Normalizer.transform (cwt_test)

#Select input features

Y_train = vib_train ['Train_arrival'].to_numpy()
Y_test = vib_test ['Train_arrival'].to_numpy()

# reshape inputs for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
print("Training data shape:", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print("Test data shape:", X_test.shape)


# define the autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(1))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model


def 1d_model (X):
    model = Sequential ()
    model.add (SimpleRNN(64, input_shape = (X.shape[1], X.shape[2]))
    model.add (Dense(1))
    return model



model1 = autoencoder_model(X_train)
model1.compile(optimizer='adam', loss='mae')
model1.summary()
model1.fit (X_train, Y_train, epochs=100, batch_size=1, verbose=2)
Y_pred_1 = model1.predict (X_test)

print (confusion_matrix(Y_test, Y_pred_1))
print (precision_score(Y_test, Y_pred_1), recall_score(Y_test, Y_pred_1), f1_score(Y_test, Y_pred_1))

model2 = 1d_model (X_train)
model2.compile (optimizer='adam', loss='mae')
model2.summary ()
model2.fit (X_train, Y_train, epochs=100, batch_size=1, verbose=2)
Y_pred_2 = model2.predict (X_test)

print (confusion_matrix(Y_test, Y_pred_2))
print (precision_score(Y_test, Y_pred_2), recall_score(Y_test, Y_pred_2), f1_score(Y_test, Y_pred_2))
