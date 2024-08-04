# Solution 1 - cens svc model if predicted as < 10 kg, take cens out for cnn
# Solution 2 - include tested audios

import numpy as np
import librosa
import os, glob
import pickle


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
    return mels_out,zcritem,centitem, cens

def main():
    mel_list_, zcr_list_, cent_list_, cens_list, label = [], [], [], [], []
    folderpath = os.getcwd() + '/0108_EPD testing'
    for wav_file in glob.glob(os.path.join(folderpath, '*')):
        name = wav_file.split('/')[-1].split('_')[0]
        mel, zcr, cent, cens = preprocess (wav_file)
    
        mel_list_.append (mel) #Important 
        zcr_list_.append (zcr) #Important 
        cent_list_.append (cent) #Important 
        cens_list.append (cens)
        label.append (name)
    
    with open(os.getcwd()+'/newfeature/mel_list_0108rec.pkl', 'wb') as file: 
        pickle.dump(mel_list_, file) 
    with open(os.getcwd()+'/newfeature/zcr_list_0108rec.pkl', 'wb') as file: 
        pickle.dump(zcr_list_, file) 
    with open(os.getcwd()+'/newfeature/cent_list_0108rec.pkl', 'wb') as file: 
        pickle.dump(cent_list_, file) 
    with open(os.getcwd()+'/newfeature/cens_list_0108rec.pkl', 'wb') as file: 
        pickle.dump(cens_list, file) 
    with open(os.getcwd()+'/newfeature/label_list_0108rec.pkl', 'wb') as file: 
        pickle.dump(label, file) 


if __name__ == "__main__":
    main()