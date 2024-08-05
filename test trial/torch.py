cd C:/Users/Ying.Hong/Arup/EPD AI Noise - General/Backup/Sound-20240314T024736Z-001

python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader, Dataset
from pydub import AudioSegment
import librosa
import os, glob
import numpy as np
from sklearn.preprocessing import LabelBinarizer

AudioSegment.converter = os.getcwd()+ "\\ffmpeg.exe"
AudioSegment.ffprobe = os.getcwd()+ "\\ffprobe.exe"

class sound_classification (nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d (6, 16, 3) # no. of channel tbc, mels_db.reshape ((6,128,16))
    self.max1 = nn.MaxPool2d (2)
    self.conv2 = nn.Conv2d (16, 32, 3)
    self.max2 = nn.MaxPool2d (2)
    self.flat1 = nn.Flatten ()
    # Take in ZCR
    self.fc1 = nn.Linear (44, 32)
    self.fc2 = nn.Linear (32, 16)
    # Take in Cent
    self.fc3 = nn.Linear (44, 32)
    self.fc4 = nn.Linear (32, 16)
    # Full
    self.fc5 = nn.Linear (1920+16+16, 32)
    self.fc6 = nn.Linear (32, 4)
  def forward (self, x, zcr, cent):
    x = F.relu(self.conv1(x))
    x = self.max1 (x)
    x = F.relu(self.conv2(x))
    x = self.max2 (x)
    x = self.flat1(x)
    zcr = F.relu (self.fc1(zcr))
    zcr = F.relu (self.fc2(zcr))
    cent = F.relu (self.fc3(cent))
    cent = F.relu (self.fc4(cent))
    full = torch.cat ((x, zcr, cent), 1)
    full = F.relu (self.fc5(full))
    full = F.softmax (self.fc6(full))
    return full
  def predict (self, x, zcr, cent):
    l = torch.max(self.forward(self, x, zcr, cent), 1)[1]
    return l

class DatasetWrapper(Dataset):
    def __init__(self, X, ZCR, CEN, y=None):
        self.X, self.ZCR, self.CEN, self.y = X, ZCR, CEN, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx],self.ZCR[idx], self.CEN[idx]
        else:
            return self.X[idx],self.ZCR[idx], self.CEN[idx], self.y[idx]

def train (net, x, zcr, cent, y, lossfunc, lr = 0.0005, batch_size=100, nepochs=50):
    device = next(net.parameters()).device # check what device the net parameters are on
    optimizer = optim.Adam(net.parameters(), lr=lr)
    dataloader = DataLoader(DatasetWrapper(x, zcr, cent, y), batch_size=batch_size, shuffle=True)
    loop = tqdm(range(nepochs), ncols=110)
    for i in loop:
        t0 = time ()
        epoch_loss = 0
        n_batches = 0
        for (x_batch, zcr_batch, cent_batch, y_batch) in dataloader: # for each mini-batch
            optimizer.zero_grad()
            loss = lossfunc (net(x_batch, zcr_batch, cent_batch), y_batch.float())
            loss = loss + torch.norm(net.conv1.weight, p=2) + torch.norm(net.conv2.weight, p=2) + torch.norm(net.fc1.weight, p=2) + torch.norm(net.fc2.weight, p=2) + torch.norm(net.fc3.weight, p=2) + torch.norm(net.fc4.weight, p=2) + torch.norm(net.fc5.weight, p=2) + torch.norm(net.fc6.weight, p=2)
            loss.backward()
            optimizer.step()
        epoch_loss += loss
        n_batches += 1
        epoch_loss /= n_batches
        # evaluate network performance
        acc = test(net, x, zcr, cent, y, batch_size=batch_size)
        # show training progress
        loop.set_postfix(loss="%5.5f" % (epoch_loss), train_acc="%.2f%%" % (100*acc))
        return acc

def test(net, x, zcr, cent, y, batch_size=100):
    with torch.no_grad(): # disable automatic gradient computation for efficiency
        device = next(net.parameters()).device
        pred_cls = []
        # make predictions on mini-batches
        dataloader = DataLoader(DatasetWrapper(x, zcr, cent), batch_size=batch_size, shuffle=False)
        for (x_batch, zcr_batch, cent_batch) in dataloader:
            x_batch = x_batch.to(device)
            pred_cls.append (net.predict (x_batch, zcr_batch, cent_batch))
        # compute accuracy
        pred_cls = torch.cat(pred_cls) # concat predictions on the mini-batches
        true_cls = torch.max(y, 1)[1].cpu()
        acc = (pred_cls == true_cls).sum().float() / len(y)
        return acc

# Feature extraction
def preprocess(input_file_path):
    audio, sr = librosa.load(path = input_file_path, sr=22050)
    audio = librosa.effects.time_stretch(y=audio, rate=len(audio)/sr)
    zcr=librosa.feature.zero_crossing_rate(audio)
    zcritem=zcr.flatten()
    zcritem=zcritem.reshape(44) #Important 
    cent = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    centitem=cent.flatten() 
    centitem=cent.reshape(44) #Important 
    mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=231,norm=np.inf)
    mels_db = librosa.power_to_db(S=mels, ref=1.0)
    mels_out=mels_db.reshape((6,128,16)) #Important 
    return mels_out, zcritem, centitem

folderpath = os.getcwd()+'\\test'
mel_list, zcr_list, cent_list, label_list = [], [], [], []
for wav_filename in glob.glob(os.path.join(folderpath, '*')):
    mel, zcr, cen = preprocess (wav_filename)
    label = wav_filename.split('\\')[-1].split('_')[0]
    label_list.append (label)
    mel_list.append (torch.from_numpy(mel).float()) #Important 
    zcr_list.append (torch.from_numpy(zcr).float()) #Important 
    cent_list.append (torch.from_numpy(cen).float()) #Important 

lb = LabelBinarizer()
labelz=lb.fit_transform(label_list)
labelz = torch.from_numpy(labelz).float()) #Important 


train (sound_classification(), mel_list, zcr_list, cent_list, labelz, 
lossfunc = nn.CrossEntropyLoss(), lr = 0.0005)