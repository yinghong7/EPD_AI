import numpy as np
import librosa
from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm

def setlr(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return optimizer

def spec_to_image(spec, eps=1e-6):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
  return spec_scaled

def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
  wav,sr = librosa.load(file_path,sr=sr)
  if wav.shape[0]<5*sr:
    wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
  else:
    wav=wav[:5*sr]
  spec=librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
  spec_db=librosa.power_to_db(spec,top_db=top_db)
  return spec_db

def get_device():
    if torch.cuda.is_available():
        device=torch.device('cuda:0')
        print("Device = GPU")
    else:
        device=torch.device('cpu')
        print("Device = CPU")
    return device
   
def build_model(device):
    resnet_model = resnet34(weights=resnet34.ResNet34_Weights.DEFAULT)
    resnet_model.fc = nn.Linear(512,50)
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet_model = resnet_model.to(device)
    return resnet_model

def train_model(model, device, train_loader, valid_loader):
    learning_rate = 2e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 50
    loss_fn = nn.CrossEntropyLoss()
    train_losses=[]
    valid_losses=[]
    def lr_decay(optimizer, epoch):
        if epoch%10==0:
            new_lr = learning_rate / (10**(epoch//10))
            optimizer = setlr(optimizer, new_lr)
            print(f'Changed learning rate to {new_lr}')
        return optimizer
    def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, change_lr=None):
        for epoch in tqdm(range(1,epochs+1)):
            model.train()
            batch_losses=[]
            if change_lr:
                optimizer = change_lr(optimizer, epoch)
            for i, data in enumerate(train_loader):
                x, y = data
                optimizer.zero_grad()
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.long)
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                loss.backward()
                batch_losses.append(loss.item())
                optimizer.step()
            train_losses.append(batch_losses)
            print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')
            model.eval()
            batch_losses=[]
            trace_y = []
            trace_yhat = []
            for i, data in enumerate(valid_loader):
                x, y = data
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.long)
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                trace_y.append(y.cpu().detach().numpy())
                trace_yhat.append(y_hat.cpu().detach().numpy())      
                batch_losses.append(loss.item())
            valid_losses.append(batch_losses)
            trace_y = np.concatenate(trace_y)
            trace_yhat = np.concatenate(trace_yhat)
            accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)
            print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-Accuracy : {accuracy}')
    train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, lr_decay)
    return model

def save_model(model, path="model.pth"):
   torch.save(model.state_dict, path)
   
def load_model(path="model.pth"):
    model = build_model(get_device())
    model.load_state_dict(torch.load(path))
    return model