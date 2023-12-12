import helpers
import os
import pandas as pd
import ESCData
from torch.utils.data import DataLoader

# process wav to image spectographs


csv_path = os.path.join(os.getcwd(),'esc50_ly.csv')
wav_path = os.path.join(os.path.join(os.getcwd(),'..'), 'ESC-50-master\\')

# load up from CSV, use rand to choose % for each
#filename	fold	target	category	esc10	src_file	take	rand
train = pd.read_csv(csv_path) #load dataframe from csv
valid = pd.read_csv(csv_path)
rand = 10 # take a random selection of 10% and use it for our validity test

# setup training data
train_data = ESCData.ESC50Data('audio', train, 'filename', 'category', rand, 100, wav_path)

# setup data to test validity
valid_data = ESCData.ESC50Data('audio', valid, 'filename', 'category', 0, rand, wav_path)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=True)

# create device and model
device = helpers.get_device()
model = helpers.build_model(device)

# train model
trained = helpers.train_model(model, device, train_loader, valid_loader)

helpers.save_model(trained, 'model.pth')

#model = helpers.load_model()

#print('loaded model')
