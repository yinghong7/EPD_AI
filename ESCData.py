import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import helpers
import os

class ESC50Data(Dataset):
  def __init__(self, base, df, in_col, out_col, start_rand=0, stop_rand=100, wav_path=''):
    self.df = df
    self.data = []
    self.labels = []
    self.c2i={}
    self.i2c={}
    self.categories = sorted(df[out_col].unique())
    for i, category in enumerate(self.categories):
      self.c2i[category]=i
      self.i2c[i]=category
    for ind in tqdm(range(len(df))):
      row = df.iloc[ind]
      if row['rand'] >= start_rand & row['rand'] <= stop_rand: # only take the values that match our random selection
        file_path = os.path.join(wav_path, os.path.join(base,row[in_col]))
        self.data.append(helpers.spec_to_image(helpers.get_melspectrogram_db(file_path))[np.newaxis,...])
        self.labels.append(self.c2i[row['category']])
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]