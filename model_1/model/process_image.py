import time
start_time = time.time()
import h5py
import json
from tqdm import tqdm, trange  #library for recording time
import os
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import model_from_json
from keras.models import Sequential, Model
from datetime import datetime
import argparse

def main():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    loaded_model = tf.keras.models.load_model(os.getcwd + "/mask_model.h5") 
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', help='file_location')
    args = parser.parse_args()
    image = keras.utils.load_img(args.location, target_size=(224, 224))
    x = keras.utils.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    loaded_model.predict(x)
    print("%s seconds elapsed" % (time.time() - start_time))

if __name__ == "__main__":
    main ()



