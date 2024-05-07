import time
start_time = time.time()
import os
import glob
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing import image
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.applications.mobilenet import preprocess_input, decode_predictions
import pandas as pd
import argparse

model = tf.keras.applications.MobileNet(weights='imagenet')

# ***Set your image folder & class list here***
# imagefolder = os.getcwd() + '\\Testimage'
#'C:\\Users\\Simon.Leuk\\Documents\\02_EPDNoise\\Imagerecognition\\MobileNet\\Testimage'

class mobileNet ():
    def __init__ (self):
        self.folder = os.getcwd()
        with open (self.folder + '\\allclasslist.txt') as f:
            self.classlist = f.readlines()
        self.allclass, self.output = [], []
        for lines in self.classlist:
            self.text = lines.rstrip("\n")
            self.text = self.text.strip("'")
            self.allclass.append (self.text)
    
    def get_features (self, location):
        parser = argparse.ArgumentParser()
        parser.add_argument('--location', help='file_location')
        args = parser.parse_args()
        image = keras.utils.load_img(location, target_size=(224, 224))
        img = keras.utils.img_to_array(image)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        item = np.array(model.predict(img)) # item is probability of our image being in each of the group of 1000 items in this model
        whichmax = np.argmax(item)
        print("%s seconds elapsed" % (time.time() - start_time))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', help='file_location')
    args = parser.parse_args()
    mobileNet().get_features(args.location)

if __name__ == "__main__":
    main ()
