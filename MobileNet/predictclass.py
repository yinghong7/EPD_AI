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
model = tf.keras.applications.MobileNet(weights='imagenet')

# ***Set your image folder & class list here***
# imagefolder = os.getcwd() + '\\Testimage'
#'C:\\Users\\Simon.Leuk\\Documents\\02_EPDNoise\\Imagerecognition\\MobileNet\\Testimage'

class mobileNet ():
    def __init__ (self):
        self.folder = os.getcwd() + '\\Testimage'
        with open (self.folder) as f:
            self.classlist = f.readlines()
        self.allclass, self.output = [], []
        for lines in self.classlist:
            self.text = lines.rstrip("\n")
            self.text = self.text.strip("'")
            self.allclass.append (self.text)
    
    def get_features (self):
        for img in glob.glob(os.path.join(self.folder, '*')): #For each jpg in the imagefolder
            img = keras.utils.load_img(img, target_size=(224, 224))
            img = keras.utils.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            item = np.array(model.predict(img)) # item is probability of our image being in each of the group of 1000 items in this model
            whichmax = np.argmax(item)
            output.append([whichmax, item[0][whichmax], self.allclass[whichmax]]) #[predicted ID, probability, predicited class name]
        print("%s seconds elapsed" % (time.time() - start_time))
    
def main():
    mobileNet().prediction()

if __name__ == "__main__":
    main ()