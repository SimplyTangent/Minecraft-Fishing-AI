#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:41:56 2020

@author: shashankkota
"""
import numpy as np
import os
import pandas as pd

import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.models import load_model, save_model
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras import backend as K
import pyscreenshot
from threading import Timer
from pynput.keyboard import Key, Controller
import pyautogui
import time
def create_feature_tensors(folder_path):
    feature_folder = sorted(os.listdir(folder_path))
    tensors = []
    for img_path in feature_folder:
        img = cv2.imread(folder_path+'/'+img_path)
        img = cv2.resize(img, (300, 300))
        tensors.append(image.img_to_array(img))
    return np.array(tensors)


def preprocess_features(features):
    features = features.astype('float32') / 255
    return features

def preprocess_labels(labels):
    labels = np_utils.to_categorical(labels, 2)
    return labels

def create_data(features, labels):

    
    reindexed_features = []
    reindexed_labels = []
    random_permutation = pd.np.random.permutation(len(features))
    
    for rand in random_permutation:
        reindexed_features.append(features[rand])
        reindexed_labels.append(labels[rand])
    
    features = np.array(reindexed_features)
    features = preprocess_features(features)
    
    labels = np.array(reindexed_labels)
    labels = preprocess_labels(labels)
    X_train = features[round(0.2 * len(features)):]
    X_test = features[:round(0.2 * len(features))]

    y_train = labels[round(0.2 * len(labels)):]
    y_test = labels[:round(0.2 * len(labels))]
    
    return ((X_train, y_train), (X_test, y_test))
 

def train_model(X_train, y_train, X_test, y_test,
                batch_size=10, num_epochs=6,
                learning_rate=0.00001,regularization=0,
                verbose=1):
    
    model = Sequential()
    model.add(Conv2D(10, (3, 3), activation='relu', padding='same', input_shape=(300,300,3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(20, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    model.fit(X_train, y_train, 
         batch_size=batch_size, epochs=num_epochs, verbose=verbose)
    
    
    return model
def nothing():
    print("ended")

def predict_screenshots(x, y, model):
    keyboard = Controller()
    screenshot = pyscreenshot.grab(bbox=(x, y, x + 600, y + 600))
    
    frame = np.array(screenshot)
    frame = cv2.resize(frame, (300, 300))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255
    
    prediction = model.predict(np.array([frame]))
    if(prediction[0][1] >= 0.47):
        print("pressed")
        pyautogui.click(button='right') 
        plt.imshow(frame)
        plt.show()
        print(prediction)
        time.sleep(2.0)
        pyautogui.click(button='right') 
        time.sleep(1.8)
    print(prediction)
    time.sleep(0.02)
    predict_screenshots(x, y, model)
    


features = create_feature_tensors('/Users/shashankkota/Desktop/Minecraft Fishing AI/minecraft_model/fishing_features')
labels = pd.read_csv('/Users/shashankkota/Desktop/Minecraft Fishing AI/minecraft_model/fishing_labels.csv').to_numpy()
(X_train, y_train), (X_test, y_test) = create_data(features, labels)

try:
    model = load_model('fishing_model.h5')
except:
    model = train_model(X_train, y_train,
                        X_test, y_test,
                        batch_size=4,
                        num_epochs=10,
                        learning_rate=0.000015)

    save_model(model, 'fishing_model.h5')
    

predict_screenshots(1100, 500, model)
 


