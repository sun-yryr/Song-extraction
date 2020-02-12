import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import numpy as np
import os
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import model_selection
from sklearn import preprocessing
import IPython.display as ipd
from pathlib import Path
import random as rd

freq          = 128
time          = 3713
song          = 0
not_song      = 1

def wavfileList(dirname = "."):
    _list = Path(dirname).glob("./*.wav")
    return list(_list)

def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    return melsp

def formatData(filename, aug=None, rates=None):
    _x, fs = librosa.load(filename, sr=48000, duration=9.9)
    _x = calculate_melsp(_x)
    _x = _x[None, ..., None]
    return _x

# 90%超えてたら判定する
def checkSong(ret):
    if ret[0][0] > 0.85:
        return "song"
    else:
        return "not-song"

songFileList = wavfileList("/Users/sun-mm/Desktop/v/toko")
songFileList.sort()
model = load_model("model_second_cnn.h5")

for filename in songFileList:
    ret = model.predict(formatData(filename))
    print("{} => {} [ {:.2%}, {:.2%} ]".format(filename, checkSong(ret), ret[0][0], ret[0][1]))

