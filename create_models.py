import os
import random
import numpy as np
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
    _list = Path().glob("./{}/*.wav".format(dirname))
    return list(_list)

def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    return melsp

def save_np_data(filename, x, aug=None, rates=None):
    np_data = np.zeros(freq*time*len(x)).reshape(len(x), freq, time)
    np_targets = np.zeros(len(x))
    for i in range(len(x)):
        _x, fs = librosa.load(x[i], sr=48000, duration=9.9)
        if aug is not None:
            _x = aug(x=_x, rate=rates[i])
        _x = calculate_melsp(_x)
        np_data[i] = _x
        np_targets[i] = target(x[i])
    np.savez(filename, x=np_data, y=np_targets)

def target(path):
    if path.parent.name == 'song':
        return song
    else:
        return not_song

# フォルダからファイル名を抽出，ランダムに並べ替えます
songFileList = wavfileList("example-data/song")
notSongFileList = wavfileList("example-data/not-song")
rd.shuffle(songFileList)
rd.shuffle(notSongFileList)
# 学習データとテストデータを 9:1 で分ける
checkIndex = int(len(songFileList) * 0.9)
learn = songFileList[:checkIndex] + notSongFileList[:checkIndex]
test = songFileList[checkIndex:] + notSongFileList[checkIndex:]
# ランダムに並べ直す
rd.shuffle(learn)
rd.shuffle(test)
# 出力
save_np_data("./v-melsp-learn.npz", learn)
save_np_data("./v-melsp-test.npz", test)