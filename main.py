import librosa.display
import glob
import numpy as np
import time

def extract_logmel(wav, sr, n_mels=128): #Output -> (timeframe, logmel_dim)
    audio, _ = librosa.load(wav, sr=sr)
    logmel = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)).T
    return logmel


# Load a flac file from 0(s) to 60(s) and resample to 4.41 KHz
filename = './example-data/song/001.wav'
y, sr = librosa.load(filename, sr=48000, offset=0.0)
librosa.display.waveplot(y=y, sr=sr)

print("finish")
