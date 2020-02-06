import librosa.display
import librosa
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def extract_logmel(wav, sr, n_mels=128): #Output -> (timeframe, logmel_dim)
    audio, _ = librosa.load(wav, sr=sr)
    logmel = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)).T
    return logmel

# Load a flac file from 0(s) to 60(s) and resample to 4.41 KHz
for i in range(1, 11):
    filename = './example-data/not-song/{:03}.wav'.format(i)
    y, sr = librosa.load(filename)

    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%02.0f dB')
    plt.savefig('./result/not-song/{:03}.png'.format(i))

