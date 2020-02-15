import wave
import subprocess
import struct
import os
from pathlib import Path
from keras.models import load_model
import librosa
import numpy as np
from statistics import mean
import sys
import shutil

class WavEdit():
    def __init__(self, path):
        wr = wave.open(path, "r")
        self._ch = wr.getnchannels()
        self._width = wr.getsampwidth()
        self._fr = wr.getframerate()
        self._fn = wr.getnframes()
        self._time = 1.0 * self._fn / self._fr
        self._data = np.frombuffer(wr.readframes(wr.getnframes()), dtype=np.int16)
        self._params = wr.getparams()
        wr.close()
    
    def print_meta_data(self):
        print("Channel: ", self._ch)
        print("Sample width: ", self._width)
        print("Frame Rate: ", self._fr)
        print("Frame num: ", self._fn)
        print("Params: ", self._params)
        print("Total time: ", self._time)

    def cut_wav_output(self, outputPath, dur, start = 0):
        startFrame = int(self._ch * self._fr * start)
        endFrame = startFrame + int(self._ch * self._fr * dur) + 1
        if os.path.exists(outputPath):
            print("File already exists")
            return
        Y = self._data[startFrame:endFrame]
        outd = struct.pack("h" * len(Y), *Y)
        ww = wave.open(outputPath, "w")
        ww.setnchannels(self._ch)
        ww.setsampwidth(self._width)
        ww.setframerate(self._fr)
        ww.writeframes(outd)
        ww.close()
    
    def all_cut_10sec(self, outputDir):
        outputPath = Path(outputDir).resolve()
        if outputPath.is_file():
            print("this is not directory")
            return
        if not outputPath.is_dir():
            outputPath.mkdir(parents=True)
        startFrame = 0
        durFrame = int(self._ch * self._fr * 10) + 1
        endFrame = startFrame + durFrame
        for i in range(int(self._time/10)):
            p = outputPath / "{:04}.wav".format(i)
            Y = self._data[startFrame:endFrame]
            outd = struct.pack("h" * len(Y), *Y)
            ww = wave.open(str(p), "w")
            ww.setnchannels(self._ch)
            ww.setsampwidth(self._width)
            ww.setframerate(self._fr)
            ww.writeframes(outd)
            ww.close()
            startFrame = endFrame
            endFrame = startFrame + durFrame

# 指定したディレクトリにあるwavファイルのリストを返す
def get_wav_filelist(p):
    _list = p.glob("./*.wav")
    _list = list(_list)
    _list.sort()
    return _list

# メル周波数スペクトラムを返す
def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    return melsp

def format_wav_cnn(filename, aug=None, rates=None):
    _x, fs = librosa.load(filename, sr=48000, duration=9.9)
    _x = calculate_melsp(_x)
    _x = _x[None, ..., None]
    return _x

# 学習済みモデルを利用して，判定結果が入った配列を返す
def validate_song_cnn(fileList):
    freq          = 128
    time          = 3713
    song          = 0
    not_song      = 1
    model = load_model("model_0214.h5")
    songFlag = [0] * len(fileList)
    for i in range(len(fileList)):
        fileName = fileList[i]
        ret = model.predict(format_wav_cnn(fileName))
        # 該当部分がsongなら1を入れる
        songFlag[i] = ret[0][0]
        print("{} => {} [ {:.2%}, {:.2%} ]".format(fileList[i], songFlag[i], ret[0][0], ret[0][1]))
    return songFlag

# 判定結果が入った配列を利用して，曲の開始・終了が入った配列を返す
# FIX いい感じの抽出アルゴリズムを書く
def get_songs(songFlag):
    flag = 0
    songStart = 0
    res = []
    for i, s in enumerate(songFlag):
        if s>0.5 and flag==0:
            songStart = i
            flag += 1
        elif s>0.5:
            flag += 1
        elif s<=0.5 and flag>=2:
            if hasProbabilityAverageSong(songFlag[songStart:i]):
                res.append((songStart, i))
            flag = 0
        else:
            flag = 0
    if flag >= 2:
        if hasProbabilityAverageSong(songFlag[songStart:]):
            res.append((songStart, len(songFlag)))
    
    # 隣り合う res の間が，2区間以下だったら結合
    # ただし，結合後に区間が36（== 6分）を超える場合，結合しない
    i = 0
    while i < len(res)-1:
        first = res[i]
        second = res[i+1]
        if (second[0] - first[1]) <= 2:
            # 2区間以下
            if (second[1] - first[0]) < 36:
                # 結合
                del res[i+1]
                res[i] = (first[0], second[1])
                i -= 1
        i += 1
    # 最後に，区間の前後を見て，確率が40%を超えていたら1区間追加する
    for i, s in enumerate(res):
        # 前
        tmp = s
        if songFlag[s[0]-1] > 0.4:
            tmp = (s[0]-1, tmp[1])
        if len(songFlag)-1 > s[1]:
            if songFlag[s[1]] > 0.4:
                tmp = (tmp[0], s[1]+1)
        res[i] = tmp
    print(res)
    return res

def hasProbabilityAverageSong(songProbabilitys):
    ave = mean(songProbabilitys)
    if ave >= 0.8:
        return True
    else:
        return False
    

def outputSongList(songlist, outputDir):
    p = outputDir / "songList.csv"
    s = "fileName, start, end\n"
    for i, song in enumerate(songlist):
        s += "{:02}.wav,{},{}\n".format(i, song[0]*10, song[1]*10)
    with open(p, "w") as f:
        f.write(s)



def main():
    workingDirectory = Path("./temp").resolve()
    workingDirectory.mkdir(parents=True, exist_ok=True)
    args = sys.argv
    if len(args) == 2:
        # youtube URL からいく
        cmd = "youtube-dl -x --audio-format wav {} -o './temp/%(id)s.%(ext)s'".format(args[1])
        subprocess.run(cmd, shell=True)
        p = list(Path("./temp").glob("./*.wav"))
        path = str(p[0].resolve())
        w = WavEdit(path)
        Path(path).unlink()
    else:
        print("intput wave path = ", end="")
        path = input()
        w = WavEdit(path)
    print("output dir path = ", end="")
    outputdir = Path(input()).resolve()
    # w.print_meta_data()
    print("wav cutting... ", end="")
    w.all_cut_10sec(str(workingDirectory))
    print("finish")
    fileList = get_wav_filelist(workingDirectory)
    print("recognition... ", end="")
    songFlag = validate_song_cnn(fileList)
    print("finish")
    # songFlag = [1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,0 ,0 ,1 ,1 ,1]
    seSongList = get_songs(songFlag)
    outputdir.mkdir(parents=True)
    outputSongList(seSongList, outputdir)
    print("output song... ", end="")
    for i in range(len(seSongList)):
        outputPath = outputdir / '{:02}.wav'.format(i)
        start = seSongList[i][0] * 10
        length = seSongList[i][1] * 10 - start
        # 3秒ずつ前後に伸ばす
        start -= 3
        length += 6
        w.cut_wav_output(str(outputPath), length, start)
    print("finish")
    shutil.rmtree(workingDirectory)


if __name__ == "__main__":
    main()