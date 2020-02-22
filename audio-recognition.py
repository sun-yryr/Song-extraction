import wave
import os
import numpy as np
import struct
from pathlib import Path
import requests

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

def main():
    fname = input()
    p = Path(fname).resolve()
    if not p.is_file():
        print("error")
        return
    w = WavEdit(str(p))
    time = w._time / 2
    w.cut_wav_output("./temp.wav", 10, time)
    bfile = open('./temp.wav', 'rb').read()
    files = {'file': ('./temp.wav', bfile, 'audio/wav')}
    url = 'https://api.audd.io/recognizeWithOffset'
    res = requests.get(url, files=files)
    print(res.status_code)
    print(res.content)
    print(res.text)


if __name__ == "__main__":
    main()