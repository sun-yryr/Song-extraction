from pathlib import Path

dirname = input()
flist = list(Path(dirname).glob("./*.wav"))
for i in range(len(flist)):
    flist[i].rename("{}{:03}.wav".format(dirname, i))