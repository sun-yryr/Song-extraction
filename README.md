# Song-extraction

# 準備

```
git clone https://github.com/sun-yryr/Song-extraction.git
cd Song-extraction
pip install pipenv
pipenv install
```

# 動画を切る

**注意** 
カレントディレクトリの `temp` フォルダを利用し，利用後削除します．この名前なら消えても問題ないものしか入ってないと思いますが気をつけてください．

```
pipenv run cut "変換したいyoutubeのurl"

待つ

output dir path = 保存するパスを入れる
```

それか
```
pipenv run cut

input wave path = 切るwaveファイルのパスを入れる
output dir path = 保存するパスを入れる
```


# モデルを作る

あとで

# Docker版

## 開発

```bash
docker build -t song-extraction:latest -f docker/Dockerfile .
docker run --rm -it song-extraction:latest bash
```

## 実行

```bash
docker run --rm -it -v $(pwd):/app song-extraction:latest {youtube url}
```
