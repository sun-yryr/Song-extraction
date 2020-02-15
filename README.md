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

# モデルを作る

あとで