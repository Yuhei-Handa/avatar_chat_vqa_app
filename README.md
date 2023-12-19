# avatar_chat_vqa_app

これはStreamlitでGeminiを使用した音声対話＆VQAアプリです。

![ff9e2872-d7ba-065b-59a4-3ef7500767a9.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3523467/f38adc06-3426-9057-6344-09d1722f486a.png)


使用したい方以下の手順に従ってください。

1. クローン後、以下のリンクからVOICEBOXをダウンロード、もしくは自身の環境に対応したパッケージを同じディレクトリにダウンロードし、起動することでローカル上にサーバーがを立ち上げます。

https://voicevox.hiroshiba.jp/

https://github.com/VOICEVOX/voicevox_engine/releases/tag/0.14.2

例：

```
#windows/CUDA版
cd windows-nvidia
run.exe --use_gpu
```

2. 以下のコマンドでapp.pyを実行します。

```python:python
streamlit run app.py
```

音声："VOICEVOX:ナースロボ＿タイプＴ"

立ち絵:【MYCOEIROINK】ナースロボ_タイプT + 立ち絵PSD
\nhttps://booth.pm/ja/items/3876977

