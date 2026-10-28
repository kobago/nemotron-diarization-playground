# Nemotron 3 Diarization on an M1 MacBook Pro

[nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) を Transformers 5.21 + PyTorch (MPS / CPU) で動かし、速度・メモリと、Desert Ant Labs の [Who](https://huggingface.co/desert-ant-labs/who) との一致を測る。

```sh
uv sync
uv run bench.py data/nasa.wav --mode offline --device mps --compare data/nasa.who.json
uv run bench.py data/nasa.wav --mode low_latency --device mps --length 300
uv run test_bench.py   # agreement() の確認
```

- `--mode offline`: モデル内部の 30.4 秒チャンクでファイル全体を処理
- `--mode low_latency | very_low_latency | ultra_low_latency`: マイク入力のように 1 チャンクずつ流し、チャンクごとの計算時間を測る
- `--compare`: `da who --json` の出力。話者を重なりの大きい順に 1 対 1 で対応付け、フレーム単位で一致率を出す
- 結果は `out/` に JSON で残る (`out/results.jsonl` は下表の元データ)

## 結果 (2026-10-28, MacBook Pro M1, macOS 26.7)

音声: NASA「Houston We Have a Podcast」1 話 (60 分 8 秒、英語、主な話者 3 人)。16 kHz モノラル WAV。
Who の数値は同じファイルを `da who` (Core ML / Neural Engine) で処理したもの。

### 一括処理

| | 音声 | 処理時間 | 実時間比 | ピーク RSS | MPS 確保 | Who と同じ話者 | 別の話者 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nemotron MPS fp32 | 60 分 | 30.2 s | 119× | 3.5 GB | 1.4 GB | 94.8% | 0.6% |
| Nemotron MPS fp16 | 5 分 | 2.9 s | 104× | 1.1 GB | 0.24 GB | 90.1% | 3.4% |
| Nemotron CPU fp32 | 5 分 | 3.9 s | 78× | 1.2 GB | — | 90.1% | 3.4% |
| Who (Core ML) | 60 分 | 8.3–8.6 s | 約 420× | — | — | — | — |

一致率は「どちらかが発話と判定したフレーム」に対する割合。残りはほぼ Who だけが発話と判定した区間 (4.5–6.5%)。

### ストリーミング (5 分, 1 チャンクずつ)

| モード | 入力バッファ | チャンク間隔 | 計算 p50 / p95 / max | 実時間比 | ピーク RSS | MPS 確保 |
|---|---:|---:|---:|---:|---:|---:|
| low_latency MPS | 1.04 s | 720 ms | 135 / 168 / 194 ms | 5.3× | 0.78 GB | 0.50 GB |
| very_low_latency MPS | 0.64 s | 480 ms | 132 / 159 / 261 ms | 3.7× | 0.75 GB | 0.52 GB |
| ultra_low_latency MPS | 0.32 s | 240 ms | 131 / 156 / 239 ms | 1.9× | 0.87 GB | 0.52 GB |
| low_latency CPU | 1.04 s | 720 ms | 192 / 275 / 436 ms | 3.7× | 1.0 GB | — |
| low_latency MPS, 60 分 | 1.04 s | 720 ms | 137 / 161 / 598 ms | 5.1× | 0.99 GB | 0.67 GB |

- どのモードもチャンク間隔より計算が短く、M1 でリアルタイムに追従できる。
- 計算時間はチャンクの長さによらず約 130 ms で、PyTorch の 1 回あたりのオーバーヘッドが大半とみられる。
- 60 分のストリーミングでも話者番号は最後まで保たれ、Who との一致は一括処理と同じ 94.8%。

### 話者ごとの発話時間 (60 分)

| | 冒頭 | ホスト | ゲスト 1 | ゲスト 2 | その他 |
|---|---:|---:|---:|---:|---|
| Nemotron 一括 | 29 s | 1199 s | 1031 s | 1143 s | 14 s, 3 s, 3 s |
| Who | 39 s | 1283 s | 1117 s | 1231 s | 10 s |

## C++ 版 (NeMo-Speech.cpp, Metal, GGUF q8_0)

NVIDIA 公式の [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) (e680fc3, nemo-speech 0.2.0) を `vendor/` にビルドして実行。

```sh
brew install cmake ninja sentencepiece abseil
git clone --depth 1 https://github.com/NVIDIA/NeMo-Speech.cpp vendor/NeMo-Speech.cpp
cd vendor/NeMo-Speech.cpp && git submodule update --init --depth 1 llama.cpp
scripts/configure.sh metal-asr && cmake --build --preset metal-asr && cd ../..
N=vendor/NeMo-Speech.cpp/build/metal-asr/bin/nemo-speech
$N diarize data/nasa.wav --device metal --preset v3-offline --format json -o out/cpp.json
uv run compare.py out/cpp.json data/nasa.who.json
```

### 一括処理 (60 分, `/usr/bin/time -l`, 起動とモデル読み込みを含む)

| | 処理時間 | 実時間比 | ピークメモリ | Who と同じ話者 | 別の話者 |
|---|---:|---:|---:|---:|---:|
| Metal `--preset v3-offline` | 20.2 s | 178× | 1.3 GB | 96.3% | 0.6% |
| Metal 既定 (ストリーミング, 1.04 s チャンク) | 259 s | 14× | 1.3 GB | 96.6% | 0.7% |
| CPU 既定 (ストリーミング) | 630 s | 5.7× | 1.3 GB | 96.6% | 0.7% |

60 分のピークメモリはファイル全体を読み込む分を含む。5 分のファイルでは 180–200 MB だった。

### ストリーミング (5 分, Metal, `bench diarize`, ウォームアップ後)

| `--diar.chunk` / `--diar.right_context` | 入力バッファ | 処理時間 | 実時間比 | 1 チャンクあたり | ピークメモリ | Who と同じ話者 |
|---|---:|---:|---:|---:|---:|---:|
| 13 / 1 (既定) | 1.12 s | 18.3 s | 16× | 約 63 ms | 203 MB | 92.4% |
| 6 / 2 | 0.64 s | 38.3 s | 7.8× | 約 61 ms | 181 MB | 92.5% |
| 3 / 1 | 0.32 s | 76.1 s | 3.9× | 約 61 ms | 180 MB | 92.6% |

- 1 チャンクの計算は約 60 ms で、PyTorch (MPS) の約 130 ms の半分。0.32 s バッファでもリアルタイムの 3.9 倍の余裕がある。
- 新しい設定での初回実行は、Metal の準備で 20 秒ほど余計にかかった (`bench` のウォームアップがこれを吸収している)。

## Web アプリ (マイク → 話者付き文字起こし)

`metal-server` プリセットをビルドしてから起動する。`nemo-speech serve` (Nemotron 3.5 ASR + Nemotron 3 Diarization, Metal) を立ち上げ、ブラウザのマイク音声をその realtime WebSocket に流す。音声は端末の外に出ない。

```sh
cd vendor/NeMo-Speech.cpp && git submodule update --init --depth 1 third_party/cpp-httplib
scripts/configure.sh metal-server && cmake --build --preset metal-server && cd ../..
uv run webapp.py   # http://localhost:5174
```

- 日本語 / English を選べる。無音 (300 / 600 / 1000 ms) で発話が区切られるたびに、話者付きの 1 行が確定する。
- `serve` はエンドポインティングが既定で無効なので、`webapp.py` が `--asr.endpointing.enable=true` を付けている。無効のままだと、クライアントが commit するまで 1 行も確定しない。
- 日本語は単語の区切りがなく、ASR の「単語」が句読点間の塊になる。1 行の話者は、その行で一番長く話した話者にしている。

ブラウザと同じプロトコルで WAV を実時間で流した結果 (M1, Metal):

| 音声 | 区切り | 発話終了 → 確定 | 話者 | サーバー RSS |
|---|---:|---:|---|---:|
| NASA 2 分 (英語, 3 人) | 600 ms | 中央値 0.70 s, 最大 0.80 s | ホスト・クリップ・ゲストを区別 | 410–460 MB |
| `make_sample_ja.sh` 47 秒 (日本語, `say` の 3 声) | 300 ms | 0.3–0.9 s | 認識できた行はすべて正しい話者 | 約 445 MB |

- 4 倍速で流すと確定までの時間が伸び続けた (10 秒超)。実時間の数倍までしか余裕はない。
- 日本語の合成音声 (Rocko, Shelley) は発話ごと認識されないことがあった。「話者」は「和社」になった。

## 未検証

- 実際のマイク入力 (人の声) での認識精度と遅延
- 日本語音声、4 人以上の会話、正解ラベル付きデータでの DER
