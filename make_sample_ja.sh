#!/bin/sh
# A three-voice Japanese conversation from macOS `say`, as 16 kHz mono WAV: data/ja3.wav
# Speakers in order: Kyoko, Rocko, Shelley (the turns below cycle through them).
set -eu
out=data/ja3.wav
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

i=0
while IFS='|' read -r voice text; do
  i=$((i + 1))
  say -v "$voice" -o "$tmp/$i.aiff" "$text"
  ffmpeg -v error -i "$tmp/$i.aiff" -af apad=pad_dur=0.6 -ac 1 -ar 16000 "$tmp/$i.wav"
  echo "file '$tmp/$i.wav'" >> "$tmp/list.txt"
done <<'EOF'
Kyoko|おはようございます。今日は話者分離モデルの検証結果を共有します。
Rocko (日本語（日本）)|お願いします。マックブックでどのくらいの速さで動きましたか。
Shelley (日本語（日本）)|一時間の音声を二十秒ほどで処理できました。メモリは一ギガバイト程度です。
Kyoko|ストリーミングでも、話者の番号は最後まで変わりませんでした。
Rocko (日本語（日本）)|それはいいですね。日本語の文字起こしも同時にできるんですか。
Shelley (日本語（日本）)|はい、ネモトロンの音声認識モデルが日本語に対応しています。
Kyoko|では、次はウェブアプリで実際に試してみましょう。
Rocko (日本語（日本）)|賛成です。会議の議事録にも使えそうですね。
EOF

ffmpeg -v error -y -f concat -safe 0 -i "$tmp/list.txt" "$out"
echo "$out: $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out")s"
