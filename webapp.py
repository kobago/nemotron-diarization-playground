"""Live speaker-labeled transcription from the microphone, all on this Mac.

  uv run webapp.py      # then open http://localhost:5174

Starts `nemo-speech serve` (Nemotron 3.5 ASR + Nemotron 3 Diarization on Metal) and serves
index.html, which streams the microphone to the server's realtime WebSocket. /stats reports the
server's memory.
"""

import atexit
import http.server
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
NEMO = ROOT / "vendor/NeMo-Speech.cpp/build/metal-server/bin/nemo-speech"
PORT, NEMO_PORT = 5174, 8080

if not NEMO.exists():
    sys.exit(f"{NEMO} がありません。README の「C++ 版」の手順で metal-server をビルドしてください。")

print("nemo-speech serve を起動中… (初回はモデルのダウンロードがあります)")
started = time.time()
server = subprocess.Popen([
    str(NEMO), "serve",
    "--asr-model", "nemotron-3.5", "--diar-model", "nemotron-3-diarization", "--device", "metal",
    "--port", str(NEMO_PORT), "--no-ui", "--cors-origin", f"http://localhost:{PORT}",
    # Off by default in serve: without it an utterance is only finalized when the client commits.
    "--asr.endpointing.enable=true",
])
atexit.register(server.terminate)
while True:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{NEMO_PORT}/health")
        break
    except OSError:
        if server.poll() is not None:
            sys.exit("nemo-speech serve が終了しました")
        time.sleep(0.5)
load_s = time.time() - started

peak_kb = 0


def stats():
    # ponytail: RSS sampled when the page polls (every second), so the peak can miss short spikes.
    global peak_kb
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(server.pid)], capture_output=True, text=True).stdout
    kb = int(out.strip() or 0)
    peak_kb = max(peak_kb, kb)
    return {"rssMB": kb / 1024, "peakMB": peak_kb / 1024, "loadS": load_s,
            "wsUrl": f"ws://127.0.0.1:{NEMO_PORT}/v1/audio/transcriptions/realtime"}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body, ctype = (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8"
        elif self.path == "/stats":
            body, ctype = json.dumps(stats()).encode(), "application/json"
        else:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


print(f"準備完了 ({load_s:.1f}s) → http://localhost:{PORT}")
try:
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
except KeyboardInterrupt:
    pass
