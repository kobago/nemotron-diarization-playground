"""Agreement of a `nemo-speech diarize --format json` output with another diarization.

  uv run compare.py out/cpp.device_metal.json data/nasa.who.json
"""

import json
import sys
from collections import Counter

from bench import agreement

ours = json.load(open(sys.argv[1]))["segments"]
# Only the stretch both cover: `ours` may come from a clip of the file `theirs` describes.
length = max(s["end"] for s in ours)
theirs = [t for t in json.load(open(sys.argv[2]))["turns"] if t["start"] < length]
talk = Counter()
for s in ours:
    talk[s["speaker"]] += s["end"] - s["start"]
print(json.dumps({
    "speaker_seconds": {k: round(v) for k, v in sorted(talk.items())},
    "agreement": agreement([{"Start": s["start"], "End": s["end"], "Speaker": s["speaker"]} for s in ours], theirs, length),
}))
