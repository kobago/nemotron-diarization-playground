"""Nemotron 3 Diarization on this Mac: speed, memory, and agreement with another diarization.

  uv run bench.py data/nasa.wav --mode offline --device mps
  uv run bench.py data/nasa.wav --mode low_latency --device mps --start 0 --length 300
  uv run bench.py data/nasa.wav --mode offline --compare data/nasa.who.json

offline runs the model's own 30.4s chunks over the whole input; the streaming modes feed the audio one
chunk at a time as a live microphone would, and time each chunk.
"""

import argparse
import json
import resource
import time
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

MODEL_ID = "nvidia/Nemotron-3-Diarization"
RATE = 16000


def load_wav(path, start, length):
    with wave.open(str(path)) as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, "need 16 kHz mono 16-bit WAV"
        w.setpos(int(start * RATE))
        frames = w.readframes(int(length * RATE) if length else w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def run_offline(model, processor, audio, device, dtype):
    inputs = processor(audio, sampling_rate=RATE).to(device, dtype=dtype)
    with torch.inference_mode():
        logits = model(**inputs).logits
    sync(device)
    return logits, []


def run_streaming(model, processor, audio, device, dtype):
    """The model card's streaming loop, timing each chunk's forward."""
    def chunks():
        yield processor(audio[: processor.num_samples_first_audio_chunk], sampling_rate=RATE,
                        is_streaming=True, is_first_audio_chunk=True)
        mel_frame_idx = processor.num_mel_frames_per_step
        start = processor.audio_chunk_start(mel_frame_idx)
        while (end := start + processor.num_samples_per_audio_chunk) <= len(audio):
            yield processor(audio[start:end], sampling_rate=RATE, is_streaming=True, is_first_audio_chunk=False)
            mel_frame_idx += processor.num_mel_frames_per_step
            start = processor.audio_chunk_start(mel_frame_idx)
        yield processor(audio[start:], sampling_rate=RATE, is_streaming=True, is_first_audio_chunk=False,
                        is_last_audio_chunk=True)

    speaker_cache, logits, chunk_ms = None, [], []
    with torch.inference_mode():
        for inputs in chunks():
            t = time.perf_counter()
            out = model(**inputs.to(device, dtype=dtype), speaker_cache=speaker_cache)
            sync(device)
            chunk_ms.append((time.perf_counter() - t) * 1000)
            logits.append(out.logits)
            speaker_cache = out.speaker_cache
    return torch.cat(logits, dim=1), chunk_ms


def agreement(ours, theirs, length, step=0.01):
    """Frame agreement between two diarizations, mapping each of our speakers to the one of theirs it
    overlaps most (one to one, greedy). Returns shares of the frames where either side hears speech."""
    n = int(length / step)

    def grid(turns, key=lambda t: (t["start"], t["end"], t["speaker"])):
        g = {}
        for t in turns:
            s, e, spk = key(t)
            g.setdefault(spk, np.zeros(n, bool))[max(0, int(s / step)):min(n, int(e / step))] = True
        return g

    a = grid(ours, key=lambda t: (t["Start"], t["End"], t["Speaker"]))
    b = grid(theirs)
    pairs = sorted(((int((a[i] & b[j]).sum()), i, j) for i in a for j in b), reverse=True)
    mapping, used = {}, set()
    for overlap, i, j in pairs:
        if overlap and i not in mapping and j not in used:
            mapping[i] = j
            used.add(j)
    any_a = np.any(list(a.values()), axis=0) if a else np.zeros(n, bool)
    any_b = np.any(list(b.values()), axis=0) if b else np.zeros(n, bool)
    either = any_a | any_b
    same = np.zeros(n, bool)
    for i, j in mapping.items():
        same |= a[i] & b[j]
    total = either.sum() or 1
    return {
        "speakers_ours": len(a), "speakers_theirs": len(b), "mapping": {str(k): v for k, v in mapping.items()},
        "same_speaker": round(same.sum() / total, 3),
        "only_ours": round((any_a & ~any_b).sum() / total, 3),
        "only_theirs": round((any_b & ~any_a).sum() / total, 3),
        "different_speaker": round(((any_a & any_b) & ~same).sum() / total, 3),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wav", type=Path)
    p.add_argument("--mode", default="offline", choices=["offline", "low_latency", "very_low_latency", "ultra_low_latency"])
    p.add_argument("--device", default="mps", choices=["mps", "cpu"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--length", type=float, default=0, help="seconds, 0 for the rest of the file")
    p.add_argument("--compare", type=Path, help="JSON with a top-level turns list of {start, end, speaker}, such as `da who --json`")
    p.add_argument("--out", type=Path, default=Path("out"))
    args = p.parse_args()
    dtype = getattr(torch, args.dtype)

    t = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioFrameClassification.from_pretrained(MODEL_ID, dtype=dtype).to(args.device).eval()
    load_s = time.perf_counter() - t

    audio = load_wav(args.wav, args.start, args.length)
    duration = len(audio) / RATE
    if args.mode != "offline":
        processor.set_streaming_mode(args.mode)
    # Warm up kernels on 10s so the first run is not mostly compilation.
    run = run_offline if args.mode == "offline" else run_streaming
    run(model, processor, audio[: 10 * RATE], args.device, dtype)

    t = time.perf_counter()
    logits, chunk_ms = run(model, processor, audio, args.device, dtype)
    wall = time.perf_counter() - t
    segments = processor.extract_speaker_dict(logits.float().cpu())[0]

    result = {
        "mode": args.mode, "device": args.device, "dtype": args.dtype,
        "audio_s": round(duration, 1), "load_s": round(load_s, 1), "wall_s": round(wall, 2),
        "x_realtime": round(duration / wall, 1),
        "speakers": len({s["Speaker"] for s in segments}),
        # ru_maxrss is bytes on macOS
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),
    }
    if args.device == "mps":
        result["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 2**20)
    if chunk_ms:
        ms = np.array(chunk_ms[1:-1] or chunk_ms)
        result["chunk"] = {
            "latency_ms": processor.streaming_latency_ms,
            "hop_ms": round(processor.num_mel_frames_per_step * 10),
            "compute_ms_p50": round(float(np.percentile(ms, 50)), 1),
            "compute_ms_p95": round(float(np.percentile(ms, 95)), 1),
            "compute_ms_max": round(float(ms.max()), 1),
        }
    if args.compare:
        ref = json.loads(args.compare.read_text())
        theirs = [{"start": t["start"] - args.start, "end": t["end"] - args.start, "speaker": t["speaker"]}
                  for t in ref["turns"] if t["end"] > args.start and t["start"] < args.start + duration]
        result["agreement"] = agreement(segments, theirs, duration)

    args.out.mkdir(exist_ok=True)
    name = f"{args.wav.stem}.{args.mode}.{args.device}.{args.dtype}.{int(args.start)}+{int(duration)}"
    (args.out / f"{name}.json").write_text(json.dumps({**result, "segments": segments}, indent=1))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
