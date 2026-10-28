"""uv run test_bench.py"""

from bench import agreement


def seg(start, end, speaker):
    return {"Start": start, "End": end, "Speaker": speaker}


def turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


# Same turns under different numbers agree completely.
r = agreement([seg(0, 2, 0), seg(2, 4, 1)], [turn(0, 2, 5), turn(2, 4, 3)], 4)
assert r["same_speaker"] == 1 and r["mapping"] == {"0": 5, "1": 3}, r

# One speaker on our side for two on theirs: only one of theirs can map to it.
r = agreement([seg(0, 4, 0)], [turn(0, 3, 1), turn(3, 4, 2)], 4)
assert (r["same_speaker"], r["different_speaker"]) == (0.75, 0.25), r

# Speech heard by one side only.
r = agreement([seg(0, 1, 0)], [turn(0, 1, 0), turn(2, 3, 0)], 4)
assert (r["same_speaker"], r["only_theirs"]) == (0.5, 0.5), r
print("ok")
