"""
Adds the behavioral feature the agenda always intended Step 4 to produce once
a human baseline existed: a process-similarity score between each automated
run and the human reference trace (study/human_baseline/transcript.json).

Each transcript's steps are collapsed to a coarse 4-symbol alphabet
(propose / execute / interpret / decide), consecutive duplicates are merged,
and a normalized Levenshtein (edit) distance is computed between that
sequence and the human trace's coarse sequence. 0 = identical action
sequence shape, 1 = maximally different (edit distance equal to the longer
sequence's length).
"""
import json
from pathlib import Path

COARSE = {
    "propose": "propose", "hypothesize": "propose", "design_experiment": "propose",
    "execute": "execute", "observe": "execute",
    "interpret": "interpret",
    "decide": "decide", "decide_next_step": "decide",
}

ROOT = Path(__file__).parent.parent
HUMAN_TRACE_PATH = ROOT / "human_baseline" / "transcript.json"


def coarse_sequence(steps) -> list[str]:
    seq = []
    for s in steps:
        sym = COARSE[s["action"]]
        if not seq or seq[-1] != sym:
            seq.append(sym)
    return seq


def edit_distance(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def human_sequence() -> list[str]:
    trace = json.loads(HUMAN_TRACE_PATH.read_text())
    return coarse_sequence(trace["steps"])


def process_distance(run_steps) -> float:
    human_seq = human_sequence()
    run_seq = coarse_sequence(run_steps)
    d = edit_distance(human_seq, run_seq)
    return round(d / max(len(human_seq), len(run_seq)), 4)


if __name__ == "__main__":
    print("human coarse sequence:", human_sequence())
    print("length:", len(human_sequence()))
