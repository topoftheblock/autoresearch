"""Batch runner for fixed-budget configs where all experiment params are decided
up front (no intermediate decision depends on a not-yet-run result).
Usage: python3 exp_batch.py <seed> '<json list of {"model":..,"params":..}>'"""
import json
import sys

from executor import run_experiment

if __name__ == "__main__":
    seed = int(sys.argv[1])
    reqs = json.loads(sys.argv[2])
    results = [run_experiment(r["model"], r["params"], seed) for r in reqs]
    print(json.dumps(results))
