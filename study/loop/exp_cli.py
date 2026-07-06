"""CLI wrapper so the acting research-agent (this session) can execute one real
experiment per call and read back real numbers, exactly as the loop's tool use
is meant to work: `python3 exp_cli.py <model> '<params_json>' <seed>`"""
import json
import sys

from executor import run_experiment

if __name__ == "__main__":
    model = sys.argv[1]
    params = json.loads(sys.argv[2])
    seed = int(sys.argv[3])
    print(json.dumps(run_experiment(model, params, seed)))
