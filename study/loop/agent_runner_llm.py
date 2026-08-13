"""
REAL autoresearch loop (Karpathy-style), LLM-in-the-loop.

Unlike loop/agent_runner.py (which replays a fixed, pre-authored proposal plan),
this runner makes a genuine per-step model call: for every run it starts a fresh
conversation seeded only with that configuration's rendered program.md, asks the
model for the next JSON action, executes real experiments with loop/executor.py
between steps, feeds the real numbers back, and lets the model interpret and
decide when to stop -- exactly the protocol program.md specifies. Each of the 5
seeds is an independent conversation at temperature > 0, so the runs are genuine
replicates rather than one strategy re-scored.

The API key is read from the environment; it is NEVER hard-coded here.

    export OPENAI_API_KEY=sk-...            # your OWN (rotated) key
    export AGENT_MODEL=gpt-4o-mini          # optional; default gpt-4o-mini
    python3 loop/agent_runner_llm.py        # writes results/<config>/seed<n>/transcript.json

Cost note: ~45 runs x a few steps x 2 calls each. With gpt-4o-mini this is a few
cents; with a larger model it is more. It bills YOUR account.
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY / "loop"))
sys.path.insert(0, str(STUDY / "config"))
from executor import run_experiment  # noqa: E402
from render_program import render_program, config_id  # noqa: E402

RESULTS = STUDY / "results"
DESIGN = STUDY / "design" / "configs.json"

API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("AGENT_MODEL", "gpt-4o-mini")
ENDPOINT = "https://api.openai.com/v1/chat/completions"

HELD_OUT = {"M": 0, "B": 0, "S": 0, "O": 0, "E": 1}
ABLATION_SEEDS = [0, 1, 2, 3, 4]
HELDOUT_SEEDS = [5, 6, 7, 8, 9]
HARD_CAP = 8          # absolute safety bound on experiments per run
MAX_BAD = 4          # tolerated malformed/invalid replies before aborting a run


def call_model(messages, temperature=0.7, retries=4):
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.load(r)
            return resp["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            wait = 2 ** attempt
            if e.code in (429, 500, 502, 503):
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    raise RuntimeError("model call failed after retries")


def parse_json(text):
    """Extract the first JSON object from the model reply."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
    raise ValueError(f"no JSON object in reply: {text[:200]!r}")


def run_one(cfg, seed):
    """Drive one full loop for (config, seed). Returns a transcript dict."""
    cid = config_id(cfg)
    program = render_program(cfg, RESULTS / cid / "program.md")
    system = (
        program
        + "\n\nYou are the research agent. Reply with EXACTLY ONE JSON object per "
          "turn and nothing else, following the two shapes specified above. "
          "Propose one experiment at a time; after you see its result, reply with "
          "an interpret object that includes a 'decision' field."
    )
    messages = [{"role": "system", "content": system}]
    steps = []
    executed = []
    bad = 0
    step = 0
    # temperature keyed to the seed so repeats genuinely differ but stay reproducible-ish
    temp = 0.6 + 0.08 * (seed % 5)

    while len(executed) < HARD_CAP and bad < MAX_BAD:
        # --- ask for a proposal ---
        messages.append({"role": "user", "content":
                         "Propose your next experiment as a JSON 'propose' object." if executed
                         else "Begin. Propose your first experiment as a JSON 'propose' object."})
        reply = call_model(messages, temperature=temp)
        messages.append({"role": "assistant", "content": reply})
        try:
            action = parse_json(reply)
        except ValueError:
            bad += 1
            messages.append({"role": "user", "content": "That was not valid JSON. Reply with one JSON object only."})
            continue
        if action.get("action") != "propose" or "model" not in action or "params" not in action:
            bad += 1
            messages.append({"role": "user", "content": "Expected a 'propose' object with 'model' and 'params'. Try again."})
            continue

        step += 1
        steps.append({"step": step, "action": "propose", "model": action["model"],
                      "params": action["params"], "content": action.get("hypothesis", "")})

        # --- execute for real ---
        result = run_experiment(action["model"], action["params"], seed)
        if "error" in result:
            steps.pop()  # not a real experiment; let the model correct
            step -= 1
            bad += 1
            messages.append({"role": "user", "content":
                             f"The executor rejected that: {result['error']}. "
                             f"Propose a valid experiment instead."})
            continue
        steps.append({"step": step, "action": "execute", "content": result})
        executed.append(result)

        # --- ask for interpretation + decision ---
        messages.append({"role": "user", "content":
                         "Executor result: " + json.dumps(result) +
                         "\nReply with a JSON 'interpret' object including a 'decision' "
                         "field ('continue' or 'stop'; include 'final_recommendation' if you stop)."})
        reply = call_model(messages, temperature=temp)
        messages.append({"role": "assistant", "content": reply})
        try:
            action = parse_json(reply)
        except ValueError:
            bad += 1
            action = {"interpretation": "", "decision": "continue"}
        steps.append({"step": step, "action": "interpret", "content": action.get("interpretation", "")})
        decision = action.get("decision", "continue")
        decide = {"step": step, "action": "decide", "decision": "stop" if decision == "stop" else "continue"}
        if decision == "stop":
            decide["final_recommendation"] = action.get("final_recommendation", "")
            steps.append(decide)
            break
        steps.append(decide)

    if not executed:
        raise RuntimeError(f"run {cid} seed{seed}: model produced no valid experiment")
    return {"config_id": cid, "config": cfg, "seed": seed, "steps": steps}


def main():
    if not API_KEY:
        sys.exit("Set OPENAI_API_KEY in your environment first (see module docstring).")
    design = json.loads(DESIGN.read_text())
    plan = [(c, ABLATION_SEEDS) for c in design] + [(HELD_OUT, HELDOUT_SEEDS)]

    total = sum(len(seeds) for _, seeds in plan)
    done = 0
    print(f"model={MODEL}  runs={total}")
    for cfg, seeds in plan:
        for seed in seeds:
            tr = run_one(cfg, seed)
            d = RESULTS / tr["config_id"] / f"seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "transcript.json").write_text(json.dumps(tr, indent=2))
            done += 1
            n_exp = sum(1 for s in tr["steps"] if s["action"] == "execute")
            best = max(s["content"]["cv_accuracy_mean"] for s in tr["steps"] if s["action"] == "execute")
            print(f"[{done}/{total}] {tr['config_id']} seed{seed}: {n_exp} exp, best_cv={best:.5f}")
    print("done. Now rerun: encoding/features.py, analysis/surrogate.py, causal_validation/validate.py")


if __name__ == "__main__":
    main()
