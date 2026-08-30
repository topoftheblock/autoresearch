"""
One run of the autoresearch loop (thesis 2.2, 5.11).

A run is one execution under one configuration, on one dataset, with one seed.
The instruction file is supplied once as the system message and is not modified
thereafter; the model alternates propose / interpret, and the harness performs
the execution step itself.

Controls enforced here, all from thesis 5.11:
  * the agent model is a pinned dated snapshot, never a moving alias;
  * the decoding temperature is one fixed value for every run, and is strictly
    above zero, because the run-to-run variance is the error term against which
    every coefficient of eq. (2) is tested;
  * an explicit sampling seed is passed to the model interface, so any single
    run can be regenerated exactly while replicates still differ from one another.
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
from experiment import run_experiment  # noqa: E402
from render_program import render_program  # noqa: E402
import study_config as C  # noqa: E402

PROTOCOL = (
    "\n\nYou are the research agent. Reply with EXACTLY ONE JSON object per turn "
    "and nothing else, following the two shapes specified above. Propose one "
    "experiment at a time; after you see its result, reply with an interpret "
    "object that includes a 'decision' field."
)


def call_model(messages, seed):
    key = os.environ.get(C.API_KEY_ENV)
    if not key:
        raise RuntimeError(f"{C.API_KEY_ENV} is not set in the environment")
    payload = {
        "model": C.AGENT_MODEL,
        "messages": messages,
        "temperature": C.TEMPERATURE,
        "response_format": {"type": "json_object"},
    }
    if C.SEND_API_SEED:
        payload["seed"] = seed
    req = urllib.request.Request(
        C.API_ENDPOINT, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(C.API_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    raise RuntimeError("model call failed after retries")


def parse_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
    raise ValueError(f"no JSON object in reply: {text[:200]!r}")


def run_one(config, dataset, seed, program_dir):
    """Execute one run. Returns a transcript dict, or raises if it produced nothing."""
    from full_factorial import config_id  # local import: design/ added by caller
    cid = config_id(config)
    program_dir = Path(program_dir)
    program_dir.mkdir(parents=True, exist_ok=True)
    program = render_program(config, program_dir / "program.md")

    messages = [{"role": "system", "content": program + PROTOCOL}]
    steps, executed, bad, step = [], [], 0, 0
    terminated_by = "hard_cap"

    while len(executed) < C.HARD_CAP and bad < C.MAX_BAD:
        messages.append({"role": "user", "content":
                         "Propose your next experiment as a JSON 'propose' object."
                         if executed else
                         "Begin. Propose your first experiment as a JSON 'propose' object."})
        reply = call_model(messages, seed)
        messages.append({"role": "assistant", "content": reply})
        try:
            action = parse_json(reply)
        except ValueError:
            bad += 1
            messages.append({"role": "user", "content":
                             "That was not valid JSON. Reply with one JSON object only."})
            continue
        if action.get("action") != "propose" or "model" not in action or "params" not in action:
            bad += 1
            messages.append({"role": "user", "content":
                             "Expected a 'propose' object with 'model' and 'params'. Try again."})
            continue

        step += 1
        steps.append({"step": step, "action": "propose", "model": action["model"],
                      "params": action["params"], "content": action.get("hypothesis", "")})

        result = run_experiment(action["model"], action["params"], seed, dataset)
        if "error" in result:
            steps.pop()
            step -= 1
            bad += 1
            messages.append({"role": "user", "content":
                             f"The executor rejected that: {result['error']}. "
                             f"Propose a valid experiment instead."})
            continue
        steps.append({"step": step, "action": "execute", "content": result})
        executed.append(result)

        messages.append({"role": "user", "content":
                         "Executor result: " + json.dumps(result) +
                         "\nReply with a JSON 'interpret' object including a 'decision' "
                         "field ('continue' or 'stop'; include 'final_recommendation' "
                         "if you stop)."})
        reply = call_model(messages, seed)
        messages.append({"role": "assistant", "content": reply})
        try:
            action = parse_json(reply)
        except ValueError:
            bad += 1
            action = {"interpretation": "", "decision": "continue"}
        steps.append({"step": step, "action": "interpret",
                      "content": action.get("interpretation", "")})
        decision = action.get("decision", "continue")
        rec = {"step": step, "action": "decide",
               "decision": "stop" if decision == "stop" else "continue"}
        if decision == "stop":
            rec["final_recommendation"] = action.get("final_recommendation", "")
            steps.append(rec)
            terminated_by = "model_stop"
            break
        steps.append(rec)
    else:
        terminated_by = "max_bad" if bad >= C.MAX_BAD else "hard_cap"

    if not executed:
        raise RuntimeError(f"run {cid}/{dataset}/seed{seed} produced no valid experiment")

    return {
        "run_id": f"{cid}__{dataset}__seed{seed}",
        "config_id": cid, "config": config, "dataset": dataset, "seed": seed,
        "agent_model": C.AGENT_MODEL, "temperature": C.TEMPERATURE,
        "api_seed": seed if C.SEND_API_SEED else None,
        "terminated_by": terminated_by, "n_bad": bad, "steps": steps,
    }
