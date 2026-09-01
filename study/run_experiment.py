"""
Master execution script for the causal study defined in thesis_new.tex.

Executes the complete 2^5 factorial over the components of program.md, crossed
with the task suite and replicated R times, and writes one regression-ready row
per run.

    N = 2^5 * D * R                                  (thesis, Experimental setup)

Balance is preserved by construction, not assumed: a run that fails to produce a
valid experiment is repeated with a fresh sampling seed until every cell holds
exactly R completed runs (thesis, Experimental setup).

Usage
    python3 run_experiment.py --check          validate config, print the plan
    python3 run_experiment.py --dry-run        one cell, no API calls
    python3 run_experiment.py --reference      compute the baseline a0(d) only
    python3 run_experiment.py                  full study
    python3 run_experiment.py --encode         rebuild the table from transcripts
    python3 run_experiment.py --analyze        fit eq. (2) and (3) on the table
"""
import argparse
import json
import sys
import time
from pathlib import Path

STUDY = Path(__file__).resolve().parent
for sub in ("config", "design", "loop", "encoding", "analysis"):
    sys.path.insert(0, str(STUDY / sub))

import study_config as C           # noqa: E402
from full_factorial import build_designs, config_id  # noqa: E402
import reference                   # noqa: E402
import metrics                     # noqa: E402


def plan():
    cfgs = build_designs()
    return cfgs, C.DATASETS, C.seeds()


def report_baseline(refs):
    """a0(d) per dataset, and which scales admit a gain (thesis, Experimental setup)."""
    print("\nBaseline a0(d):")
    for d, r in refs.items():
        exc = C.GAIN_SCALES_EXCLUDED.get(d, [])
        note = f"  EXCLUDED: {', '.join(exc)}" if exc else ""
        print(f"  {d:14s} cv {r['a0_cv']:.4f}   val {r['a0_val']:.4f}{note}")
        for scale in ("cv", "val"):
            if r[f"a0_{scale}"] >= 1.0 and scale not in exc:
                print(f"    WARNING: {d}/{scale} a0 is at the ceiling but is not in "
                      f"GAIN_SCALES_EXCLUDED; the gain will be undefined.")
    print(f"  gain computed on the '{C.GAIN_SCALE}' scale")


def cmd_check():
    miss = C.missing()
    print("Configuration audit")
    print("-------------------")
    print(f"  agent model      : {C.AGENT_MODEL}")
    print(f"  temperature      : {C.TEMPERATURE}   (fixed for every run, thesis, Experimental setup)")
    print(f"  API sampling seed: {'sent per run' if C.SEND_API_SEED else 'NOT SENT'}")
    print(f"  split            : {int((1-C.TEST_SIZE)*100)}/{int(C.TEST_SIZE*100)} "
          f"stratified, seed {C.SPLIT_SEED}")
    print(f"  cv               : {C.CV_FOLDS}-fold {C.SCORING}")
    print(f"  caps             : HARD_CAP={C.HARD_CAP}, MAX_BAD={C.MAX_BAD}")
    print(f"  window           : first {C.WINDOW} experiments (+ _full columns)")
    print(f"  gain scale       : {C.GAIN_SCALE}   excluded {C.GAIN_SCALES_EXCLUDED}")
    print(f"  robustness scale : {C.GAIN_SCALE_ROBUSTNESS}   (M claimed only where both agree)")
    print(f"  dataset control  : D_j fixed effect, reference level {C.DATASETS[0]!r}")
    print(f"  configurations   : {len(build_designs())}")
    print(f"  datasets (D)     : {C.DATASETS}")
    print(f"  replicates (R)   : {C.REPLICATES_R}")
    if not miss:
        cfgs, ds, sd = plan()
        print(f"  total runs N     : {len(cfgs)} x {len(ds)} x {len(sd)} "
              f"= {len(cfgs)*len(ds)*len(sd)}")
        print("\nStatus: READY")
        return 0
    print("\nStatus: BLOCKED. Not specified in thesis_new.tex:")
    for m in miss:
        print(f"  - {m}")
    return 1


def _provisional_for_dry_run():
    """Values used ONLY to exercise the pipeline; discarded when the process exits."""
    print("=" * 72)
    print("PROVISIONAL VALUES -- dry run only, not written to study_config.py")
    print("=" * 72)
    for k, v in {"REPLICATES_R": 1}.items():
        setattr(C, k, v)
        print(f"  {k} = {v}")
    print(f"  DATASETS = {C.DATASETS}  (both, so the dataset fixed effect is identified)")
    print()


def _synthetic_plan(cfg, ds):
    """A deterministic proposal sequence for one cell, derived from the config.

    Exercises every path the encoder has to handle -- a malformed reply, a
    whitelist rejection, an exact duplicate, a refinement and a family switch --
    and lets the config drive the shape, so the resulting table has real
    variation on all five axes and on the dataset. No model is called.
    """
    rf = lambda **kw: ("random_forest", {"n_estimators": 50, **kw})
    gb = lambda **kw: ("gradient_boosting", {"n_estimators": 100, **kw})
    n = 3 if cfg["S"] == 0 else 5                       # S: fixed vs adaptive budget
    second = rf(max_depth=5) if cfg["E"] == 1 else gb(learning_rate=0.05)
    if cfg["B"] == 0:                                   # B: one family for the session
        second = rf(max_depth=5)
    plan = [(*rf(max_depth=4), "executed"), (*second, "executed")]
    if cfg["M"] == 1:                                   # a malformed reply
        plan.insert(1, (None, None, "malformed_json"))
    if cfg["O"] == 1:                                   # a whitelist rejection
        plan.append(("random_forest", {"bogus_param": 1}, "executor_rejected"))
    plan.append((*rf(max_depth=4), "executed"))         # exact duplicate of the first
    filler = [rf(max_depth=d) if cfg["B"] == 0 else gb(max_depth=d) for d in (3, 6, 7, 9)]
    while sum(1 for a in plan if a[2] == "executed") < n:
        plan.append((*filler.pop(0), "executed"))
    return plan


def _synthetic_transcript(cfg, ds, seed, execute):
    plan = _synthetic_plan(cfg, ds)
    tr = {"run_id": f"DRY__{config_id(cfg)}__{ds}__seed{seed}", "config_id": config_id(cfg),
          "config": cfg, "dataset": ds, "seed": seed, "terminated_by": "model_stop",
          "n_bad": 0, "steps": [], "attempts": []}
    prose = 200 if cfg["O"] == 1 else 40                # O: verbose vs terse wording
    seen, step, last = set(), 0, None
    for i, (m, prm, outcome) in enumerate(plan, 1):
        key = None if prm is None else (m, tuple(sorted((str(k), str(v)) for k, v in prm.items())))
        tr["attempts"].append({"attempt": i, "model": m, "params": prm, "outcome": outcome,
                               "duplicate": bool(key and key in seen), "error": None})
        if key:
            seen.add(key)
        if outcome != "executed":
            tr["n_bad"] += 1
            continue
        r = execute(m, prm, seed, ds)
        assert "error" not in r, r
        step += 1
        tr["attempts"][-1]["step"] = step
        tr["steps"] += [{"step": step, "action": "propose", "model": m, "params": prm,
                         "content": "h" * prose},
                        {"step": step, "action": "execute", "content": r},
                        {"step": step, "action": "interpret", "content": "i" * prose},
                        {"step": step, "action": "decide", "decision": "continue"}]
        last = (m, prm)
    tr["steps"].append({"step": step, "action": "decide", "decision": "stop",
                        "final_recommendation": "see params", "final_model": last[0],
                        "final_params": last[1]})
    return tr


def cmd_dry_run():
    """Prove the pipeline runs end to end without touching the API."""
    _provisional_for_dry_run()
    import datasets as D
    from experiment import run_experiment

    dsets = C.DATASETS
    for ds in dsets:
        print("dataset:", json.dumps(D.describe(ds)))
    refs = reference.load_or_compute(dsets)
    report_baseline(refs)

    cache = {}                                   # the executor is deterministic here
    def execute(model, params, seed, ds):
        k = (model, tuple(sorted(params.items())), seed, ds)
        if k not in cache:
            cache[k] = run_experiment(model, params, seed, ds)
        return cache[k]

    rows = []
    for ds in dsets:
        for cfg in build_designs():
            for seed in C.seeds():
                tr = _synthetic_transcript(cfg, ds, seed, execute)
                rows.append(metrics.encode_run(tr, refs[ds]))
    print(f"\nencoded {len(rows)} synthetic runs ({len(rows[0])} columns); first row:")
    for k, v in rows[0].items():
        print(f"    {k:34s} {v}")

    out = metrics.write_table(rows, STUDY / "dry_run_out")
    print(f"\nwrote {out}")

    import regression
    regression.run(table=out)
    print("\nDry run OK: executor, baseline, encoder and estimator all functional.")
    return 0


def _one_cell(cfg, ds, seed, results_dir):
    """Execute (or reload) one cell of the design. Returns (transcript, cached).

    Transcripts are written under results_full/<config>/<dataset>/seed<n>/, so an
    interrupted study resumes instead of re-billing completed runs.

    thesis, Experimental setup: "When a run fails to produce a valid experiment it is
    repeated with a fresh seed until the cell holds exactly R completed runs, so
    balance survives by construction." run_one raises when a run produced no
    valid experiment; the replacement seed is drawn from beyond the seed list so
    it can never collide with another cell's replicate, and the transcript
    records which seed actually produced it.
    """
    from runner import run_one, EmptyRun
    cell = Path(results_dir) / config_id(cfg) / ds / f"seed{seed}"
    tpath = cell / "transcript.json"
    if tpath.exists():
        return json.loads(tpath.read_text()), True

    stride = len(C.seeds())
    for k in range(C.CELL_RETRIES + 1):
        use = seed + k * stride
        try:
            tr = run_one(cfg, ds, use, cell)
        except EmptyRun as exc:
            # Only this case earns a replacement seed. ModelUnavailable is an API
            # fault and propagates: substituting a seed because the endpoint was
            # briefly unreachable would silently corrupt the seed list.
            print(f"    {config_id(cfg)}/{ds}/seed{use}: {exc}; retrying with a fresh seed")
            continue
        tr["planned_seed"] = seed
        cell.mkdir(parents=True, exist_ok=True)
        tpath.write_text(json.dumps(tr, indent=2))
        return tr, False
    raise RuntimeError(
        f"cell {config_id(cfg)}/{ds}/seed{seed} produced no valid experiment in "
        f"{C.CELL_RETRIES + 1} attempts; the cell cannot be filled")


def cmd_encode():
    """Rebuild the regression table from the transcripts already on disk.

    Metrics come straight off the structured transcript (thesis, Performance metrics), so
    a change to a metric definition never requires re-running the loop or
    touching the API: re-encode and refit instead.
    """
    C.validate(need_api=False)
    results = STUDY / C.RESULTS_DIRNAME
    trs = [json.loads(p.read_text()) for p in sorted(results.rglob("transcript.json"))]
    if not trs:
        raise SystemExit(f"no transcripts under {results}")
    refs = reference.load_or_compute(sorted({t["dataset"] for t in trs}))
    rows = [metrics.encode_run(t, refs[t["dataset"]]) for t in trs]
    rows.sort(key=lambda r: (r["dataset_id"], r["config_id"], r["seed"]))
    out = metrics.write_table(rows, results)
    print(f"re-encoded {len(rows)} transcripts -> {out}")
    return rows


def cmd_run(workers=1):
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    C.validate()
    cfgs, dsets, seedlist = plan()
    results = STUDY / C.RESULTS_DIRNAME
    results.mkdir(exist_ok=True)

    C.WORKERS = workers
    if workers > 1:
        # RandomForest with n_jobs=-1 already saturates the cores; nesting it
        # inside a thread pool oversubscribes and is slower, not faster.
        C.EXECUTOR_N_JOBS = 1

    print("Computing baseline a0 ...")
    refs = reference.load_or_compute(dsets)
    report_baseline(refs)

    import datasets as D
    for d in dsets:                      # pre-warm the split cache before forking out
        D.load(d)

    work = [(cfg, ds, seed) for ds in dsets for cfg in cfgs for seed in seedlist]
    total, t0 = len(work), time.time()
    print(f"\n{total} runs on {workers} worker(s); executor n_jobs={C.EXECUTOR_N_JOBS}\n")

    rows, done, lock = [], 0, threading.Lock()

    def task(item):
        cfg, ds, seed = item
        tr, cached = _one_cell(cfg, ds, seed, results)
        return metrics.encode_run(tr, refs[ds]), cached

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task, w): w for w in work}
        failed = []
        try:
            for fut in as_completed(futures):
                cfg, ds, seed = futures[fut]
                try:
                    row, cached = fut.result()
                except Exception as exc:                  # noqa: BLE001
                    failed.append((config_id(cfg), ds, seed, repr(exc)))
                    continue
                with lock:
                    rows.append(row)
                    done += 1
                    el = (time.time() - t0) / 60
                    eta = el / done * (total - done)
                    print(f"[{done}/{total}] {ds} {config_id(cfg)} seed{seed} "
                          f"n={row['n_executed_full']} term={row['terminated_by']}"
                          f"{' (cached)' if cached else ''} "
                          f"| {el:.1f} min elapsed, ~{eta:.0f} min left")
        except KeyboardInterrupt:
            print("\ninterrupted; completed transcripts are on disk, rerun to resume")
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    if failed:
        print(f"\n{len(failed)} run(s) did not produce a row:")
        for cid, ds, seed, exc in failed[:20]:
            print(f"  {cid}/{ds}/seed{seed}: {exc}")
        raise RuntimeError(
            f"{len(failed)} of {total} runs failed; the design is incomplete, so no "
            f"table is written. Completed transcripts are on disk; rerun to resume.")

    rows.sort(key=lambda r: (r["dataset_id"], r["config_id"], r["seed"]))
    out = metrics.write_table(rows, results)
    print(f"\nwrote {len(rows)} rows to {out}")
    print("\nFitting thesis eq. (2) and (3) with the dataset fixed effect ...\n")
    import regression
    regression.run(results_dir=C.RESULTS_DIRNAME)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--encode", action="store_true",
                    help="rebuild the table from transcripts on disk; no API calls")
    ap.add_argument("--analyze", action="store_true",
                    help="fit eq. (2) and (3) with the dataset fixed effect")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel runs; runs are independent (thesis, Experimental setup)")
    a = ap.parse_args()
    if a.check:
        sys.exit(cmd_check())
    if a.dry_run:
        sys.exit(cmd_dry_run())
    if a.encode:
        cmd_encode()
        sys.exit(0)
    if a.analyze:
        import regression
        regression.run(results_dir=C.RESULTS_DIRNAME)
        sys.exit(0)
    if a.reference:
        C.validate(need_api=False)
        refs = reference.load_or_compute(C.DATASETS)
        report_baseline(refs)
        print(json.dumps(refs, indent=2))
        sys.exit(0)
    sys.exit(cmd_run(workers=a.workers))
