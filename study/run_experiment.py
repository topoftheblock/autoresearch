"""
Master execution script for the causal study defined in thesis_new.tex.

Executes the complete 2^5 factorial over the components of program.md, crossed
with the task suite and replicated R times, and writes one regression-ready row
per run.

    N = 2^5 * D * R                                        (thesis 5.7)

Balance is preserved by construction, not assumed: a run that fails to produce a
valid experiment is repeated with a fresh sampling seed until every cell holds
exactly R completed runs (thesis 5.11).

Usage
    python3 run_experiment.py --check          validate config, print the plan
    python3 run_experiment.py --dry-run        one cell, no API calls
    python3 run_experiment.py --reference      compute a0(d) and a*(d) only
    python3 run_experiment.py                  full study
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


def report_headroom(refs):
    """Which regret scales are usable per dataset (thesis 5.5 exclusion rule)."""
    print("\nHeadroom check  a*(d) - a0(d):")
    for d, r in refs.items():
        hc, hv = r["a_star_cv"] - r["a0_cv"], r["a_star_val"] - r["a0_val"]
        exc = C.REGRET_SCALES_EXCLUDED.get(d, [])
        note = f"  EXCLUDED: {', '.join(exc)}" if exc else ""
        print(f"  {d:14s} cv {hc:+.4f}   val {hv:+.4f}{note}")
        for scale, h in (("cv", hc), ("val", hv)):
            if h <= 0 and scale not in exc:
                print(f"    WARNING: {d}/{scale} has no headroom but is not in "
                      f"REGRET_SCALES_EXCLUDED; regret will be undefined.")


def cmd_check():
    miss = C.missing()
    print("Configuration audit")
    print("-------------------")
    print(f"  agent model      : {C.AGENT_MODEL}")
    print(f"  temperature      : {C.TEMPERATURE}   (fixed for every run, thesis 5.11)")
    print(f"  API sampling seed: {'sent per run' if C.SEND_API_SEED else 'NOT SENT'}")
    print(f"  split            : {int((1-C.TEST_SIZE)*100)}/{int(C.TEST_SIZE*100)} "
          f"stratified, seed {C.SPLIT_SEED}")
    print(f"  cv               : {C.CV_FOLDS}-fold {C.SCORING}")
    print(f"  caps             : HARD_CAP={C.HARD_CAP}, MAX_BAD={C.MAX_BAD}")
    print(f"  window           : first {C.WINDOW} experiments (+ _full columns)")
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
    """Values used ONLY to exercise the pipeline. Not written to study_config.py,
    not sanctioned by the thesis, and not usable for a real run."""
    print("=" * 72)
    print("PROVISIONAL VALUES -- dry run only. These are NOT from thesis_new.tex")
    print("and are discarded when the process exits. A real run requires them to")
    print("be decided and written into config/study_config.py.")
    print("=" * 72)
    prov = {
        "DATASETS": ["breast_cancer"],          # the task the pilot used
        "REPLICATES_R": 2,
        "N_ESTIMATORS_RANGE": (10, 500),
        "REFINEMENT_FLOAT_TOL": 1e-9,
        "REFERENCE_GRID": {                      # deliberately tiny, to stay quick
            "random_forest": {"n_estimators": [50, 200], "max_depth": [4, None]},
            "gradient_boosting": {"n_estimators": [50, 200], "learning_rate": [0.05, 0.1]},
        },
    }
    for k, v in prov.items():
        setattr(C, k, v)
        print(f"  {k} = {v}")
    print()


def cmd_dry_run():
    """Prove the pipeline runs end to end without touching the API."""
    _provisional_for_dry_run()
    print("Dry run: executor, reference constants and metric encoding only.\n")
    ds = C.DATASETS[0]
    import datasets as D
    print("dataset:", json.dumps(D.describe(ds)))

    from experiment import run_experiment
    rec = run_experiment("random_forest", {"n_estimators": 50, "max_depth": 4}, 0, ds)
    assert "error" not in rec, rec
    print("executor OK:", {k: rec[k] for k in ("cv_accuracy_mean", "val_accuracy")})

    ref = reference.load_or_compute([ds])[ds]
    print("reference  :", {k: ref[k] for k in
                           ("a0_cv", "a_star_cv", "a0_val", "a_star_val", "grid_points_evaluated")})

    # A synthetic transcript exercising every metric path. No model is called and
    # no such run is written to results/: this validates the encoder, nothing more.
    cfg = build_designs()[0]
    tr = {"run_id": "DRYRUN", "config_id": config_id(cfg), "config": cfg,
          "dataset": ds, "seed": 0, "terminated_by": "model_stop", "n_bad": 0, "steps": []}
    props = [("random_forest", {"n_estimators": 50, "max_depth": 4}, "baseline"),
             ("random_forest", {"n_estimators": 50, "max_depth": 6}, "deepen the trees"),
             ("gradient_boosting", {"n_estimators": 200, "learning_rate": 0.1}, "switch family"),
             ("gradient_boosting", {"n_estimators": 200, "learning_rate": 0.05}, "lower the rate")]
    for i, (m, p, h) in enumerate(props, 1):
        r = run_experiment(m, p, 0, ds)
        assert "error" not in r, r
        tr["steps"] += [{"step": i, "action": "propose", "model": m, "params": p, "content": h},
                        {"step": i, "action": "execute", "content": r},
                        {"step": i, "action": "interpret", "content": "ok"},
                        {"step": i, "action": "decide", "decision": "continue"}]
    row = metrics.encode_run(tr, ref)
    print("\nencoded row ({} columns):".format(len(row)))
    for k, v in row.items():
        print(f"    {k:28s} {v}")
    out = metrics.write_table([row], STUDY / "dry_run_out")
    print(f"\nwrote {out}")
    print("\nDry run OK: executor, reference and encoder all functional.")
    return 0


def _one_cell(cfg, dataset, seed, results):
    """Execute (or reload) one cell. Safe to call from a worker thread."""
    from runner import run_one
    cid = config_id(cfg)
    cell = results / dataset / cid / f"seed{seed}"
    tpath = cell / "transcript.json"
    if tpath.exists():
        return json.loads(tpath.read_text()), True
    # thesis 5.11: repeat with a fresh sampling seed until the cell holds a
    # completed run, so the balance the analysis depends on is preserved.
    attempt, use_seed = 0, seed
    while True:
        try:
            tr = run_one(cfg, dataset, use_seed, cell)
            break
        except RuntimeError as exc:
            attempt += 1
            if attempt > 5:
                raise RuntimeError(
                    f"cell {dataset}/{cid}/seed{seed} failed {attempt} times: {exc}")
            use_seed = 10_000 * (attempt + 1) + seed
    tr["nominal_seed"] = seed
    tr["seed_retries"] = attempt
    cell.mkdir(parents=True, exist_ok=True)
    tpath.write_text(json.dumps(tr, indent=2))
    return tr, False


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

    print("Computing reference constants ...")
    refs = reference.load_or_compute(dsets)
    report_headroom(refs)

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
        try:
            for fut in as_completed(futures):
                cfg, ds, seed = futures[fut]
                row, cached = fut.result()
                with lock:
                    rows.append(row)
                    done += 1
                    el = (time.time() - t0) / 60
                    eta = el / done * (total - done)
                    print(f"[{done}/{total}] {ds} {config_id(cfg)} seed{seed} "
                          f"n={row['n_experiments_full']} term={row['terminated_by']}"
                          f"{' (cached)' if cached else ''} "
                          f"| {el:.1f} min elapsed, ~{eta:.0f} min left")
        except KeyboardInterrupt:
            print("\ninterrupted; completed transcripts are on disk, rerun to resume")
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    rows.sort(key=lambda r: (r["dataset"], r["config_id"], r["seed"]))
    out = metrics.write_table(rows, results)
    print(f"\nwrote {len(rows)} rows to {out}")
    print("Regression table is ready for the model of thesis eq. (2).")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel runs; runs are independent (thesis 5.11)")
    a = ap.parse_args()
    if a.check:
        sys.exit(cmd_check())
    if a.dry_run:
        sys.exit(cmd_dry_run())
    if a.reference:
        C.validate(need_api=False)
        refs = reference.load_or_compute(C.DATASETS)
        report_headroom(refs)
        print(json.dumps(refs, indent=2))
        sys.exit(0)
    sys.exit(cmd_run(workers=a.workers))
