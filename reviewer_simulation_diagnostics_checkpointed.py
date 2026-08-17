#!/usr/bin/env python3
"""Checkpointed/resumable driver for reviewer_simulation_diagnostics.py.

Completed datasets are committed after every batch. Restart with the same
command and output directory to skip completed condition/replicate keys.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from reviewer_simulation_diagnostics import (
    Config,
    failure_summary,
    model_selection_summary,
    parameter_summary,
    run_one,
)


KEYS = ["mu_true", "alpha_true", "gamma_true", "phi_true", "r_true", "rep"]
FILES = {
    "fits": "reviewer_raw_fits.csv",
    "attempts": "reviewer_all_optimizer_starts.csv",
    "mcse": "reviewer_likelihood_mcse.csv",
    "intervals": "reviewer_bootstrap_intervals.csv",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--bootstrap-reps", type=int, default=50)
    p.add_argument("--bootstrap-starts", type=int, default=1)
    p.add_argument("--n-starts", type=int, default=3)
    p.add_argument("--mus", type=float, nargs="+", default=[-0.11])
    p.add_argument("--alphas", type=float, nargs="+", default=[-7.7, -4.0, -1.58])
    p.add_argument("--gammas", type=float, nargs="+", default=[-2.5, -0.06, 0.0, 1.17])
    p.add_argument("--phis", type=float, nargs="+", default=[0.22])
    p.add_argument("--rs", type=float, nargs="+", default=[0.004])
    p.add_argument("--T", type=int, default=60)
    p.add_argument("--fit-particles", type=int, default=1200)
    p.add_argument("--selection-particles", type=int, default=500)
    p.add_argument("--eval-repeats", type=int, default=10)
    p.add_argument("--eval-particles", type=int, nargs="+", default=[500, 1000, 2000])
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--output-dir", default="reviewer_simulation_results")
    return p.parse_args()


def atomic_csv(df: pd.DataFrame, path: Path):
    """Replace a CSV only after its temporary copy has been written completely."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_csv(path: Path):
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def combine(old: pd.DataFrame, new: pd.DataFrame, subset: list[str]):
    if old.empty:
        result = new.copy()
    elif new.empty:
        result = old.copy()
    else:
        result = pd.concat([old, new], ignore_index=True)
    available = [c for c in subset if c in result.columns]
    return result.drop_duplicates(subset=available, keep="last") if available else result


def key_from_values(values):
    # Stable textual representation avoids minor CSV float round-trip differences.
    return tuple(format(float(x), ".12g") for x in values[:-1]) + (int(values[-1]),)


def task_key(task):
    mu, alpha, gamma, phi, r, rep, _seed = task
    return key_from_values((mu, alpha, gamma, phi, r, rep))


def completed_keys(fits: pd.DataFrame):
    if fits.empty or not set(KEYS).issubset(fits.columns):
        return set()
    # A dataset is complete only when rows for both fitted models were committed.
    counts = fits.groupby(KEYS, dropna=False)["model"].nunique().reset_index(name="n_models")
    counts = counts[counts.n_models >= 2]
    return {key_from_values(row) for row in counts[KEYS].itertuples(index=False, name=None)}


def build_tasks(cfg: Config):
    total = (len(cfg.mus) * len(cfg.alphas) * len(cfg.gammas)
             * len(cfg.phis) * len(cfg.rs) * cfg.reps)
    seeds = np.random.SeedSequence(cfg.seed).spawn(total)
    tasks, i = [], 0
    for mu in cfg.mus:
        for alpha in cfg.alphas:
            for gamma in cfg.gammas:
                for phi in cfg.phis:
                    for r in cfg.rs:
                        for rep in range(cfg.reps):
                            seed = int(seeds[i].generate_state(1)[0])
                            tasks.append((mu, alpha, gamma, phi, r, rep, seed))
                            i += 1
    return tasks


def write_summaries(out: Path, tables: dict[str, pd.DataFrame]):
    fits, attempts = tables["fits"], tables["attempts"]
    mcse, intervals = tables["mcse"], tables["intervals"]
    if not fits.empty:
        atomic_csv(parameter_summary(fits, intervals), out / "reviewer_parameter_summary.csv")
    if not fits.empty and not attempts.empty:
        atomic_csv(failure_summary(fits, attempts), out / "reviewer_failure_summary.csv")
    if not mcse.empty:
        atomic_csv(model_selection_summary(mcse), out / "reviewer_model_selection_summary.csv")


def config_for_file(cfg: Config, batch_size: int):
    value = asdict(cfg)
    value["batch_size"] = batch_size
    # Normalize tuples to the same list representation produced by JSON reads.
    return json.loads(json.dumps(value))


def validate_resume_config(path: Path, current: dict):
    if not path.exists():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    # Parallelism and batching may change safely; statistical settings may not.
    ignored = {"n_jobs", "batch_size", "output_dir"}
    a = {k: v for k, v in previous.items() if k not in ignored}
    b = {k: v for k, v in current.items() if k not in ignored}
    if a != b:
        raise RuntimeError(
            "This output directory contains checkpoints from different statistical "
            "settings. Use a new --output-dir or restore the original settings."
        )


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    cfg = Config(
        reps=args.reps, T=args.T, mus=tuple(args.mus), alphas=tuple(args.alphas),
        gammas=tuple(args.gammas), phis=tuple(args.phis), rs=tuple(args.rs),
        fit_particles=args.fit_particles, n_starts=args.n_starts,
        selection_particles=args.selection_particles,
        eval_particles=tuple(args.eval_particles), eval_repeats=args.eval_repeats,
        bootstrap_reps=args.bootstrap_reps, bootstrap_starts=args.bootstrap_starts,
        seed=args.seed, n_jobs=args.n_jobs, output_dir=args.output_dir,
    )
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "configuration.json"
    current_config = config_for_file(cfg, args.batch_size)
    validate_resume_config(config_path, current_config)
    config_path.write_text(json.dumps(current_config, indent=2), encoding="utf-8")

    tables = {name: read_csv(out / filename) for name, filename in FILES.items()}
    done = completed_keys(tables["fits"])
    all_tasks = build_tasks(cfg)
    tasks = [task for task in all_tasks if task_key(task) not in done]
    print(f"Planned datasets: {len(all_tasks)}")
    print(f"Already checkpointed: {len(done)}")
    print(f"Remaining: {len(tasks)}")

    for start in range(0, len(tasks), args.batch_size):
        batch = tasks[start:start + args.batch_size]
        results = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
            delayed(run_one)(task, cfg) for task in batch
        )
        new = {
            "fits": pd.DataFrame([x for result in results for x in result[0]]),
            "attempts": pd.DataFrame([x for result in results for x in result[1]]),
            "mcse": pd.DataFrame([x for result in results for x in result[2]]),
            "intervals": pd.DataFrame([x for result in results for x in result[3]]),
        }
        dedupe = {
            "fits": KEYS + ["model"],
            "attempts": KEYS + ["model", "start"],
            "mcse": KEYS + ["particles", "quantity"],
            "intervals": KEYS,
        }
        for name, filename in FILES.items():
            tables[name] = combine(tables[name], new[name], dedupe[name])
            # Empty interval files are intentionally omitted when bootstrap_reps=0.
            if not tables[name].empty:
                atomic_csv(tables[name], out / filename)
        write_summaries(out, tables)
        completed = min(start + len(batch), len(tasks))
        print(f"Checkpoint saved: {completed}/{len(tasks)} remaining-task datasets")

    write_summaries(out, tables)
    print("Done. Re-running this command will resume and skip completed datasets.")


if __name__ == "__main__":
    main()
