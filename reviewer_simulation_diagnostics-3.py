#!/usr/bin/env python3
"""Reviewer-facing simulation diagnostics for the SCSV and gamma=0 models.

Outputs
-------
reviewer_raw_fits.csv
    One row per data-generating condition/replicate/model, including fit validity.
reviewer_parameter_summary.csv
    Bias, empirical variance/SD, RMSE, bootstrap interval coverage, and failures.
reviewer_likelihood_mcse.csv
    NLL and paired Delta-BIC Monte Carlo SE at every evaluation particle budget.
reviewer_model_selection_summary.csv
    NLL/BIC win rates and their persistence as the particle budget increases.
reviewer_bootstrap_intervals.csv
    Per-replicate parametric-bootstrap confidence intervals.

The script assumes model_fits.py exposes:
    fit_joint_particle_em(y, N=..., seed=...)
    fit_particle_gamma0(y, N=..., seed=...)

Important: an optimizer failure is reported only when a fitter explicitly returns
a convergence flag. Otherwise, failure means an exception or invalid/non-finite fit.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from model_fits import fit_joint_particle_em, fit_particle_gamma0


PARAMS = ("mu", "phi", "alpha", "gamma", "r")
JOINT_K = 5
GAMMA0_K = 4


@dataclass(frozen=True)
class Config:
    reps: int = 100
    T: int = 60
    mus: tuple[float, ...] = (-0.11,)
    alphas: tuple[float, ...] = (-7.7, -4.0, -1.58)
    gammas: tuple[float, ...] = (-2.5, -0.06, 0.0, 1.17)
    phis: tuple[float, ...] = (0.22,)
    rs: tuple[float, ...] = (0.004,)
    fit_particles: int = 1200
    n_starts: int = 3
    selection_particles: int = 500
    eval_particles: tuple[int, ...] = (500, 1000, 2000)
    eval_repeats: int = 10
    bootstrap_reps: int = 50
    bootstrap_starts: int = 1
    bootstrap_level: float = 0.95
    seed: int = 20260812
    n_jobs: int = 8
    output_dir: str = "reviewer_simulation_results"


def get_param(result: Any, *keys: str, default=np.nan):
    for key in keys:
        if isinstance(result, dict) and key in result:
            return result[key]
        if hasattr(result, key):
            return getattr(result, key)
    return default


def get_convergence(result: Any) -> tuple[bool | None, str]:
    """Return explicit optimizer convergence when the fitter exposes it."""
    for key in ("converged", "optimizer_success", "success"):
        value = get_param(result, key, default=None)
        if value is not None:
            return bool(value), key
    opt = get_param(result, "optimizer_result", "opt_result", default=None)
    if opt is not None:
        value = get_param(opt, "success", "converged", default=None)
        if value is not None:
            return bool(value), "optimizer_result.success"
    return None, "not_reported"


def simulate_scsv(T, mu, phi, alpha, gamma, r, rng, clip_logq=(-20, 10)):
    x = np.zeros(T)
    y = np.zeros(T)
    q = np.zeros(T)
    phi = float(np.clip(phi, -0.999, 0.999))
    r = max(float(r), 1e-8)
    q0 = float(np.exp(np.clip(alpha, *clip_logq)))
    x[0] = rng.normal(mu, math.sqrt(q0 / max(1e-8, 1.0 - phi**2)))
    y[0] = x[0] + rng.normal(0.0, math.sqrt(r))
    q[0] = q0
    for t in range(1, T):
        q[t] = np.exp(np.clip(alpha + gamma * abs(x[t - 1] - mu), *clip_logq))
        mean = mu + phi * (x[t - 1] - mu)
        x[t] = mean + rng.normal(0.0, math.sqrt(q[t]))
        y[t] = x[t] + rng.normal(0.0, math.sqrt(r))
    return y


def systematic_resample(weights, rng):
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    return np.searchsorted(np.cumsum(weights), positions)


def pf_loglik(y, theta, particles, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    mu, phi, alpha, gamma, r = (theta[p] for p in PARAMS)
    phi = float(np.clip(phi, -0.999, 0.999))
    r = max(float(r), 1e-12)
    q0 = np.exp(np.clip(alpha, -20, 10))
    states = rng.normal(mu, math.sqrt(q0 / max(1e-8, 1.0 - phi**2)), particles)
    loglik = 0.0
    for t, obs in enumerate(y):
        if t:
            q = np.exp(np.clip(alpha + gamma * np.abs(states - mu), -20, 10))
            states = mu + phi * (states - mu) + rng.normal(0.0, np.sqrt(q))
        logw = -0.5 * (np.log(2 * np.pi * r) + (obs - states) ** 2 / r)
        m = np.max(logw)
        mean_weight = np.mean(np.exp(logw - m))
        if not np.isfinite(mean_weight) or mean_weight <= 0:
            return -np.inf
        loglik += m + np.log(mean_weight)
        weights = np.exp(logw - m)
        weights /= weights.sum()
        states = states[systematic_resample(weights, rng)]
    return float(loglik)


def extract_theta(result, model):
    theta = {
        "mu": float(get_param(result, "mu", "mu_est")),
        "phi": float(get_param(result, "phi", "phi_est")),
        "alpha": float(get_param(result, "alpha", "alpha_est")),
        "gamma": 0.0 if model == "gamma0" else float(get_param(result, "gamma", "gamma_est")),
        "r": float(get_param(result, "r", "r_est")),
    }
    return theta


def admissible(theta):
    vals = np.array(list(theta.values()), dtype=float)
    return bool(np.all(np.isfinite(vals)) and abs(theta["phi"]) < 1 and theta["r"] > 0)


def call_fitter(y, model, particles, seed):
    if model == "joint":
        return fit_joint_particle_em(y, N=particles, seed=seed)
    return fit_particle_gamma0(y, N=particles, seed=seed)


def multistart_fit(y, model, cfg, base_seed, n_starts=None):
    """Run matched starts and select the valid fit with greatest reevaluated LL."""
    n_starts = cfg.n_starts if n_starts is None else n_starts
    attempts = []
    for start in range(n_starts):
        seed = base_seed + 100_003 * start
        try:
            result = call_fitter(y, model, cfg.fit_particles, seed)
            theta = extract_theta(result, model)
            explicit_converged, convergence_source = get_convergence(result)
            valid = admissible(theta)
            ll = pf_loglik(y, theta, cfg.selection_particles, seed + 50_000) if valid else -np.inf
            attempts.append({
                "start": start, "theta": theta, "valid": valid, "ll": ll,
                "explicit_converged": explicit_converged,
                "convergence_source": convergence_source, "error": "",
            })
        except Exception as exc:
            attempts.append({
                "start": start, "theta": None, "valid": False, "ll": -np.inf,
                "explicit_converged": False, "convergence_source": "exception",
                "error": repr(exc),
            })
    candidates = [a for a in attempts if a["valid"] and np.isfinite(a["ll"])]
    best = max(candidates, key=lambda z: z["ll"]) if candidates else None
    return best, attempts


def likelihood_diagnostics(y, joint_theta, gamma0_theta, cfg, base_seed):
    rows = []
    for N in cfg.eval_particles:
        ll_joint, ll_g0, delta_bic = [], [], []
        for repeat in range(cfg.eval_repeats):
            seed = base_seed + 1_000_003 * repeat + 31 * N
            a = pf_loglik(y, joint_theta, N, seed)
            b = pf_loglik(y, gamma0_theta, N, seed + 500_000)
            ll_joint.append(a)
            ll_g0.append(b)
            delta_bic.append(2 * (a - b) - np.log(len(y)))
        for model, vals in (("joint", ll_joint), ("gamma0", ll_g0)):
            vals = np.asarray(vals)
            rows.append({
                "particles": N, "quantity": f"{model}_nll",
                "mean": -vals.mean(), "sd": vals.std(ddof=1),
                "mcse": vals.std(ddof=1) / math.sqrt(len(vals)),
                "repeats": len(vals),
            })
        delta_nll = np.asarray(ll_g0) * -1 - np.asarray(ll_joint) * -1
        rows.append({
            "particles": N, "quantity": "delta_nll",
            "mean": delta_nll.mean(), "sd": delta_nll.std(ddof=1),
            "mcse": delta_nll.std(ddof=1) / math.sqrt(len(delta_nll)),
            "repeats": len(delta_nll),
        })
        vals = np.asarray(delta_bic)
        rows.append({
            "particles": N, "quantity": "delta_bic",
            "mean": vals.mean(), "sd": vals.std(ddof=1),
            "mcse": vals.std(ddof=1) / math.sqrt(len(vals)),
            "repeats": len(vals),
        })
    return rows


def bootstrap_intervals(y, fitted_theta, cfg, base_seed):
    """Parametric-bootstrap percentile intervals for the joint model."""
    estimates = []
    failures = 0
    for b in range(cfg.bootstrap_reps):
        seed = base_seed + 2_000_003 * b
        yb = simulate_scsv(cfg.T, **fitted_theta, rng=np.random.default_rng(seed))
        best, _ = multistart_fit(
            yb, "joint", cfg, seed + 70_000, n_starts=cfg.bootstrap_starts
        )
        if best is None:
            failures += 1
        else:
            estimates.append(best["theta"])
    alpha = 1.0 - cfg.bootstrap_level
    out = {"bootstrap_successes": len(estimates), "bootstrap_failures": failures}
    for p in PARAMS:
        values = np.array([e[p] for e in estimates], dtype=float)
        if len(values) >= max(10, int(0.5 * cfg.bootstrap_reps)):
            out[f"{p}_lower"] = np.quantile(values, alpha / 2)
            out[f"{p}_upper"] = np.quantile(values, 1 - alpha / 2)
        else:
            out[f"{p}_lower"] = np.nan
            out[f"{p}_upper"] = np.nan
    return out


def run_one(task, cfg):
    mu, alpha, gamma, phi, r, rep, seed = task
    truth = {"mu": mu, "phi": phi, "alpha": alpha, "gamma": gamma, "r": r}
    y = simulate_scsv(cfg.T, **truth, rng=np.random.default_rng(seed))
    fit_rows, attempt_rows = [], []
    fits = {}
    for model, offset in (("joint", 10_000), ("gamma0", 20_000)):
        started = time.time()
        best, attempts = multistart_fit(y, model, cfg, seed + offset)
        for a in attempts:
            attempt_rows.append({
                **{f"{p}_true": truth[p] for p in PARAMS}, "rep": rep,
                "seed": seed, "model": model, "start": a["start"],
                "valid": a["valid"], "selection_loglik": a["ll"],
                "explicit_converged": a["explicit_converged"],
                "convergence_source": a["convergence_source"], "error": a["error"],
            })
        row = {
            **{f"{p}_true": truth[p] for p in PARAMS}, "rep": rep, "seed": seed,
            "model": model, "fit_completed": best is not None,
            "all_starts_failed": best is None, "n_starts": cfg.n_starts,
            "n_valid_starts": sum(a["valid"] for a in attempts),
            "runtime_sec": time.time() - started,
        }
        if best:
            fits[model] = best["theta"]
            row.update({f"{p}_est": best["theta"][p] for p in PARAMS})
            row["selected_start"] = best["start"]
            row["explicit_converged"] = best["explicit_converged"]
            row["convergence_source"] = best["convergence_source"]
        fit_rows.append(row)

    likelihood_rows, interval_rows = [], []
    if "joint" in fits and "gamma0" in fits:
        likelihood_rows = likelihood_diagnostics(y, fits["joint"], fits["gamma0"], cfg, seed + 30_000)
        for row in likelihood_rows:
            row.update({"mu_true": mu, "alpha_true": alpha, "gamma_true": gamma,
                        "phi_true": phi, "r_true": r, "rep": rep, "seed": seed})
        if cfg.bootstrap_reps > 0:
            interval = bootstrap_intervals(y, fits["joint"], cfg, seed + 40_000)
            interval.update({"mu_true": mu, "alpha_true": alpha, "gamma_true": gamma,
                             "phi_true": phi, "r_true": r, "rep": rep, "seed": seed})
            for p in PARAMS:
                if np.isfinite(interval[f"{p}_lower"]):
                    interval[f"{p}_covered"] = bool(
                        interval[f"{p}_lower"] <= truth[p] <= interval[f"{p}_upper"]
                    )
                else:
                    interval[f"{p}_covered"] = np.nan
            interval_rows.append(interval)
    return fit_rows, attempt_rows, likelihood_rows, interval_rows


def parameter_summary(fits, intervals):
    joint = fits[(fits.model == "joint") & fits.fit_completed].copy()
    keys = ["mu_true", "alpha_true", "gamma_true", "phi_true", "r_true"]
    coverage = None
    if len(intervals):
        coverage = intervals.groupby(keys).agg(**{
            f"{p}_coverage": (f"{p}_covered", "mean") for p in PARAMS
        }).reset_index()
    rows = []
    attempted = fits[fits.model == "joint"].groupby(keys).size().rename("n_attempted")
    completed = joint.groupby(keys).size().rename("n_completed")
    for condition, group in joint.groupby(keys):
        row = dict(zip(keys, condition))
        row["n_attempted"] = int(attempted.loc[condition])
        row["n_completed"] = int(completed.loc[condition])
        row["fit_failure_rate"] = 1 - row["n_completed"] / row["n_attempted"]
        for p in PARAMS:
            errors = group[f"{p}_est"] - group[f"{p}_true"]
            row[f"{p}_bias"] = errors.mean()
            row[f"{p}_variance"] = group[f"{p}_est"].var(ddof=1)
            row[f"{p}_sd"] = group[f"{p}_est"].std(ddof=1)
            row[f"{p}_rmse"] = np.sqrt(np.mean(errors**2))
        rows.append(row)
    summary = pd.DataFrame(rows)
    if coverage is not None:
        summary = summary.merge(coverage, on=keys, how="left")
    return summary


def model_selection_summary(mcse):
    keys = ["mu_true", "alpha_true", "gamma_true", "phi_true", "r_true", "particles"]
    bic = mcse[mcse.quantity == "delta_bic"].copy()
    bic["joint_wins_bic"] = bic["mean"] > 0
    bic_summary = bic.groupby(keys).agg(
        n=("mean", "size"), mean_delta_bic=("mean", "mean"),
        median_delta_bic=("mean", "median"), mean_delta_bic_mcse=("mcse", "mean"),
        joint_bic_win_rate=("joint_wins_bic", "mean"),
    ).reset_index()
    nll = mcse[mcse.quantity == "delta_nll"].copy()
    nll["joint_wins_nll"] = nll["mean"] > 0
    nll_summary = nll.groupby(keys).agg(
        mean_delta_nll=("mean", "mean"), median_delta_nll=("mean", "median"),
        mean_delta_nll_mcse=("mcse", "mean"),
        joint_nll_win_rate=("joint_wins_nll", "mean"),
    ).reset_index()
    return bic_summary.merge(nll_summary, on=keys, how="outer")


def failure_summary(fits, attempts):
    keys = ["mu_true", "alpha_true", "gamma_true", "phi_true", "r_true", "model"]
    fit_summary = fits.groupby(keys).agg(
        n_datasets=("fit_completed", "size"),
        fit_failure_rate=("fit_completed", lambda x: 1 - x.astype(bool).mean()),
        all_starts_failure_rate=("all_starts_failed", "mean"),
    ).reset_index()
    # This is only a genuine optimizer-failure rate when convergence_source is
    # not "not_reported". Missing flags remain NaN rather than being called failures.
    a = attempts.copy()
    a["optimizer_failure"] = np.where(
        a["convergence_source"].eq("not_reported"),
        np.nan,
        ~a["explicit_converged"].fillna(False).astype(bool),
    )
    start_summary = a.groupby(keys).agg(
        n_start_attempts=("start", "size"),
        invalid_start_rate=("valid", lambda x: 1 - x.astype(bool).mean()),
        optimizer_failure_rate=("optimizer_failure", "mean"),
        optimizer_flags_available=("optimizer_failure", lambda x: x.notna().mean()),
    ).reset_index()
    return fit_summary.merge(start_summary, on=keys, how="left")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--bootstrap-reps", type=int, default=50)
    p.add_argument("--n-starts", type=int, default=3)
    p.add_argument("--mus", type=float, nargs="+", default=[-0.11])
    p.add_argument("--eval-repeats", type=int, default=10)
    p.add_argument("--eval-particles", type=int, nargs="+", default=[500, 1000, 2000])
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--output-dir", default="reviewer_simulation_results")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(reps=args.reps, mus=tuple(args.mus), bootstrap_reps=args.bootstrap_reps,
                 n_starts=args.n_starts, eval_repeats=args.eval_repeats,
                 eval_particles=tuple(args.eval_particles), n_jobs=args.n_jobs,
                 output_dir=args.output_dir)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = []
    ss = np.random.SeedSequence(cfg.seed)
    total = (len(cfg.mus) * len(cfg.alphas) * len(cfg.gammas)
             * len(cfg.phis) * len(cfg.rs) * cfg.reps)
    child_seeds = ss.spawn(total)
    i = 0
    for mu in cfg.mus:
        for alpha in cfg.alphas:
            for gamma in cfg.gammas:
                for phi in cfg.phis:
                    for r in cfg.rs:
                        for rep in range(cfg.reps):
                            seed = int(child_seeds[i].generate_state(1)[0]); i += 1
                            tasks.append((mu, alpha, gamma, phi, r, rep, seed))
    print(f"Running {len(tasks)} datasets; configuration saved to {out}")
    results = Parallel(n_jobs=cfg.n_jobs, verbose=10)(delayed(run_one)(t, cfg) for t in tasks)
    fits = pd.DataFrame([x for result in results for x in result[0]])
    attempts = pd.DataFrame([x for result in results for x in result[1]])
    mcse = pd.DataFrame([x for result in results for x in result[2]])
    intervals = pd.DataFrame([x for result in results for x in result[3]])
    fits.to_csv(out / "reviewer_raw_fits.csv", index=False)
    attempts.to_csv(out / "reviewer_all_optimizer_starts.csv", index=False)
    mcse.to_csv(out / "reviewer_likelihood_mcse.csv", index=False)
    intervals.to_csv(out / "reviewer_bootstrap_intervals.csv", index=False)
    parameter_summary(fits, intervals).to_csv(out / "reviewer_parameter_summary.csv", index=False)
    model_selection_summary(mcse).to_csv(out / "reviewer_model_selection_summary.csv", index=False)
    failure_summary(fits, attempts).to_csv(out / "reviewer_failure_summary.csv", index=False)
    with open(out / "configuration.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
