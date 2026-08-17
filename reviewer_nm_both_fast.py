#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from joblib import Parallel, delayed


# ============================================================
# Arguments
# ============================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--T", type=int, default=60)

    p.add_argument(
        "--mus",
        type=float,
        nargs="+",
        default=[-0.11]
    )

    p.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[-7.7, -4.0, -1.58]
    )

    p.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[-2.5, -0.06, 1.17]
    )

    p.add_argument(
        "--phis",
        type=float,
        nargs="+",
        default=[0.22]
    )

    p.add_argument(
        "--rs",
        type=float,
        nargs="+",
        default=[0.004]
    )

    # CHEAPER objective during optimization
    p.add_argument(
        "--fit-particles",
        type=int,
        default=250
    )

    # STRONGER common final evaluation
    p.add_argument(
        "--eval-particles",
        type=int,
        nargs="+",
        default=[500,1000]
    )

    p.add_argument(
        "--n-starts",
        type=int,
        default=2
    )

    p.add_argument(
        "--maxiter",
        type=int,
        default=400
    )

    p.add_argument(
        "--n-jobs",
        type=int,
        default=8
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260814
    )

    p.add_argument(
        "--output-dir",
        default="reviewer_nm_both_fast"
    )

    return p.parse_args()


# ============================================================
# Simulation
# ============================================================

def simulate_scsv(
    T,
    mu,
    phi,
    alpha,
    gamma,
    r,
    rng
):

    x = np.zeros(T)
    y = np.zeros(T)

    phi = float(
        np.clip(phi, -0.999, 0.999)
    )

    r = max(float(r), 1e-8)

    q0 = np.exp(
        np.clip(alpha, -20, 10)
    )

    init_var = (
        q0 /
        max(1e-8, 1 - phi**2)
    )

    x[0] = rng.normal(
        mu,
        np.sqrt(init_var)
    )

    y[0] = (
        x[0]
        + rng.normal(
            0,
            np.sqrt(r)
        )
    )

    for t in range(1, T):

        log_q = (
            alpha
            + gamma
            * abs(x[t - 1] - mu)
        )

        q = np.exp(
            np.clip(log_q, -20, 10)
        )

        mean = (
            mu
            + phi
            * (x[t - 1] - mu)
        )

        x[t] = (
            mean
            + rng.normal(
                0,
                np.sqrt(q)
            )
        )

        y[t] = (
            x[t]
            + rng.normal(
                0,
                np.sqrt(r)
            )
        )

    return y


# ============================================================
# Particle filter
# ============================================================

def systematic_resample(weights, rng):

    weights = np.asarray(weights, dtype=float)

    # Safety checks
    if (
        len(weights) == 0
        or not np.all(np.isfinite(weights))
        or weights.sum() <= 0
    ):
        raise ValueError("Invalid particle weights")

    # Normalize explicitly
    weights = weights / weights.sum()

    n = len(weights)

    positions = (
        rng.random() + np.arange(n)
    ) / n

    cdf = np.cumsum(weights)

    # Important numerical safeguard
    cdf[-1] = 1.0

    idx = np.searchsorted(
        cdf,
        positions,
        side="left"
    )

    # Extra defensive safeguard
    idx = np.clip(
        idx,
        0,
        n - 1
    )

    return idx


def pf_loglik(
    y,
    mu,
    phi,
    alpha,
    gamma,
    r,
    particles=250,
    seed=123
):

    if not (
        -0.999 < phi < 0.999
    ):
        return -np.inf

    if (
        not np.isfinite(r)
        or r <= 0
    ):
        return -np.inf

    rng = np.random.default_rng(seed)

    y = np.asarray(y, dtype=float)

    q0 = np.exp(
        np.clip(alpha, -20, 10)
    )

    init_var = (
        q0 /
        max(1e-8, 1 - phi**2)
    )

    states = rng.normal(
        mu,
        np.sqrt(init_var),
        particles
    )

    loglik = 0.0

    for obs in y:

        logw = -0.5 * (
            np.log(2 * np.pi * r)
            + ((obs - states) ** 2) / r
        )

        m = np.max(logw)

        w = np.exp(logw - m)

        mw = np.mean(w)

        if (
            not np.isfinite(mw)
            or mw <= 0
        ):
            return -np.inf

        loglik += (
            m + np.log(mw)
        )

        w /= w.sum()

        weights = w / np.sum(w)

        idx = systematic_resample(
            weights,
            rng
        )

        states = states[idx]

        # propagate after weighting
        log_q = (
            alpha
            + gamma
            * np.abs(states - mu)
        )

        q = np.exp(
            np.clip(log_q, -20, 10)
        )

        mean = (
            mu
            + phi
            * (states - mu)
        )

        states = (
            mean
            + rng.normal(
                0,
                np.sqrt(q)
            )
        )

    return float(loglik)


# ============================================================
# Parameter transformations
# ============================================================

def decode_joint(z):

    return {
        "mu": float(z[0]),
        "phi": float(np.tanh(z[1])),
        "alpha": float(z[2]),
        "gamma": float(z[3]),
        "r": float(np.exp(z[4])),
    }


def decode_gamma0(z):

    return {
        "mu": float(z[0]),
        "phi": float(np.tanh(z[1])),
        "alpha": float(z[2]),
        "gamma": 0.0,
        "r": float(np.exp(z[3])),
    }


# ============================================================
# Starting values
# ============================================================

def base_start(y, joint=True):

    y = np.asarray(y, dtype=float)

    mu0 = float(np.mean(y))

    if len(y) > 2:

        phi0 = np.corrcoef(
            y[:-1],
            y[1:]
        )[0, 1]

    else:
        phi0 = 0.2

    if not np.isfinite(phi0):
        phi0 = 0.2

    phi0 = np.clip(
        phi0,
        -0.8,
        0.8
    )

    var_y = max(
        float(np.var(y)),
        1e-4
    )

    alpha0 = np.log(
        var_y * 0.5
    )

    r0 = max(
        var_y * 0.25,
        1e-6
    )

    if joint:

        return np.array([
            mu0,
            np.arctanh(phi0),
            alpha0,
            0.1,
            np.log(r0),
        ])

    return np.array([
        mu0,
        np.arctanh(phi0),
        alpha0,
        np.log(r0),
    ])


# ============================================================
# Nelder-Mead fit
# ============================================================

def fit_nm(
    y,
    model,
    seed,
    fit_particles,
    eval_particles,
    n_starts,
    maxiter
):

    is_joint = (model == "joint")

    base = base_start(
        y,
        joint=is_joint
    )

    rng = np.random.default_rng(seed)

    candidates = []

    # Deliberately spread gamma starts for joint model
    gamma_start_grid = [
        -3.0,
        -1.0,
        0.0,
        1.0,
        3.0,
    ]

    for start in range(n_starts):

        z0 = base.copy()

        if is_joint:
            # Force gamma to explore very different regions
            z0[3] = gamma_start_grid[
                start % len(gamma_start_grid)
            ]

            # Perturb other parameters slightly
            z0[0] += rng.normal(0, 0.10)   # mu
            z0[1] += rng.normal(0, 0.25)   # transformed phi
            z0[2] += rng.normal(0, 0.40)   # alpha
            z0[4] += rng.normal(0, 0.40)   # log r

        else:
            # Gamma0 still gets same number of starts
            # with dispersed perturbations in its free parameters
            z0[0] += rng.normal(0, 0.10)
            z0[1] += rng.normal(0, 0.25)
            z0[2] += rng.normal(0, 0.40)
            z0[3] += rng.normal(0, 0.40)

        objective_seed = (
            seed
            + 100_003 * start
        )

        def objective(z):

            theta = (
                decode_joint(z)
                if is_joint
                else decode_gamma0(z)
            )

            ll = pf_loglik(
                y=y,
                **theta,
                particles=fit_particles,
                seed=objective_seed
            )

            if not np.isfinite(ll):
                return 1e12

            return -ll

        # Build a deliberately wider simplex
        ndim = len(z0)

        simplex = np.tile(
            z0,
            (ndim + 1, 1)
        )

        step_sizes = (
            np.array([
                0.20,   # mu
                0.40,   # transformed phi
                0.75,   # alpha
                1.00,   # gamma
                0.75,   # log r
            ])
            if is_joint
            else np.array([
                0.20,
                0.40,
                0.75,
                0.75,
            ])
        )

        for j in range(ndim):
            simplex[j + 1, j] += step_sizes[j]

        result = minimize(
            objective,
            z0,
            method="Nelder-Mead",
            options={
                "maxiter": maxiter,
                "xatol": 1e-3,
                "fatol": 1e-3,
                "adaptive": True,
                "initial_simplex": simplex,
            }
        )

        theta = (
            decode_joint(result.x)
            if is_joint
            else decode_gamma0(result.x)
        )

        # Independent likelihood used to select best start
        ll_eval = pf_loglik(
            y=y,
            **theta,
            particles=eval_particles,
            seed=(
                seed
                + 500_000
                + start
            )
        )

        if np.isfinite(ll_eval):

            candidates.append({
                **theta,

                "start": start,

                "optimizer_success":
                    bool(result.success),

                "nit":
                    int(result.nit),

                "nfev":
                    int(result.nfev),

                "selection_loglik":
                    ll_eval,
            })

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda z:
            z["selection_loglik"]
    )


# ============================================================
# One replicate
# ============================================================

def run_one(task, args):

    (
        mu,
        alpha,
        gamma,
        phi,
        r,
        rep,
        seed
    ) = task

    truth = {
        "mu": mu,
        "phi": phi,
        "alpha": alpha,
        "gamma": gamma,
        "r": r,
    }

    y = simulate_scsv(
        T=args.T,
        **truth,
        rng=np.random.default_rng(seed)
    )

    # ------------------------------------------------
    # Fit JOINT and GAMMA0 models
    # ------------------------------------------------

    joint = fit_nm(
        y,
        "joint",
        seed + 10_000,
        args.fit_particles,
        args.eval_particles,
        args.n_starts,
        args.maxiter
    )

    gamma0 = fit_nm(
        y,
        "gamma0",
        seed + 20_000,
        args.fit_particles,
        args.eval_particles,
        args.n_starts,
        args.maxiter
    )

    base_row = {
        "mu_true": mu,
        "alpha_true": alpha,
        "gamma_true": gamma,
        "phi_true": phi,
        "r_true": r,
        "rep": rep,
        "seed": seed,
    }

    if joint is None or gamma0 is None:

        return {
            **base_row,
            "success": False
        }

    # ------------------------------------------------
    # Common final evaluation at MULTIPLE particle counts
    # ------------------------------------------------

    eval_results = {}

    for N in args.eval_particles:

        joint_ll = pf_loglik(
            y=y,
            mu=joint["mu"],
            phi=joint["phi"],
            alpha=joint["alpha"],
            gamma=joint["gamma"],
            r=joint["r"],
            particles=N,
            seed=seed + 900_000 + N
        )

        gamma0_ll = pf_loglik(
            y=y,
            mu=gamma0["mu"],
            phi=gamma0["phi"],
            alpha=gamma0["alpha"],
            gamma=0.0,
            r=gamma0["r"],
            particles=N,
            seed=seed + 1_400_000 + N
        )

        joint_nll = -joint_ll
        gamma0_nll = -gamma0_ll

        n = len(y)

        joint_bic = (
            2 * joint_nll
            + 5 * np.log(n)
        )

        gamma0_bic = (
            2 * gamma0_nll
            + 4 * np.log(n)
        )

        delta_nll = (
            gamma0_nll
            - joint_nll
        )

        delta_bic = (
            gamma0_bic
            - joint_bic
        )

        eval_results.update({

            f"joint_nll_{N}":
                joint_nll,

            f"gamma0_nll_{N}":
                gamma0_nll,

            f"joint_bic_{N}":
                joint_bic,

            f"gamma0_bic_{N}":
                gamma0_bic,

            f"delta_nll_{N}":
                delta_nll,

            f"delta_bic_{N}":
                delta_bic,

            f"joint_wins_nll_{N}":
                delta_nll > 0,

            f"joint_wins_bic_{N}":
                delta_bic > 0,
        })

    # ------------------------------------------------
    # Return estimates + optimization diagnostics
    # + all particle-budget evaluations
    # ------------------------------------------------

    return {
        **base_row,

        "success": True,

        # Joint estimates
        "joint_mu_est":
            joint["mu"],

        "joint_phi_est":
            joint["phi"],

        "joint_alpha_est":
            joint["alpha"],

        "joint_gamma_est":
            joint["gamma"],

        "joint_r_est":
            joint["r"],

        # Gamma0 estimates
        "gamma0_mu_est":
            gamma0["mu"],

        "gamma0_phi_est":
            gamma0["phi"],

        "gamma0_alpha_est":
            gamma0["alpha"],

        "gamma0_r_est":
            gamma0["r"],

        # Optimization diagnostics
        "joint_optimizer_success":
            joint["optimizer_success"],

        "gamma0_optimizer_success":
            gamma0["optimizer_success"],

        "joint_selected_start":
            joint["start"],

        "gamma0_selected_start":
            gamma0["start"],

        "joint_nit":
            joint["nit"],

        "gamma0_nit":
            gamma0["nit"],

        "joint_nfev":
            joint["nfev"],

        "gamma0_nfev":
            gamma0["nfev"],

        # Multiple particle-budget results
        **eval_results,
    }

# ============================================================
# Tasks
# ============================================================

def build_tasks(args):

    tasks = []

    total = (
        len(args.mus)
        * len(args.alphas)
        * len(args.gammas)
        * len(args.phis)
        * len(args.rs)
        * args.reps
    )

    seeds = np.random.SeedSequence(
        args.seed
    ).spawn(total)

    i = 0

    for mu in args.mus:
        for alpha in args.alphas:
            for gamma in args.gammas:
                for phi in args.phis:
                    for r in args.rs:
                        for rep in range(args.reps):

                            seed = int(
                                seeds[i]
                                .generate_state(1)[0]
                            )

                            tasks.append((
                                mu,
                                alpha,
                                gamma,
                                phi,
                                r,
                                rep,
                                seed
                            ))

                            i += 1

    return tasks


# ============================================================
# Summary
# ============================================================

def summarize(df):

    good = df[
        df["success"] == True
    ].copy()

    keys = [
        "mu_true",
        "alpha_true",
        "gamma_true",
        "phi_true",
        "r_true",
    ]

    return (
        good
        .groupby(keys)
        .agg(
            n=("rep", "size"),

            mean_delta_nll=(
                "delta_nll",
                "mean"
            ),

            median_delta_nll=(
                "delta_nll",
                "median"
            ),

            joint_nll_win_rate=(
                "joint_wins_nll",
                "mean"
            ),

            mean_delta_bic=(
                "delta_bic",
                "mean"
            ),

            median_delta_bic=(
                "delta_bic",
                "median"
            ),

            joint_bic_win_rate=(
                "joint_wins_bic",
                "mean"
            ),

            joint_convergence_rate=(
                "joint_optimizer_success",
                "mean"
            ),

            gamma0_convergence_rate=(
                "gamma0_optimizer_success",
                "mean"
            ),

            median_joint_nfev=(
                "joint_nfev",
                "median"
            ),

            median_gamma0_nfev=(
                "gamma0_nfev",
                "median"
            ),
        )
        .reset_index()
    )


# ============================================================
# Main with checkpointing
# ============================================================

def main():

    args = parse_args()

    out = Path(
        args.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_path = (
        out / "nelder_mead_raw.csv"
    )

    with open(
        out / "configuration.json",
        "w"
    ) as f:
        json.dump(
            vars(args),
            f,
            indent=2
        )

    tasks = build_tasks(args)

    print(
        "Total datasets:",
        len(tasks)
    )

    key_cols = [
        "mu_true",
        "alpha_true",
        "gamma_true",
        "phi_true",
        "r_true",
        "rep",
    ]

    if raw_path.exists():

        existing = pd.read_csv(
            raw_path
        )

        # Only successful rows count as finished
        completed_df = existing[
            existing["success"] == True
        ].copy()

        done = set(
            map(
                tuple,
                completed_df[key_cols].values
            )
        )

        existing = completed_df

    else:
        existing = pd.DataFrame()
        done = set()

    unfinished = []

    for task in tasks:

        key = (
            task[0],
            task[1],
            task[2],
            task[3],
            task[4],
            task[5],
        )

        if key not in done:
            unfinished.append(task)

    print(
        "Already completed:",
        len(done)
    )

    print(
        "Remaining:",
        len(unfinished)
    )

    for start in range(
        0,
        len(unfinished),
        args.batch_size
    ):

        batch = unfinished[
            start:
            start + args.batch_size
        ]

        print(
            f"\nBatch "
            f"{start // args.batch_size + 1}"
        )

        rows = Parallel(
            n_jobs=args.n_jobs,
            backend="loky",
            verbose=10
        )(
            delayed(run_one)(
                task,
                args
            )
            for task in batch
        )

        new = pd.DataFrame(rows)

        existing = pd.concat(
            [existing, new],
            ignore_index=True
        )

        existing = (
            existing
            .drop_duplicates(
                subset=key_cols,
                keep="last"
            )
        )

        existing.to_csv(
            raw_path,
            index=False
        )

        print(
            "Saved:",
            len(existing)
        )

    summary = summarize(
        existing
    )

    summary.to_csv(
        out / "nelder_mead_summary.csv",
        index=False
    )

    good = existing[
        existing["success"] == True
    ]

    print("\nFINAL")
    print(
        "Successful:",
        len(good)
    )

    print(
        "Joint NLL win rate:",
        good[
            "joint_wins_nll"
        ].mean()
    )

    print(
        "Joint BIC win rate:",
        good[
            "joint_wins_bic"
        ].mean()
    )

    print(
        "Median Delta NLL:",
        good[
            "delta_nll"
        ].median()
    )

    print(
        "Median Delta BIC:",
        good[
            "delta_bic"
        ].median()
    )

    print(
        "Joint convergence:",
        good[
            "joint_optimizer_success"
        ].mean()
    )

    print(
        "Gamma0 convergence:",
        good[
            "gamma0_optimizer_success"
        ].mean()
    )


if __name__ == "__main__":
    main()