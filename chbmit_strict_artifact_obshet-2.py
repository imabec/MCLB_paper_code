#!/usr/bin/env python3

import os
import re
import time
import hashlib
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import mne

from scipy.signal import welch
from scipy.special import logsumexp
from scipy.optimize import minimize

from joblib import Parallel, delayed
from tqdm import tqdm

from model_fits import (
    fit_joint_particle_em,
    fit_particle_gamma0,
    fit_sv_state_space,
)


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-root",
        default="chbmit_seizures"
    )

    p.add_argument(
        "--output-csv",
        default="chbmit_strict_artifact_obshet_results.csv"
    )

    # Example:
    # --patients chb01 chb03 chb05
    p.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help="Only analyze these CHB-MIT patients."
    )

    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=10)

    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--obshet-fit-particles", type=int, default=500)
    p.add_argument("--obshet-starts", type=int, default=3)
    p.add_argument("--obshet-maxiter", type=int, default=600)

    p.add_argument(
        "--gamma-low",
        type=float,
        default=30.0
    )

    # Could also rerun 30--45 as a secondary sensitivity analysis.
    p.add_argument(
        "--gamma-high",
        type=float,
        default=80.0
    )

    p.add_argument("--win-sec", type=float, default=2.0)
    p.add_argument("--step-sec", type=float, default=2.0)
    p.add_argument("--block-len", type=int, default=60)

    p.add_argument("--preictal-sec", type=int, default=600)
    p.add_argument("--postictal-sec", type=int, default=600)
    p.add_argument("--fit-particles", type=int, default=500)

    # Raw EEG artifact sensitivity thresholds.
    # MNE stores EDF amplitudes in volts.
    p.add_argument("--max-abs-uv", type=float, default=750.0)
    p.add_argument("--min-sd-uv", type=float, default=0.5)
    p.add_argument("--max-ptp-z", type=float, default=8.0)
    p.add_argument("--max-diff-z", type=float, default=8.0)

    # Feature-level QC retained from original analysis.
    p.add_argument("--max-feature-z", type=float, default=8.0)
    p.add_argument("--max-prop-z6", type=float, default=0.20)

    return p.parse_args()


# ============================================================
# Utilities
# ============================================================

CHANNELS = [
    "FP1-F7", "F7-T7",
    "FP2-F8", "F8-T8",
    "F3-C3", "F4-C4",
]


def stable_seed(*items, modulo=1_000_000):
    s = "_".join(map(str, items))
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % modulo


def systematic_resample(weights, rng):
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(weights), positions)


def normal_logpdf(y, mean, var):
    var = np.maximum(var, 1e-12)

    return -0.5 * (
        np.log(2.0 * np.pi * var)
        + ((y - mean) ** 2) / var
    )


def aic_bic(nll, k, n):
    return (
        2 * k + 2 * nll,
        k * np.log(n) + 2 * nll,
    )


def robust_zscore(x):
    x = np.asarray(x, dtype=float)

    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))

    if not np.isfinite(mad) or mad < 1e-12:
        return np.zeros_like(x)

    return 0.6745 * (x - med) / mad


# ============================================================
# CHB-MIT metadata
# ============================================================

def parse_summary_file(summary_path):
    with open(summary_path, "r", errors="ignore") as f:
        txt = f.read()

    patient = os.path.basename(summary_path).replace(
        "-summary.txt", ""
    )

    rows = []

    sections = re.split(r"File Name:\s*", txt)

    for section in sections[1:]:

        lines = section.strip().splitlines()

        if len(lines) == 0:
            continue

        edf_file = lines[0].strip()

        starts = re.findall(
            r"Seizure(?:\s*\d+)?\s*Start Time:\s*(\d+)",
            section
        )

        ends = re.findall(
            r"Seizure(?:\s*\d+)?\s*End Time:\s*(\d+)",
            section
        )

        for seizure_id, (s, e) in enumerate(
            zip(starts, ends)
        ):
            rows.append({
                "patient": patient,
                "edf_file": edf_file,
                "seizure_id": seizure_id,
                "seizure_start": int(s),
                "seizure_end": int(e),
            })

    return rows


# ============================================================
# Raw EEG artifact QC
# ============================================================

def raw_window_metrics(chunk):
    chunk = np.asarray(chunk, dtype=float)

    return {
        "sd": float(np.std(chunk)),
        "ptp": float(np.ptp(chunk)),
        "max_abs": float(np.max(np.abs(chunk))),
        "max_diff": float(
            np.max(np.abs(np.diff(chunk)))
            if len(chunk) > 1
            else 0.0
        ),
    }


def bandpower(x, fs, fmin, fmax):

    freqs, psd = welch(
        x,
        fs=fs,
        nperseg=min(
            len(x),
            int(fs * 2)
        )
    )

    mask = (
        (freqs >= fmin)
        & (freqs <= fmax)
    )

    if not np.any(mask):
        return np.nan

    return np.trapezoid(
        psd[mask],
        freqs[mask]
    )


def make_feature_strict(
    eeg,
    fs,
    win_sec,
    step_sec,
    fmin,
    fmax,
    max_abs_uv=750.0,
    min_sd_uv=0.5,
    max_ptp_z=8.0,
    max_diff_z=8.0,
):

    eeg = np.asarray(eeg, dtype=float)

    win = int(win_sec * fs)
    step = int(step_sec * fs)

    windows = []

    for start in range(
        0,
        len(eeg) - win,
        step
    ):

        chunk = eeg[
            start:start + win
        ]

        metrics = raw_window_metrics(
            chunk
        )

        windows.append({
            "start": start,
            "time_sec": start / fs,
            "chunk": chunk,
            **metrics,
        })

    if len(windows) == 0:
        raise ValueError(
            "No raw EEG windows available."
        )

    metric_df = pd.DataFrame([
        {
            "sd": w["sd"],
            "ptp": w["ptp"],
            "max_abs": w["max_abs"],
            "max_diff": w["max_diff"],
        }
        for w in windows
    ])

    ptp_z = robust_zscore(
        metric_df["ptp"].values
    )

    diff_z = robust_zscore(
        metric_df["max_diff"].values
    )

    max_abs_v = max_abs_uv * 1e-6
    min_sd_v = min_sd_uv * 1e-6

    values = []
    times = []
    qc_rows = []

    for i, w in enumerate(windows):

        chunk = w["chunk"]

        keep = True
        reason = "pass"

        if not np.all(
            np.isfinite(chunk)
        ):
            keep = False
            reason = "nonfinite"

        elif w["sd"] < min_sd_v:
            keep = False
            reason = "flat_signal"

        elif w["max_abs"] > max_abs_v:
            keep = False
            reason = "extreme_amplitude"

        elif abs(ptp_z[i]) > max_ptp_z:
            keep = False
            reason = "extreme_peak_to_peak"

        elif abs(diff_z[i]) > max_diff_z:
            keep = False
            reason = "abrupt_transient"

        qc_rows.append({
            "time_sec": w["time_sec"],
            "raw_qc_pass": keep,
            "raw_qc_reason": reason,
            "raw_sd": w["sd"],
            "raw_ptp": w["ptp"],
            "raw_max_abs": w["max_abs"],
            "raw_max_diff": w["max_diff"],
            "raw_ptp_z": ptp_z[i],
            "raw_diff_z": diff_z[i],
        })

        if not keep:
            continue

        bp = bandpower(
            chunk,
            fs,
            fmin,
            fmax,
        )

        if not np.isfinite(bp):
            continue

        values.append(bp)
        times.append(w["time_sec"])

    values = np.asarray(
        values,
        dtype=float
    )

    times = np.asarray(
        times,
        dtype=float
    )

    if len(values) < 2:
        raise ValueError(
            "Too few clean windows after raw artifact rejection."
        )

    values = np.log(
        values + 1e-8
    )

    values = (
        values - values.mean()
    ) / (
        values.std() + 1e-8
    )

    return (
        times,
        values,
        pd.DataFrame(qc_rows)
    )


# ============================================================
# Feature-level QC
# ============================================================

def block_quality_check(
    y,
    max_feature_z=8.0,
    max_prop_z6=0.20
):

    y = np.asarray(
        y,
        dtype=float
    )

    if len(y) == 0:
        return False, "empty_block"

    if not np.all(
        np.isfinite(y)
    ):
        return False, "nonfinite_feature"

    if np.std(y) < 1e-4:
        return False, "near_zero_variance"

    rz = robust_zscore(y)

    if np.max(
        np.abs(rz)
    ) > max_feature_z:

        return False, "feature_extreme"

    if np.mean(
        np.abs(rz) > 6
    ) > max_prop_z6:

        return False, "too_many_extremes"

    return True, "pass"


# ============================================================
# Stage labels
# ============================================================

def label_period(
    t,
    seizure_start,
    seizure_end,
    preictal_sec,
    postictal_sec
):

    if seizure_start <= t <= seizure_end:
        return "ictal"

    if (
        seizure_start - preictal_sec
        <= t
        < seizure_start
    ):
        return "preictal"

    if (
        seizure_end
        < t
        <= seizure_end + postictal_sec
    ):
        return "postictal"

    return "interictal"


def block_period(periods):
    periods = np.asarray(periods)

    if np.any(
        periods == "ictal"
    ):
        return "ictal"

    if np.any(
        periods == "preictal"
    ):
        return "preictal"

    if np.any(
        periods == "postictal"
    ):
        return "postictal"

    return "interictal"


# ============================================================
# Existing SCSV likelihood
# ============================================================

def pf_loglik_scsv(
    y,
    mu,
    phi,
    alpha,
    gamma,
    r,
    N=500,
    seed=123
):

    rng = np.random.default_rng(
        seed
    )

    y = np.asarray(
        y,
        dtype=float
    )

    q0 = np.exp(
        np.clip(alpha, -20, 10)
    )

    x = rng.normal(
        mu,
        np.sqrt(
            q0 /
            max(
                1e-6,
                1 - phi**2
            )
        ),
        size=N
    )

    loglik = 0.0

    for t in range(len(y)):

        if t > 0:

            log_q = (
                alpha
                + gamma
                * np.abs(x - mu)
            )

            q = np.exp(
                np.clip(
                    log_q,
                    -20,
                    10
                )
            )

            x_mean = (
                mu
                + phi * (x - mu)
            )

            x = rng.normal(
                x_mean,
                np.sqrt(q),
                size=N
            )

        logw = normal_logpdf(
            y[t],
            x,
            r
        )

        m = np.max(logw)

        loglik += (
            m
            + np.log(
                np.mean(
                    np.exp(
                        logw - m
                    )
                )
            )
        )

        w = np.exp(
            logw
            - logsumexp(logw)
        )

        idx = systematic_resample(
            w,
            rng
        )

        x = x[idx]

    return float(loglik)


# ============================================================
# Observation-heteroscedastic baseline
# ============================================================

def pf_loglik_obshet(
    y,
    mu,
    phi,
    alpha_q,
    beta0,
    beta1,
    N=500,
    seed=123,
):

    rng = np.random.default_rng(
        seed
    )

    y = np.asarray(
        y,
        dtype=float
    )

    if not (
        -0.999 < phi < 0.999
    ):
        return -np.inf

    q = np.exp(
        np.clip(
            alpha_q,
            -20,
            10
        )
    )

    init_var = (
        q /
        max(
            1e-8,
            1 - phi**2
        )
    )

    x = rng.normal(
        mu,
        np.sqrt(init_var),
        size=N
    )

    loglik = 0.0

    for t in range(len(y)):

        if t > 0:

            x_mean = (
                mu
                + phi * (x - mu)
            )

            x = rng.normal(
                x_mean,
                np.sqrt(q),
                size=N
            )

        log_r = (
            beta0
            + beta1
            * np.abs(x - mu)
        )

        r_t = np.exp(
            np.clip(
                log_r,
                -20,
                10
            )
        )

        logw = normal_logpdf(
            y[t],
            x,
            r_t
        )

        m = np.max(logw)

        weights_unnorm = np.exp(
            logw - m
        )

        mean_weight = np.mean(
            weights_unnorm
        )

        if (
            not np.isfinite(
                mean_weight
            )
            or mean_weight <= 0
        ):
            return -np.inf

        loglik += (
            m
            + np.log(mean_weight)
        )

        weights = (
            weights_unnorm
            / weights_unnorm.sum()
        )

        idx = systematic_resample(
            weights,
            rng
        )

        x = x[idx]

    return float(loglik)


def fit_obshet(
    y,
    seed,
    N=500,
    n_starts=3,
    maxiter=600
):

    y = np.asarray(
        y,
        dtype=float
    )

    mu0 = float(
        np.mean(y)
    )

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

    base = np.array([
        mu0,
        np.arctanh(phi0),
        np.log(var_y / 2),
        np.log(var_y / 2),
        0.1,
    ])

    rng = np.random.default_rng(
        seed
    )

    candidates = []

    for start in range(
        n_starts
    ):

        z0 = base.copy()

        if start > 0:

            z0 += rng.normal(
                0,
                [
                    0.10,
                    0.30,
                    0.50,
                    0.50,
                    0.30,
                ]
            )

        objective_seed = (
            seed
            + 100_000
            + 10_000 * start
        )

        def objective(z):

            mu = z[0]
            phi = np.tanh(z[1])
            alpha_q = z[2]
            beta0 = z[3]
            beta1 = z[4]

            ll = pf_loglik_obshet(
                y=y,
                mu=mu,
                phi=phi,
                alpha_q=alpha_q,
                beta0=beta0,
                beta1=beta1,
                N=N,
                seed=objective_seed
            )

            if not np.isfinite(ll):
                return 1e12

            return -ll

        result = minimize(
            objective,
            z0,
            method="Nelder-Mead",
            options={
                "maxiter": maxiter,
                "xatol": 1e-4,
                "fatol": 1e-4,
            }
        )

        z = result.x

        theta = {
            "mu_est": z[0],
            "phi_est": np.tanh(
                z[1]
            ),
            "alpha_q_est": z[2],
            "beta0_est": z[3],
            "beta1_est": z[4],
        }

        # Independent comparison likelihood
        ll_eval = pf_loglik_obshet(
            y=y,
            mu=theta["mu_est"],
            phi=theta["phi_est"],
            alpha_q=theta[
                "alpha_q_est"
            ],
            beta0=theta[
                "beta0_est"
            ],
            beta1=theta[
                "beta1_est"
            ],
            N=N,
            seed=(
                seed
                + 900_000
                + start
            )
        )

        if np.isfinite(ll_eval):

            candidates.append({
                **theta,
                "start": start,
                "optimizer_success":
                    bool(result.success),
                "optimizer_message":
                    str(result.message),
                "loglik_eval":
                    ll_eval,
            })

    if len(candidates) == 0:

        raise RuntimeError(
            "All observation-heteroscedastic starts failed."
        )

    return max(
        candidates,
        key=lambda z:
            z["loglik_eval"]
    )


# ============================================================
# Task creation
# ============================================================

def build_tasks_for_event(
    row,
    args
):

    patient = row["patient"]

    # IMPORTANT:
    # Explicit patient restriction
    if (
        args.patients is not None
        and patient not in args.patients
    ):
        return [], Counter()

    edf_file = row["edf_file"]

    edf_path = os.path.join(
        args.data_root,
        patient,
        edf_file
    )

    if not os.path.exists(
        edf_path
    ):
        return [], Counter(
            {"missing_edf": 1}
        )

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        verbose=False
    )

    fs = raw.info["sfreq"]

    valid_channels = [
        ch
        for ch in CHANNELS
        if ch in raw.ch_names
    ]

    tasks = []
    qc_counts = Counter()

    for ch in valid_channels:

        eeg = raw.get_data(
            picks=[ch]
        )[0]

        try:

            times, values, raw_qc = (
                make_feature_strict(
                    eeg=eeg,
                    fs=fs,
                    win_sec=args.win_sec,
                    step_sec=args.step_sec,
                    fmin=args.gamma_low,
                    fmax=args.gamma_high,
                    max_abs_uv=args.max_abs_uv,
                    min_sd_uv=args.min_sd_uv,
                    max_ptp_z=args.max_ptp_z,
                    max_diff_z=args.max_diff_z,
                )
            ) 

        except Exception:

            qc_counts[
                "feature_failed"
            ] += 1

            continue

        qc_counts[
            "raw_windows_total"
        ] += len(raw_qc)

        qc_counts[
            "raw_windows_rejected"
        ] += int(
            (~raw_qc[
                "raw_qc_pass"
            ]).sum()
        )

        feature_df = pd.DataFrame({
            "time_sec": times,
            "feature": values,
        })

        feature_df[
            "period"
        ] = feature_df[
            "time_sec"
        ].apply(
            lambda t:
                label_period(
                    t,
                    row[
                        "seizure_start"
                    ],
                    row[
                        "seizure_end"
                    ],
                    args.preictal_sec,
                    args.postictal_sec,
                )
        )

        if len(
            feature_df
        ) < args.block_len:

            qc_counts[
                "too_short"
            ] += 1

            continue

        for start_idx in range(
            0,
            len(feature_df)
            - args.block_len,
            args.block_len
        ):

            chunk = feature_df.iloc[
                start_idx:
                start_idx
                + args.block_len
            ]

            y = chunk[
                "feature"
            ].values

            qc_pass, qc_reason = (
                block_quality_check(
                    y,
                    max_feature_z=
                        args.max_feature_z,
                    max_prop_z6=
                        args.max_prop_z6,
                )
            )

            qc_counts[
                qc_reason
            ] += 1

            if not qc_pass:
                continue

            tasks.append({
                "patient": patient,
                "edf_file": edf_file,
                "seizure_id":
                    row["seizure_id"],
                "channel": ch,

                "band":
                    f"{args.gamma_low:g}-"
                    f"{args.gamma_high:g}",

                "block":
                    start_idx
                    // args.block_len,

                "start_time":
                    chunk[
                        "time_sec"
                    ].iloc[0],

                "end_time":
                    chunk[
                        "time_sec"
                    ].iloc[-1],

                "period":
                    block_period(
                        chunk[
                            "period"
                        ].values
                    ),

                "seizure_start":
                    row[
                        "seizure_start"
                    ],

                "seizure_end":
                    row[
                        "seizure_end"
                    ],

                "y": y,
            })

    return tasks, qc_counts


# ============================================================
# Fit one feature block
# ============================================================

def fit_one_task(
    task,
    args
):

    started = time.time()

    seed = stable_seed(
        task["patient"],
        task["edf_file"],
        task["seizure_id"],
        task["channel"],
        task["band"],
        task["block"],
    )

    try:

        y = np.asarray(
            task["y"],
            dtype=float
        )

        # --------------------------------
        # Existing models
        # --------------------------------

        joint = fit_joint_particle_em(
            y,
            N=500,
            seed=seed
        )

        gamma0 = fit_particle_gamma0(
            y,
            N=500,
            seed=seed
        )

        sv = fit_sv_state_space(
            y,
            N=500,
            seed=seed
        )

        # --------------------------------
        # Observation-heteroscedastic
        # --------------------------------

        obshet = fit_obshet(
            y=y,
            seed=seed + 40_000,
            N=args.obshet_fit_particles,
            n_starts=args.obshet_starts,
            maxiter=args.obshet_maxiter,
        )

        # --------------------------------
        # Common final evaluation
        # --------------------------------

        joint_ll = pf_loglik_scsv(
            y=y,
            mu=joint["mu_est"],
            phi=joint["phi_est"],
            alpha=joint["alpha_est"],
            gamma=joint[
                "gamma_est"
            ],
            r=joint["r_est"],
            N=args.n_eval,
            seed=seed + 100_000,
        )

        gamma0_ll = pf_loglik_scsv(
            y=y,
            mu=gamma0["mu_est"],
            phi=gamma0["phi_est"],
            alpha=gamma0[
                "alpha_est"
            ],
            gamma=0.0,
            r=gamma0["r_est"],
            N=args.n_eval,
            seed=seed + 200_000,
        )

        obshet_ll = pf_loglik_obshet(
            y=y,
            mu=obshet["mu_est"],
            phi=obshet["phi_est"],
            alpha_q=obshet[
                "alpha_q_est"
            ],
            beta0=obshet[
                "beta0_est"
            ],
            beta1=obshet[
                "beta1_est"
            ],
            N=args.n_eval,
            seed=seed + 300_000,
        )

        joint_nll = -joint_ll
        gamma0_nll = -gamma0_ll
        obshet_nll = -obshet_ll

        n = len(y)

        # joint = 5 parameters
        # gamma0 = 4 parameters
        # obshet = 5 parameters
        joint_aic, joint_bic = (
            aic_bic(
                joint_nll,
                5,
                n
            )
        )

        gamma0_aic, gamma0_bic = (
            aic_bic(
                gamma0_nll,
                4,
                n
            )
        )

        obshet_aic, obshet_bic = (
            aic_bic(
                obshet_nll,
                5,
                n
            )
        )

        return {

            **{
                k: v
                for k, v
                in task.items()
                if k != "y"
            },

            "success": True,
            "error": "",

            # Proposed model
            "mu_est":
                joint["mu_est"],
            "phi_est":
                joint["phi_est"],
            "alpha_est":
                joint["alpha_est"],
            "gamma_est":
                joint["gamma_est"],
            "r_est":
                joint["r_est"],

            # Joint likelihood
            "joint_nll":
                joint_nll,
            "joint_aic":
                joint_aic,
            "joint_bic":
                joint_bic,

            # Gamma0
            "gamma0_nll":
                gamma0_nll,
            "gamma0_aic":
                gamma0_aic,
            "gamma0_bic":
                gamma0_bic,

            # Observation heteroscedastic
            "obshet_nll":
                obshet_nll,
            "obshet_aic":
                obshet_aic,
            "obshet_bic":
                obshet_bic,

            "obshet_mu_est":
                obshet[
                    "mu_est"
                ],
            "obshet_phi_est":
                obshet[
                    "phi_est"
                ],
            "obshet_alpha_q_est":
                obshet[
                    "alpha_q_est"
                ],
            "obshet_beta0_est":
                obshet[
                    "beta0_est"
                ],
            "obshet_beta1_est":
                obshet[
                    "beta1_est"
                ],

            # Positive means JOINT wins
            "delta_nll_gamma0":
                gamma0_nll
                - joint_nll,

            "delta_bic_gamma0":
                gamma0_bic
                - joint_bic,

            "delta_nll_obshet":
                obshet_nll
                - joint_nll,

            "delta_bic_obshet":
                obshet_bic
                - joint_bic,

            "joint_wins_obshet_nll":
                joint_nll
                < obshet_nll,

            "joint_wins_obshet_bic":
                joint_bic
                < obshet_bic,

            "runtime_sec":
                time.time()
                - started,
        }

    except Exception as exc:

        return {

            **{
                k: v
                for k, v
                in task.items()
                if k != "y"
            },

            "success": False,
            "error": repr(exc),
            "runtime_sec":
                time.time()
                - started,
        }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if args.patients is not None:

        print(
            "ONLY analyzing patients:",
            args.patients
        )

    all_seizures = []

    # -----------------------------------
    # Only read requested patient folders
    # -----------------------------------

    for root, dirs, files in os.walk(
        args.data_root
    ):

        for fname in files:

            if not fname.endswith(
                "-summary.txt"
            ):
                continue

            patient = fname.replace(
                "-summary.txt",
                ""
            )

            if (
                args.patients
                is not None
                and patient
                not in args.patients
            ):
                continue

            summary_path = os.path.join(
                root,
                fname
            )

            rows = parse_summary_file(
                summary_path
            )

            all_seizures.extend(
                rows
            )

    seizures_df = pd.DataFrame(
        all_seizures
    )

    if len(seizures_df) == 0:

        raise RuntimeError(
            "No seizures found for requested patients."
        )

    print(
        "Patients found:",
        sorted(
            seizures_df[
                "patient"
            ].unique()
        )
    )

    print(
        "Total seizures:",
        len(seizures_df)
    )

    # -----------------------------------
    # Build strict-QC tasks
    # -----------------------------------

    tasks = []
    total_qc = Counter()

    for _, row in (
        seizures_df.iterrows()
    ):

        event_tasks, qc = (
            build_tasks_for_event(
                row,
                args
            )
        )

        tasks.extend(
            event_tasks
        )

        total_qc.update(
            qc
        )

    print(
        "Clean modeling tasks:",
        len(tasks)
    )

    print(
        "QC summary:",
        dict(total_qc)
    )

    if len(tasks) == 0:

        raise RuntimeError(
            "No clean tasks remained after QC."
        )

    task_df = pd.DataFrame([
        {
            k: v
            for k, v
            in t.items()
            if k != "y"
        }
        for t in tasks
    ])

    print(
        "\nTasks by patient:"
    )

    print(
        task_df[
            "patient"
        ].value_counts()
    )

    print(
        "\nTasks by stage:"
    )

    print(
        task_df[
            "period"
        ].value_counts()
    )

    # -----------------------------------
    # Resume support
    # -----------------------------------

    if os.path.exists(
        args.output_csv
    ):

        existing = pd.read_csv(
            args.output_csv
        )

        key_cols = [
            "patient",
            "edf_file",
            "seizure_id",
            "channel",
            "band",
            "block",
        ]

        done = set(
            map(
                tuple,
                existing[
                    key_cols
                ].values
            )
        )

    else:

        existing = pd.DataFrame()
        done = set()

    unfinished = []

    for t in tasks:

        key = (
            t["patient"],
            t["edf_file"],
            t["seizure_id"],
            t["channel"],
            t["band"],
            t["block"],
        )

        if key not in done:
            unfinished.append(t)

    print(
        "\nRemaining tasks:",
        len(unfinished)
    )

    # -----------------------------------
    # Fit in checkpointed batches
    # -----------------------------------

    new_rows = []

    for batch_start in range(
        0,
        len(unfinished),
        args.batch_size
    ):

        batch = unfinished[
            batch_start:
            batch_start
            + args.batch_size
        ]

        print(
            f"\nBatch "
            f"{batch_start // args.batch_size + 1}"
        )

        rows = Parallel(
            n_jobs=args.n_jobs,
            backend="loky",
            verbose=50,
        )(
            delayed(
                fit_one_task
            )(
                task,
                args
            )
            for task
            in tqdm(batch)
        )

        new_rows.extend(
            rows
        )

        save_df = pd.concat(
            [
                existing,
                pd.DataFrame(
                    new_rows
                )
            ],
            ignore_index=True
        )

        key_cols = [
            "patient",
            "edf_file",
            "seizure_id",
            "channel",
            "band",
            "block",
        ]

        save_df = (
            save_df
            .drop_duplicates(
                subset=key_cols,
                keep="last"
            )
        )

        save_df.to_csv(
            args.output_csv,
            index=False
        )

        print(
            "Saved:",
            len(save_df)
        )

    print(
        "\nDone:",
        args.output_csv
    )


if __name__ == "__main__":
    main()