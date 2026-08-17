import os
import re
import time
import hashlib
from collections import Counter

import numpy as np
import pandas as pd
import mne

from scipy.signal import welch
from scipy.special import logsumexp
from joblib import Parallel, delayed
from tqdm import tqdm

from model_fits import (
    fit_joint_particle_em,
    fit_particle_gamma0,
    fit_sv_state_space,
)


# =========================
# User settings
# =========================

DATA_ROOT = "chbmit_seizures"
OUTFILE = "all_chbmit_seizure_model_result_clean_common500.csv"

CHANNELS = [
    "FP1-F7", "F7-T7",
    "FP2-F8", "F8-T8",
    "F3-C3",
    "F4-C4",
]

BANDS = {
    "gamma": (30, 80)
}

WIN_SEC = 2
STEP_SEC = 2
BLOCK_LEN = 60

PREICTAL_SEC = 600
POSTICTAL_SEC = 600

N_JOBS = 8
BATCH_SIZE = 10

# Common final likelihood evaluation budget
N_EVAL = 500

# For quick testing, set MAX_TASKS = 10.
# For full run, set MAX_TASKS = None.
MAX_TASKS = None


# =========================
# Pre-fit feature QC settings
# =========================

SKIP_BAD_BLOCKS = True

MIN_BLOCK_SD = 1e-4
MAX_ABS_Z = 8.0
MAX_PROP_ABS_Z_GT6 = 0.20


# =========================
# Utility functions
# =========================

def stable_seed(*items, modulo=1_000_000):
    s = "_".join(map(str, items))
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % modulo


def find_edf_path(patient, edf_file):
    path = os.path.join(DATA_ROOT, patient, edf_file)
    if os.path.exists(path):
        return path
    return None


def parse_summary_file(summary_path):
    with open(summary_path, "r", errors="ignore") as f:
        txt = f.read()

    patient = os.path.basename(summary_path).replace("-summary.txt", "")
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

        if len(starts) != len(ends):
            print("Mismatch:", summary_path, edf_file, starts, ends)

        for seizure_id, (s, e) in enumerate(zip(starts, ends)):
            rows.append({
                "patient": patient,
                "edf_file": edf_file,
                "seizure_id": seizure_id,
                "seizure_start": int(s),
                "seizure_end": int(e),
            })

    return rows


def bandpower(x, fs, fmin, fmax):
    freqs, psd = welch(
        x,
        fs=fs,
        nperseg=min(len(x), int(fs * 2))
    )
    mask = (freqs >= fmin) & (freqs <= fmax)
    return np.trapz(psd[mask], freqs[mask])


def make_feature(eeg, fs, win_sec=2, step_sec=2, fmin=30, fmax=80):
    win = int(win_sec * fs)
    step = int(step_sec * fs)

    vals, times = [], []

    for start in range(0, len(eeg) - win, step):
        chunk = eeg[start:start + win]
        vals.append(bandpower(chunk, fs, fmin, fmax))
        times.append(start / fs)

    vals = np.asarray(vals, dtype=float)
    vals = np.log(vals + 1e-8)
    vals = (vals - vals.mean()) / (vals.std() + 1e-8)

    return np.asarray(times), vals


def label_period(t, seizure_start, seizure_end):
    if seizure_start <= t <= seizure_end:
        return "ictal"
    elif seizure_start - PREICTAL_SEC <= t < seizure_start:
        return "preictal"
    elif seizure_end < t <= seizure_end + POSTICTAL_SEC:
        return "postictal"
    else:
        return "interictal"


def block_period(chunk_periods):
    chunk_periods = np.asarray(chunk_periods)

    if (chunk_periods == "ictal").any():
        return "ictal"
    elif (chunk_periods == "preictal").any():
        return "preictal"
    elif (chunk_periods == "postictal").any():
        return "postictal"
    else:
        return "interictal"


def aic_bic(nll, k, n):
    aic = 2 * k + 2 * nll
    bic = k * np.log(n) + 2 * nll
    return aic, bic


# =========================
# Pre-fit QC
# =========================

def robust_zscore(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))

    if not np.isfinite(mad) or mad < 1e-8:
        return x - med

    return 0.6745 * (x - med) / mad


def block_quality_check(y):
    """
    Conservative pre-fit QC for a modeled feature block.
    This uses only the observed feature sequence, not fitted model parameters.
    """
    y = np.asarray(y, dtype=float)

    if len(y) == 0:
        return False, "empty_block"

    if not np.all(np.isfinite(y)):
        return False, "nonfinite_values"

    if np.std(y) < MIN_BLOCK_SD:
        return False, "near_zero_variance"

    rz = robust_zscore(y)

    if np.nanmax(np.abs(rz)) > MAX_ABS_Z:
        return False, "extreme_outlier"

    if np.mean(np.abs(rz) > 6.0) > MAX_PROP_ABS_Z_GT6:
        return False, "too_many_extreme_points"

    return True, "pass"


# =========================
# Particle likelihood helpers
# =========================

def systematic_resample(weights, rng):
    n_particles = len(weights)
    positions = (rng.random() + np.arange(n_particles)) / n_particles
    indexes = np.zeros(n_particles, dtype=int)
    cumulative_sum = np.cumsum(weights)

    i, j = 0, 0
    while i < n_particles:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    return indexes


def normal_logpdf(y, mean, var):
    var = np.maximum(var, 1e-12)
    return -0.5 * (np.log(2 * np.pi * var) + ((y - mean) ** 2) / var)


def pf_loglik_scsv(y, mu, phi, alpha, gamma, r, N=500, seed=123):
    """
    Bootstrap particle-filter log likelihood for the proposed SCSV model.
    Setting gamma=0 gives the constant-variance SSM.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)

    q0 = np.exp(np.clip(alpha, -20, 10))
    x = rng.normal(
        mu,
        np.sqrt(q0 / max(1e-6, 1 - phi ** 2)),
        size=N
    )

    loglik = 0.0

    for t in range(len(y)):
        if t > 0:
            log_q = alpha + gamma * np.abs(x - mu)
            q = np.exp(np.clip(log_q, -20, 10))

            x_mean = mu + phi * (x - mu)
            x = rng.normal(x_mean, np.sqrt(q), size=N)

        logw = normal_logpdf(y[t], x, r)

        m = np.max(logw)
        loglik += m + np.log(np.mean(np.exp(logw - m)))

        w = np.exp(logw - logsumexp(logw))
        idx = systematic_resample(w, rng)
        x = x[idx]

    return loglik


def pf_loglik_sv(y, mu, phi, a, rho, sigma_h, r, N=500, seed=123):
    """
    Bootstrap particle-filter log likelihood for independent SV-SSM:
        h_t = a + rho(h_{t-1}-a) + eta_t
        q_t = exp(h_t)
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)

    h_sd0 = sigma_h / np.sqrt(max(1e-6, 1 - rho ** 2))
    h = rng.normal(a, h_sd0, size=N)
    h = np.clip(h, -20, 10)

    q = np.exp(h)
    x = rng.normal(
        mu,
        np.sqrt(q / max(1e-6, 1 - phi ** 2)),
        size=N
    )

    loglik = 0.0

    for t in range(len(y)):
        if t > 0:
            h = a + rho * (h - a) + rng.normal(0, sigma_h, size=N)
            h = np.clip(h, -20, 10)
            q = np.exp(h)

            x_mean = mu + phi * (x - mu)
            x = rng.normal(x_mean, np.sqrt(q), size=N)

        logw = normal_logpdf(y[t], x, r)

        m = np.max(logw)
        loglik += m + np.log(np.mean(np.exp(logw - m)))

        w = np.exp(logw - logsumexp(logw))
        idx = systematic_resample(w, rng)
        x = x[idx]
        h = h[idx]

    return loglik


def common_budget_bic(y, fit_joint, fit_gamma0, fit_sv, seed, N_eval=500):
    """
    Recompute final NLL/AIC/BIC using the same particle budget for all models.
    This is used only for final model comparison.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    joint_nll = -pf_loglik_scsv(
        y=y,
        mu=fit_joint["mu_est"],
        phi=fit_joint["phi_est"],
        alpha=fit_joint["alpha_est"],
        gamma=fit_joint["gamma_est"],
        r=fit_joint["r_est"],
        N=N_eval,
        seed=seed + 1000,
    )

    gamma0_nll = -pf_loglik_scsv(
        y=y,
        mu=fit_gamma0["mu_est"],
        phi=fit_gamma0["phi_est"],
        alpha=fit_gamma0["alpha_est"],
        gamma=0.0,
        r=fit_gamma0["r_est"],
        N=N_eval,
        seed=seed + 2000,
    )

    sv_nll = -pf_loglik_sv(
        y=y,
        mu=fit_sv["mu_est"],
        phi=fit_sv["phi_est"],
        a=fit_sv["alpha_est"],
        rho=fit_sv["rho_h_est"],
        sigma_h=fit_sv["sigma_h_est"],
        r=fit_sv["r_est"],
        N=N_eval,
        seed=seed + 3000,
    )

    k_joint = 5
    k_gamma0 = 4
    k_sv = 6

    aic_joint, bic_joint = aic_bic(joint_nll, k_joint, n)
    aic_gamma0, bic_gamma0 = aic_bic(gamma0_nll, k_gamma0, n)
    aic_sv, bic_sv = aic_bic(sv_nll, k_sv, n)

    return {
        "n_obs": n,
        "N_eval_common": N_eval,

        "joint_nll": joint_nll,
        "gamma0_nll": gamma0_nll,
        "sv_nll": sv_nll,

        "aic_joint": aic_joint,
        "bic_joint": bic_joint,
        "aic_gamma0": aic_gamma0,
        "bic_gamma0": bic_gamma0,
        "aic_sv": aic_sv,
        "bic_sv": bic_sv,

        "delta_nll_gamma0": gamma0_nll - joint_nll,
        "delta_nll_sv": sv_nll - joint_nll,

        "delta_aic_gamma0": aic_gamma0 - aic_joint,
        "delta_bic_gamma0": bic_gamma0 - bic_joint,

        "delta_aic_sv": aic_sv - aic_joint,
        "delta_bic_sv": bic_sv - bic_joint,

        "joint_wins_gamma0": gamma0_nll > joint_nll,
        "joint_wins_sv": sv_nll > joint_nll,
    }


# =========================
# Task construction
# =========================

def build_tasks_for_event(row):
    patient = row["patient"]
    edf_file = row["edf_file"]
    seizure_id = row["seizure_id"]
    seizure_start = row["seizure_start"]
    seizure_end = row["seizure_end"]

    edf_path = find_edf_path(patient, edf_file)

    if edf_path is None:
        print("Missing EDF:", patient, edf_file)
        return [], Counter({"missing_edf": 1})

    try:
        raw = mne.io.read_raw_edf(
            edf_path,
            preload=True,
            verbose=False
        )
    except Exception as e:
        print("Could not load EDF:", patient, edf_file, e)
        return [], Counter({"load_failed": 1})

    fs = raw.info["sfreq"]
    available = set(raw.ch_names)

    valid_channels = [
        ch for ch in CHANNELS
        if ch in available and not ch.startswith("-")
    ]

    if len(valid_channels) == 0:
        print("No valid channels:", patient, edf_file)
        return [], Counter({"no_valid_channels": 1})

    tasks = []
    qc_counts = Counter()

    for ch in valid_channels:
        try:
            eeg = raw.get_data(picks=[ch])[0]
        except Exception as e:
            print("Could not get channel:", patient, edf_file, ch, e)
            qc_counts["channel_failed"] += 1
            continue

        for band_name, (fmin, fmax) in BANDS.items():
            try:
                times, y = make_feature(
                    eeg,
                    fs,
                    win_sec=WIN_SEC,
                    step_sec=STEP_SEC,
                    fmin=fmin,
                    fmax=fmax
                )
            except Exception as e:
                print("Feature failed:", patient, edf_file, ch, band_name, e)
                qc_counts["feature_failed"] += 1
                continue

            if len(y) < BLOCK_LEN:
                qc_counts["too_short"] += 1
                continue

            feature_df = pd.DataFrame({
                "time_sec": times,
                "feature": y
            })

            feature_df["period"] = feature_df["time_sec"].apply(
                lambda t: label_period(t, seizure_start, seizure_end)
            )

            for start_idx in range(0, len(feature_df) - BLOCK_LEN, BLOCK_LEN):
                chunk = feature_df.iloc[start_idx:start_idx + BLOCK_LEN]

                period = block_period(chunk["period"].values)
                y_block = chunk["feature"].values

                qc_pass, qc_reason = block_quality_check(y_block)
                qc_counts[qc_reason] += 1

                if SKIP_BAD_BLOCKS and not qc_pass:
                    continue

                rz = robust_zscore(y_block)

                tasks.append({
                    "patient": patient,
                    "edf_file": edf_file,
                    "seizure_id": seizure_id,
                    "channel": ch,
                    "band": band_name,
                    "block": start_idx // BLOCK_LEN,
                    "start_time": chunk["time_sec"].iloc[0],
                    "end_time": chunk["time_sec"].iloc[-1],
                    "period": period,
                    "seizure_start": seizure_start,
                    "seizure_end": seizure_end,
                    "time_to_seizure": seizure_start - chunk["time_sec"].iloc[0],
                    "minutes_to_seizure": (
                        seizure_start - chunk["time_sec"].iloc[0]
                    ) / 60,

                    # QC metadata
                    "qc_pass": qc_pass,
                    "qc_reason": qc_reason,
                    "block_mean": float(np.mean(y_block)),
                    "block_sd": float(np.std(y_block)),
                    "block_max_abs_robust_z": float(np.max(np.abs(rz))),

                    "y": y_block,
                })

    return tasks, qc_counts


# =========================
# Model fitting for one task
# =========================

def fit_one_task(task):
    start = time.time()

    seed = stable_seed(
        task["patient"],
        task["edf_file"],
        task["seizure_id"],
        task["channel"],
        task["band"],
        task["block"]
    )

    try:
        y = np.asarray(task["y"], dtype=float)

        fit_joint = fit_joint_particle_em(y, seed=seed)
        fit_gamma0 = fit_particle_gamma0(y, seed=seed)
        fit_sv = fit_sv_state_space(y, seed=seed)

        common = common_budget_bic(
            y=y,
            fit_joint=fit_joint,
            fit_gamma0=fit_gamma0,
            fit_sv=fit_sv,
            seed=seed,
            N_eval=N_EVAL
        )

        return {
            **{k: v for k, v in task.items() if k != "y"},

            "mu_est": fit_joint["mu_est"],
            "phi_est": fit_joint["phi_est"],
            "alpha_est": fit_joint["alpha_est"],
            "gamma_est": fit_joint["gamma_est"],
            "r_est": fit_joint["r_est"],

            "joint_nll_fit": fit_joint.get("nll", np.nan),
            "gamma0_nll_fit": fit_gamma0.get("nll", np.nan),
            "sv_nll_fit": fit_sv.get("nll", np.nan),

            **common,

            "runtime_sec": time.time() - start,
            "success": True,
            "error": "",
        }

    except Exception as e:
        return {
            **{k: v for k, v in task.items() if k != "y"},
            "runtime_sec": time.time() - start,
            "success": False,
            "error": str(e),
        }


# =========================
# Main script
# =========================

def main():
    all_seizures = []

    for root, dirs, files in os.walk(DATA_ROOT):
        for fname in files:
            if fname.endswith("-summary.txt"):
                path = os.path.join(root, fname)
                rows = parse_summary_file(path)
                print(path, "rows:", len(rows))
                all_seizures.extend(rows)

    seizures_df = pd.DataFrame(all_seizures)

    print(seizures_df.head())
    print(seizures_df.columns)
    print("Total seizures:", len(seizures_df))

    if len(seizures_df) > 0:
        print("Patients:", seizures_df["patient"].nunique())

    tasks = []
    total_qc_counts = Counter()

    for _, row in seizures_df.iterrows():
        event_tasks, event_qc_counts = build_tasks_for_event(row)
        tasks.extend(event_tasks)
        total_qc_counts.update(event_qc_counts)

    print("Total clean tasks:", len(tasks))
    print("QC counts:", dict(total_qc_counts))

    if len(tasks) > 0:
        task_df = pd.DataFrame([{k: v for k, v in t.items() if k != "y"} for t in tasks])
        print("Tasks by period:")
        print(task_df["period"].value_counts())

    if os.path.exists(OUTFILE):
        existing = pd.read_csv(OUTFILE)

        done = set(zip(
            existing["patient"],
            existing["edf_file"],
            existing["seizure_id"],
            existing["seizure_start"],
            existing["seizure_end"],
            existing["channel"],
            existing["band"],
            existing["block"]
        ))

        print("Loaded existing rows:", len(existing))

    else:
        existing = pd.DataFrame()
        done = set()

    unfinished = [
        t for t in tasks
        if (
            t["patient"],
            t["edf_file"],
            t["seizure_id"],
            t["seizure_start"],
            t["seizure_end"],
            t["channel"],
            t["band"],
            t["block"]
        ) not in done
    ]

    if MAX_TASKS is not None:
        unfinished = unfinished[:MAX_TASKS]

    print("Total tasks:", len(tasks))
    print("Unfinished:", len(unfinished))
    print("N_JOBS:", N_JOBS)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("Common final N_eval:", N_EVAL)
    print("Output:", OUTFILE)

    new_rows = []

    for batch_start in range(0, len(unfinished), BATCH_SIZE):
        batch = unfinished[batch_start:batch_start + BATCH_SIZE]

        print(
            f"Running batch {batch_start // BATCH_SIZE + 1} / "
            f"{int(np.ceil(len(unfinished) / BATCH_SIZE))}"
        )

        rows = Parallel(
            n_jobs=N_JOBS,
            backend="loky",
            verbose=100
        )(
            delayed(fit_one_task)(t)
            for t in tqdm(batch)
        )

        new_rows.extend(rows)

        save_df = pd.concat(
            [existing, pd.DataFrame(new_rows)],
            ignore_index=True
        )

        save_df = save_df.drop_duplicates(
            subset=[
                "patient",
                "edf_file",
                "seizure_id",
                "seizure_start",
                "seizure_end",
                "channel",
                "band",
                "block"
            ],
            keep="last"
        )

        save_df.to_csv(OUTFILE, index=False)

        print("Saved rows:", len(save_df))

    print("Done.")
    print("Final output:", OUTFILE)


if __name__ == "__main__":
    main()