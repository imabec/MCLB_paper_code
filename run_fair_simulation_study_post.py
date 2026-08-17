#!/usr/bin/env python3

#!/usr/bin/env python3

"""
Fair alpha-gamma-phi-r simulation study for the state-coupled stochastic-volatility SSM.

Purpose:
    Test parameter recovery, sensitivity to alpha/phi/r, and false-positive control
    under the nested gamma=0 null.

Data-generating model:
    x_t = mu + phi (x_{t-1} - mu) + w_t
    w_t ~ N(0, q_t)
    log q_t = alpha + gamma |x_{t-1} - mu|
    y_t = x_t + v_t
    v_t ~ N(0, r)

Models fit:
    1. Proposed state-coupled SSM
    2. Nested gamma=0 constant-variance SSM

Important:
    Calibration is performed separately for each alpha, phi, and r setting because
    recovery of gamma may depend on baseline process variance, temporal persistence,
    and observation noise.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from model_fits import fit_joint_particle_em, fit_particle_gamma0


# ============================================================
# Settings
# ============================================================

OUTFILE = "fair_simulation_results_alpha_gamma_post.csv"
SUMMARY_OUTFILE = "fair_simulation_summary_by_alpha_gamma_post.csv"
DETECTION_OUTFILE = "fair_simulation_detection_summary_by_alpha_gamma_post.csv"
RELIABILITY_OUTFILE = "fair_simulation_reliability_features.csv"

RANDOM_SEED = 123

N_REPS_PER_CONDITION = 100

GAMMA_VALUES = [-3.42,.01,1.62]
ALPHA_VALUES = [-8.33, -5.55, -2.49]
PHI_VALUES = [0.21]
R_VALUES = [0.0009]

T = 60
TRUE_MU = -.16

N_PARTICLES_FIT = 1200
N_PARTICLES_EVAL = 500
N_EVAL_REPEATS = 10

N_JOBS = 8
BATCH_SIZE = 10

K_PROPOSED = 5
K_GAMMA0 = 4


# ============================================================
# Simulation
# ============================================================

def simulate_scsv(T, mu, phi, alpha, gamma, r, rng, clip_logq=(-20, 10)):
    x = np.zeros(T)
    y = np.zeros(T)
    q = np.zeros(T)

    phi = float(np.clip(phi, -0.999, 0.999))
    r = max(float(r), 1e-8)

    q0 = np.exp(np.clip(alpha, clip_logq[0], clip_logq[1]))
    init_var = q0 / max(1e-8, 1.0 - phi**2)

    x[0] = rng.normal(mu, np.sqrt(init_var))
    y[0] = x[0] + rng.normal(0.0, np.sqrt(r))
    q[0] = q0

    for t in range(1, T):
        log_q_t = alpha + gamma * abs(x[t - 1] - mu)
        log_q_t = np.clip(log_q_t, clip_logq[0], clip_logq[1])
        q[t] = np.exp(log_q_t)

        x_mean = mu + phi * (x[t - 1] - mu)
        x[t] = x_mean + rng.normal(0.0, np.sqrt(q[t]))
        y[t] = x[t] + rng.normal(0.0, np.sqrt(r))

    return y, x, q


# ============================================================
# Particle-filter likelihoods
# ============================================================

def systematic_resample(weights, rng):
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative_sum = np.cumsum(weights)
    return np.searchsorted(cumulative_sum, positions)


def pf_loglik_scsv(y, mu, phi, alpha, gamma, r, N=500, seed=0, clip_logq=(-20, 10)):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    T_local = len(y)

    r = max(float(r), 1e-8)
    phi = float(np.clip(phi, -0.999, 0.999))

    q0 = np.exp(np.clip(alpha, clip_logq[0], clip_logq[1]))
    init_var = q0 / max(1e-8, 1.0 - phi**2)

    particles = rng.normal(mu, np.sqrt(init_var), size=N)
    loglik = 0.0

    for t in range(T_local):
        if t > 0:
            log_q = alpha + gamma * np.abs(particles - mu)
            log_q = np.clip(log_q, clip_logq[0], clip_logq[1])
            q = np.exp(log_q)

            mean = mu + phi * (particles - mu)
            particles = mean + rng.normal(0.0, np.sqrt(q), size=N)

        logw = -0.5 * (
            np.log(2.0 * np.pi * r)
            + ((y[t] - particles) ** 2) / r
        )

        max_logw = np.max(logw)
        weights_unnorm = np.exp(logw - max_logw)
        mean_weight = np.mean(weights_unnorm)

        if not np.isfinite(mean_weight) or mean_weight <= 0:
            return -np.inf

        loglik += max_logw + np.log(mean_weight)

        weights = weights_unnorm / np.sum(weights_unnorm)
        idx = systematic_resample(weights, rng)
        particles = particles[idx]

    return float(loglik)


def pf_loglik_gamma0(y, mu, phi, alpha, r, N=500, seed=0):
    return pf_loglik_scsv(
        y=y,
        mu=mu,
        phi=phi,
        alpha=alpha,
        gamma=0.0,
        r=r,
        N=N,
        seed=seed,
    )


def repeated_loglik_scsv(y, mu, phi, alpha, gamma, r, base_seed, repeats=N_EVAL_REPEATS):
    vals = []
    for j in range(repeats):
        vals.append(
            pf_loglik_scsv(
                y=y,
                mu=mu,
                phi=phi,
                alpha=alpha,
                gamma=gamma,
                r=r,
                N=N_PARTICLES_EVAL,
                seed=base_seed + 10000 * j,
            )
        )
    return float(np.mean(vals))


def repeated_loglik_gamma0(y, mu, phi, alpha, r, base_seed, repeats=N_EVAL_REPEATS):
    vals = []
    for j in range(repeats):
        vals.append(
            pf_loglik_gamma0(
                y=y,
                mu=mu,
                phi=phi,
                alpha=alpha,
                r=r,
                N=N_PARTICLES_EVAL,
                seed=base_seed + 10000 * j,
            )
        )
    return float(np.mean(vals))


# ============================================================
# Fit helpers
# ============================================================

def get_param(result, *keys, default=np.nan):
    for key in keys:
        if isinstance(result, dict) and key in result:
            return result[key]
        if hasattr(result, key):
            return getattr(result, key)
    return default


def fit_one_sim(task):
    alpha_true, gamma_true, phi_true, r_true, rep, seed = task
    rng = np.random.default_rng(seed)

    t0 = time.time()

    row = {
        "alpha_true": alpha_true,
        "gamma_true": gamma_true,
        "phi_true": phi_true,
        "r_true": r_true,
        "rep": rep,
        "seed": seed,
        "success": False,
        "error": "",
    }

    try:
        y, x, q = simulate_scsv(
            T=T,
            mu=TRUE_MU,
            phi=phi_true,
            alpha=alpha_true,
            gamma=gamma_true,
            r=r_true,
            rng=rng,
        )

        fit_joint = fit_joint_particle_em(
            y,
            N=N_PARTICLES_FIT,
            seed=seed + 1000,
        )

        fit_g0 = fit_particle_gamma0(
            y,
            N=N_PARTICLES_FIT,
            seed=seed + 2000,
        )

        mu_hat = float(get_param(fit_joint, "mu", "mu_est"))
        phi_hat = float(get_param(fit_joint, "phi", "phi_est"))
        alpha_hat = float(get_param(fit_joint, "alpha", "alpha_est"))
        gamma_hat = float(get_param(fit_joint, "gamma", "gamma_est"))
        r_hat = float(get_param(fit_joint, "r", "r_est"))

        g0_mu_hat = float(get_param(fit_g0, "mu", "mu_est"))
        g0_phi_hat = float(get_param(fit_g0, "phi", "phi_est"))
        g0_alpha_hat = float(get_param(fit_g0, "alpha", "alpha_est"))
        g0_r_hat = float(get_param(fit_g0, "r", "r_est"))

        joint_loglik = repeated_loglik_scsv(
            y=y,
            mu=mu_hat,
            phi=phi_hat,
            alpha=alpha_hat,
            gamma=gamma_hat,
            r=r_hat,
            base_seed=seed + 3000,
        )

        gamma0_loglik = repeated_loglik_gamma0(
            y=y,
            mu=g0_mu_hat,
            phi=g0_phi_hat,
            alpha=g0_alpha_hat,
            r=g0_r_hat,
            base_seed=seed + 4000,
        )

        joint_nll = -joint_loglik
        gamma0_nll = -gamma0_loglik

        bic_joint = K_PROPOSED * np.log(T) + 2.0 * joint_nll
        bic_gamma0 = K_GAMMA0 * np.log(T) + 2.0 * gamma0_nll

        delta_nll_gamma0 = gamma0_nll - joint_nll
        delta_bic_gamma0 = bic_gamma0 - bic_joint

        row.update({
            "success": True,

            "T": T,
            "mu_true": TRUE_MU,
            "phi_true": phi_true,
            "alpha_true": alpha_true,
            "gamma_true": gamma_true,
            "r_true": r_true,

            "mu_est": mu_hat,
            "phi_est": phi_hat,
            "alpha_est": alpha_hat,
            "gamma_est": gamma_hat,
            "r_est": r_hat,

            "gamma0_mu_est": g0_mu_hat,
            "gamma0_phi_est": g0_phi_hat,
            "gamma0_alpha_est": g0_alpha_hat,
            "gamma0_r_est": g0_r_hat,

            "joint_nll": joint_nll,
            "gamma0_nll": gamma0_nll,
            "bic_joint": bic_joint,
            "bic_gamma0": bic_gamma0,

            "delta_nll_gamma0": delta_nll_gamma0,
            "delta_bic_gamma0": delta_bic_gamma0,

            "joint_wins_nll": delta_nll_gamma0 > 0,
            "joint_wins_bic": delta_bic_gamma0 > 0,

            "q_center_true": np.exp(alpha_true),
            "q_dist_1_true": np.exp(alpha_true + gamma_true),
            "q_multiplier_1_true": np.exp(gamma_true),

            "q_center_est": np.exp(np.clip(alpha_hat, -20, 10)),
            "q_dist_1_est": np.exp(np.clip(alpha_hat + gamma_hat, -20, 10)),
            "q_multiplier_1_est": np.exp(np.clip(gamma_hat, -20, 10)),

            "log_r_est": np.log(max(r_hat, 1e-12)),
            "log_q_center_est": alpha_hat,
            "log_q_dist_1_est": alpha_hat + gamma_hat,

            "gamma_error": gamma_hat - gamma_true,
            "alpha_error": alpha_hat - alpha_true,
            "phi_error": phi_hat - phi_true,
            "r_error": r_hat - r_true,

            "abs_gamma_error": abs(gamma_hat - gamma_true),
            "abs_alpha_error": abs(alpha_hat - alpha_true),
            "abs_phi_error": abs(phi_hat - phi_true),
            "abs_r_error": abs(r_hat - r_true),

            "true_positive_coupling": gamma_true > 0,
            "true_ictal_scale_coupling": gamma_true >= 0.30,

            "gamma_regime": (
                "null" if np.isclose(gamma_true, 0.0)
                else "ictal_scale" if gamma_true <= 0.30
                else "strong"
            ),
            "alpha_regime": (
                "low" if alpha_true <= -4.0
                else "high" if alpha_true >= -2.0
                else "mid"
            ),
            "phi_regime": (
                "low" if phi_true <= 0.3
                else "high" if phi_true >= 0.5
                else "mid"
            ),
            "r_regime": (
                "low" if r_true <= 0.01
                else "high" if r_true >= 0.10
                else "mid"
            ),

            "mean_x": np.mean(x),
            "sd_x": np.std(x),
            "mean_q": np.mean(q),
            "median_q": np.median(q),
            "max_q": np.max(q),

            "N_particles_fit": N_PARTICLES_FIT,
            "N_particles_eval": N_PARTICLES_EVAL,
            "N_eval_repeats": N_EVAL_REPEATS,

            "runtime_sec": time.time() - t0,
        })

    except Exception as e:
        row["success"] = False
        row["error"] = repr(e)
        row["runtime_sec"] = time.time() - t0

    return row


# ============================================================
# Summaries
# ============================================================

def summarize_results(df):
    df_success = df[df["success"] == True].copy()

    if len(df_success) == 0:
        return pd.DataFrame(), pd.DataFrame()

    group_cols = ["alpha_true", "gamma_true", "phi_true", "r_true"]

    for name in ["gamma", "alpha", "phi", "r"]:
        df_success[f"{name}_error"] = df_success[f"{name}_est"] - df_success[f"{name}_true"]
        df_success[f"abs_{name}_error"] = np.abs(df_success[f"{name}_error"])

    summary = (
        df_success
        .groupby(group_cols)
        .agg(
            n=("gamma_est", "size"),

            median_gamma_hat=("gamma_est", "median"),
            mean_gamma_hat=("gamma_est", "mean"),
            sd_gamma_hat=("gamma_est", "std"),
            median_gamma_error=("gamma_error", "median"),
            median_abs_gamma_error=("abs_gamma_error", "median"),

            median_alpha_hat=("alpha_est", "median"),
            mean_alpha_hat=("alpha_est", "mean"),
            sd_alpha_hat=("alpha_est", "std"),
            median_alpha_error=("alpha_error", "median"),
            median_abs_alpha_error=("abs_alpha_error", "median"),

            median_phi_hat=("phi_est", "median"),
            mean_phi_hat=("phi_est", "mean"),
            sd_phi_hat=("phi_est", "std"),
            median_phi_error=("phi_error", "median"),
            median_abs_phi_error=("abs_phi_error", "median"),

            median_r_hat=("r_est", "median"),
            mean_r_hat=("r_est", "mean"),
            sd_r_hat=("r_est", "std"),
            median_r_error=("r_error", "median"),
            median_abs_r_error=("abs_r_error", "median"),

            median_q_center_est=("q_center_est", "median"),
            median_q_dist_1_est=("q_dist_1_est", "median"),
            median_q_multiplier_1_est=("q_multiplier_1_est", "median"),

            median_delta_nll=("delta_nll_gamma0", "median"),
            mean_delta_nll=("delta_nll_gamma0", "mean"),
            win_rate_nll=("joint_wins_nll", "mean"),

            median_delta_bic=("delta_bic_gamma0", "median"),
            mean_delta_bic=("delta_bic_gamma0", "mean"),
            win_rate_bic=("joint_wins_bic", "mean"),

            median_runtime_sec=("runtime_sec", "median"),
        )
        .reset_index()
    )

    null = df_success[np.isclose(df_success["gamma_true"], 0.0)].copy()
    detection = pd.DataFrame()

    if len(null) > 0:
        # Null thresholds conditional on alpha, phi, and r.
        threshold_cols = ["alpha_true", "phi_true", "r_true"]

        thresholds = (
            null
            .groupby(threshold_cols)["delta_bic_gamma0"]
            .quantile(0.95)
            .rename("null_calibrated_delta_bic_threshold")
            .reset_index()
        )

        df_det = df_success.merge(thresholds, on=threshold_cols, how="left")
        df_det["detected"] = (
            df_det["delta_bic_gamma0"]
            > df_det["null_calibrated_delta_bic_threshold"]
        )

        detection = (
            df_det
            .groupby(group_cols)
            .agg(
                n=("detected", "size"),
                null_calibrated_delta_bic_threshold=(
                    "null_calibrated_delta_bic_threshold",
                    "first",
                ),
                detection_rate=("detected", "mean"),
                median_delta_bic=("delta_bic_gamma0", "median"),
                median_delta_nll=("delta_nll_gamma0", "median"),
                median_gamma_est=("gamma_est", "median"),
                median_alpha_est=("alpha_est", "median"),
                median_phi_est=("phi_est", "median"),
                median_r_est=("r_est", "median"),
            )
            .reset_index()
        )

    return summary, detection


def save_reliability_features(df):
    df_success = df[df["success"] == True].copy()

    reliability_cols = [
        "alpha_true", "gamma_true", "phi_true", "r_true",
        "rep", "seed",

        "gamma_regime", "alpha_regime", "phi_regime", "r_regime",
        "true_positive_coupling", "true_ictal_scale_coupling",

        "mu_est", "phi_est", "alpha_est", "gamma_est", "r_est",

        "q_center_est", "q_dist_1_est", "q_multiplier_1_est",
        "log_r_est", "log_q_center_est", "log_q_dist_1_est",

        "delta_nll_gamma0", "delta_bic_gamma0",
        "joint_wins_nll", "joint_wins_bic",

        "gamma_error", "alpha_error", "phi_error", "r_error",
        "abs_gamma_error", "abs_alpha_error", "abs_phi_error", "abs_r_error",

        "runtime_sec",
    ]

    reliability_cols = [c for c in reliability_cols if c in df_success.columns]
    df_success[reliability_cols].to_csv(RELIABILITY_OUTFILE, index=False)


# ============================================================
# Main
# ============================================================

def main():
    warnings.filterwarnings("ignore")

    # ------------------------------------------------------------
    # Build deterministic task list
    # ------------------------------------------------------------
    tasks = []

    for alpha in ALPHA_VALUES:
        for gamma in GAMMA_VALUES:
            for phi in PHI_VALUES:
                for r in R_VALUES:
                    for rep in range(N_REPS_PER_CONDITION):
                        seed = abs(
                            hash((RANDOM_SEED, alpha, gamma, phi, r, rep))
                        ) % (2**31 - 1)

                        tasks.append((alpha, gamma, phi, r, rep, seed))

    print("Fair alpha-gamma-phi-r simulation study")
    print("---------------------------------------")
    print(f"T = {T}")
    print(f"True mu = {TRUE_MU}")
    print(f"Alpha values = {ALPHA_VALUES}")
    print(f"Gamma values = {GAMMA_VALUES}")
    print(f"Phi values = {PHI_VALUES}")
    print(f"R values = {R_VALUES}")
    print(f"Reps per condition = {N_REPS_PER_CONDITION}")
    print(f"Fit particles = {N_PARTICLES_FIT}")
    print(f"Eval particles = {N_PARTICLES_EVAL}")
    print(f"Eval repeats = {N_EVAL_REPEATS}")
    print(f"N jobs = {N_JOBS}")
    print(f"Batch size = {BATCH_SIZE}")
    print(f"Total planned tasks = {len(tasks)}")
    print(f"Output = {OUTFILE}")

    # ------------------------------------------------------------
    # Resume from existing output if present
    # ------------------------------------------------------------
    outfile = Path(OUTFILE)

    completed_keys = set()
    all_batches = []

    key_cols = ["alpha_true", "gamma_true", "phi_true", "r_true", "rep"]

    if outfile.exists():
        old = pd.read_csv(outfile)

        if len(old) > 0:
            old_unique = old.drop_duplicates(subset=key_cols)

            completed_keys = set(
                tuple(row) for row in old_unique[key_cols].to_numpy()
            )

            all_batches.append(old)

            print(f"\nFound existing output file: {OUTFILE}")
            print(f"Existing rows: {len(old)}")
            print(f"Completed/attempted unique tasks: {len(completed_keys)}")
    else:
        print("\nNo existing output file found. Starting fresh.")

    # Skip tasks already written
    tasks = [
        task for task in tasks
        if (task[0], task[1], task[2], task[3], task[4]) not in completed_keys
    ]

    print(f"Remaining tasks to run: {len(tasks)}")

    if len(tasks) == 0:
        print("\nNo remaining tasks. Rebuilding summaries from existing output.")

        df = pd.concat(all_batches, ignore_index=True)

        summary, detection = summarize_results(df)

        summary.to_csv(SUMMARY_OUTFILE, index=False)
        detection.to_csv(DETECTION_OUTFILE, index=False)
        save_reliability_features(df)

        print("\nSaved files:")
        print(f"  {OUTFILE}")
        print(f"  {SUMMARY_OUTFILE}")
        print(f"  {DETECTION_OUTFILE}")
        print(f"  {RELIABILITY_OUTFILE}")
        return

    # ------------------------------------------------------------
    # Run remaining tasks
    # ------------------------------------------------------------
    for start in range(0, len(tasks), BATCH_SIZE):
        batch = tasks[start:start + BATCH_SIZE]

        print(
            f"\nRunning remaining batch {start + 1} "
            f"to {start + len(batch)} of {len(tasks)}"
        )

        rows = Parallel(n_jobs=N_JOBS, verbose=100)(
            delayed(fit_one_sim)(task) for task in batch
        )

        batch_df = pd.DataFrame(rows)
        all_batches.append(batch_df)

        write_header = not outfile.exists()
        batch_df.to_csv(outfile, mode="a", header=write_header, index=False)

        print(f"Finished {start + len(batch)}/{len(tasks)} remaining tasks")
        print(batch_df["success"].value_counts(dropna=False))

        if "runtime_sec" in batch_df.columns:
            print("Median batch runtime per task:", batch_df["runtime_sec"].median())

    # ------------------------------------------------------------
    # Build final summaries
    # ------------------------------------------------------------
    df = pd.concat(all_batches, ignore_index=True)

    summary, detection = summarize_results(df)

    summary.to_csv(SUMMARY_OUTFILE, index=False)
    detection.to_csv(DETECTION_OUTFILE, index=False)
    save_reliability_features(df)

    print("\nSimulation complete.")
    print("\nSuccess counts:")
    print(df["success"].value_counts(dropna=False))

    print("\nSummary by true alpha, gamma, phi, and r:")
    print(summary)

    print("\nConditional null-calibrated detection summary:")
    print(detection)

    print("\nSaved files:")
    print(f"  {OUTFILE}")
    print(f"  {SUMMARY_OUTFILE}")
    print(f"  {DETECTION_OUTFILE}")
    print(f"  {RELIABILITY_OUTFILE}")
if __name__ == "__main__":
    main()