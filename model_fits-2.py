import numpy as np
import pandas as pd

from scipy.special import logsumexp
from scipy.optimize import minimize
import time
def systematic_resample(weights, rng):
    N = len(weights)
    positions = (rng.random() + np.arange(N)) / N
    indexes = np.zeros(N, dtype=int)

    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0

    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    return indexes


def normal_logpdf(y, mean, var):
    var = np.maximum(var, 1e-12)
    return -0.5 * (np.log(2 * np.pi * var) + ((y - mean) ** 2) / var)


def timed_fit(method_name, fit_func, y, **kwargs):
    start = time.time()

    try:
        out = dict(fit_func(y, **kwargs))
        out["success"] = out.get("success", True)
        out["error"] = out.get("error", "")
    except Exception as e:
        out = {
            "method": method_name,
            "success": False,
            "error": str(e),
            "mu_est": np.nan,
            "phi_est": np.nan,
            "alpha_est": np.nan,
            "gamma_est": np.nan,
            "r_est": np.nan,
            "nll": np.nan,
        }

    out["method"] = method_name
    out["runtime_sec"] = time.time() - start
    return out


def fit_particle_gamma0(y, seed=123, N=500, maxiter=80):
    y = np.asarray(y)
    T = len(y)

    def unpack(theta):
        mu = theta[0]
        phi = np.tanh(theta[1])
        alpha = theta[2]
        r = np.exp(theta[3])
        return mu, phi, alpha, r

    def particle_nll(theta):
        rng = np.random.default_rng(seed)
        mu, phi, alpha, r = unpack(theta)

        q = np.exp(np.clip(alpha, -20, 10))
        x = rng.normal(mu, np.sqrt(q / max(1e-6, 1 - phi ** 2)), size=N)

        loglik = 0.0

        for t in range(T):
            if t > 0:
                x = mu + phi * (x - mu) + rng.normal(0, np.sqrt(q), size=N)

            logw = normal_logpdf(y[t], x, r)
            m = np.max(logw)
            loglik += m + np.log(np.mean(np.exp(logw - m)))

            w = np.exp(logw - logsumexp(logw))
            idx = systematic_resample(w, rng)
            x = x[idx]

        return -loglik

    theta0 = np.array([
        np.mean(y),
        np.arctanh(0.5),
        np.log(np.var(np.diff(y)) + 1e-6),
        np.log(0.2 * np.var(y) + 1e-6),
    ])

    opt = minimize(
        particle_nll,
        theta0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "disp": False},
    )

    mu, phi, alpha, r = unpack(opt.x)

    return {
        "method": "particle_gamma0",
        "mu_est": mu,
        "phi_est": phi,
        "alpha_est": alpha,
        "gamma_est": 0.0,
        "r_est": r,
        "nll": opt.fun,
        "success": bool(opt.success),
        "error": "" if opt.success else str(opt.message),
    }


def fit_observed_hetero_proxy(y, seed=123, N=500, maxiter=80):
    y = np.asarray(y)
    T = len(y)

    z = np.zeros(T)
    z[1:] = np.abs(y[:-1] - np.mean(y))

    if np.std(z) > 1e-8:
        z = (z - np.mean(z)) / np.std(z)

    def unpack(theta):
        mu = theta[0]
        phi = np.tanh(theta[1])
        alpha = theta[2]
        beta = theta[3]
        r = np.exp(theta[4])
        return mu, phi, alpha, beta, r

    def particle_nll(theta):
        rng = np.random.default_rng(seed)
        mu, phi, alpha, beta, r = unpack(theta)

        q0 = np.exp(np.clip(alpha, -20, 10))
        x = rng.normal(mu, np.sqrt(q0 / max(1e-6, 1 - phi ** 2)), size=N)

        loglik = 0.0

        for t in range(T):
            if t > 0:
                q_t = np.exp(np.clip(alpha + beta * z[t], -20, 10))
                x = mu + phi * (x - mu) + rng.normal(0, np.sqrt(q_t), size=N)

            logw = normal_logpdf(y[t], x, r)
            m = np.max(logw)
            loglik += m + np.log(np.mean(np.exp(logw - m)))

            w = np.exp(logw - logsumexp(logw))
            idx = systematic_resample(w, rng)
            x = x[idx]

        return -loglik

    theta0 = np.array([
        np.mean(y),
        np.arctanh(0.5),
        np.log(np.var(np.diff(y)) + 1e-6),
        0.0,
        np.log(0.2 * np.var(y) + 1e-6),
    ])

    opt = minimize(
        particle_nll,
        theta0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "disp": False},
    )

    mu, phi, alpha, beta, r = unpack(opt.x)

    return {
        "method": "observed_hetero_proxy",
        "mu_est": mu,
        "phi_est": phi,
        "alpha_est": alpha,
        "beta_est": beta,
        "gamma_est": np.nan,
        "r_est": r,
        "nll": opt.fun,
        "success": bool(opt.success),
        "error": "" if opt.success else str(opt.message),
    }


def fit_sv_state_space(y, seed=123, N=500, maxiter=100):
    y = np.asarray(y)
    T = len(y)

    def unpack(theta):
        mu = theta[0]
        phi = np.tanh(theta[1])
        a = theta[2]
        rho = np.tanh(theta[3])
        sigma_h = np.exp(theta[4])
        r = np.exp(theta[5])
        return mu, phi, a, rho, sigma_h, r

    def particle_nll(theta):
        rng = np.random.default_rng(seed)
        mu, phi, a, rho, sigma_h, r = unpack(theta)

        h_sd0 = sigma_h / np.sqrt(max(1e-6, 1 - rho ** 2))
        h = rng.normal(a, h_sd0, size=N)
        q = np.exp(np.clip(h, -20, 10))
        x = rng.normal(mu, np.sqrt(q / max(1e-6, 1 - phi ** 2)), size=N)

        loglik = 0.0

        for t in range(T):
            if t > 0:
                h = a + rho * (h - a) + rng.normal(0, sigma_h, size=N)
                h = np.clip(h, -20, 10)
                q = np.exp(h)

                x = mu + phi * (x - mu) + rng.normal(0, np.sqrt(q), size=N)

            logw = normal_logpdf(y[t], x, r)
            m = np.max(logw)
            loglik += m + np.log(np.mean(np.exp(logw - m)))

            w = np.exp(logw - logsumexp(logw))
            idx = systematic_resample(w, rng)
            x = x[idx]
            h = h[idx]

        return -loglik

    theta0 = np.array([
        np.mean(y),
        np.arctanh(0.5),
        np.log(np.var(np.diff(y)) + 1e-6),
        np.arctanh(0.8),
        np.log(0.2),
        np.log(0.2 * np.var(y) + 1e-6),
    ])

    opt = minimize(
        particle_nll,
        theta0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "disp": False},
    )

    mu, phi, a, rho, sigma_h, r = unpack(opt.x)

    return {
        "method": "sv_state_space",
        "mu_est": mu,
        "phi_est": phi,
        "alpha_est": a,
        "gamma_est": np.nan,
        "r_est": r,
        "rho_h_est": rho,
        "sigma_h_est": sigma_h,
        "nll": opt.fun,
        "success": bool(opt.success),
        "error": "" if opt.success else str(opt.message),
    }


def particle_filter_scsv(y, mu, phi, alpha, gamma, r, N=1200, seed=123):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    T = len(y)

    particles = np.zeros((T, N))
    ancestors = np.zeros((T, N), dtype=int)
    log_weights = np.zeros((T, N))

    q0 = np.exp(np.clip(alpha, -20, 10))
    particles[0] = rng.normal(mu, np.sqrt(q0 / max(1e-6, 1 - phi ** 2)), N)

    log_w0 = normal_logpdf(y[0], particles[0], r)
    m0 = np.max(log_w0)
    loglik = m0 + np.log(np.mean(np.exp(log_w0 - m0)))

    log_weights[0] = log_w0 - logsumexp(log_w0)

    for t in range(1, T):
        weights = np.exp(log_weights[t - 1])
        idx = systematic_resample(weights, rng)
        ancestors[t] = idx

        x_prev = particles[t - 1, idx]

        log_q = alpha + gamma * np.abs(x_prev - mu)
        q = np.exp(np.clip(log_q, -20, 10))

        x_mean = mu + phi * (x_prev - mu)
        particles[t] = rng.normal(x_mean, np.sqrt(q))

        log_w = normal_logpdf(y[t], particles[t], r)

        m = np.max(log_w)
        loglik += m + np.log(np.mean(np.exp(log_w - m)))

        log_weights[t] = log_w - logsumexp(log_w)

    return {
        "particles": particles,
        "ancestors": ancestors,
        "log_weights": log_weights,
        "loglik": loglik,
    }


def sample_particle_paths(pf, M=200, seed=456):
    rng = np.random.default_rng(seed)

    particles = pf["particles"]
    ancestors = pf["ancestors"]

    T, N = particles.shape
    paths = np.zeros((M, T))

    w = np.exp(pf["log_weights"][-1])

    for m in range(M):
        idx = rng.choice(N, p=w)
        paths[m, -1] = particles[-1, idx]

        for t in range(T - 1, 0, -1):
            idx = ancestors[t, idx]
            paths[m, t - 1] = particles[t - 1, idx]

    return paths


def estimate_joint_params_from_paths(paths, y):
    paths = np.asarray(paths)
    y = np.asarray(y)

    mu0 = np.mean(y)
    phi0 = 0.5
    alpha0 = np.log(np.var(np.diff(y)) + 1e-6)
    gamma0 = 0.3
    log_r0 = np.log(np.var(y - np.mean(y)) * 0.1 + 1e-6)

    bounds = [
        (None, None),
        (None, None),
        (-10, 5),
        (-5, 5),
        (-20, 5),
    ]

    inits = [
        [mu0, np.arctanh(0.3), alpha0, 0.1, log_r0],
        [mu0, np.arctanh(0.6), alpha0, 0.5, log_r0],
        [mu0, np.arctanh(0.8), alpha0, 1.0, log_r0],
        [mu0, np.arctanh(0.6), alpha0 - 0.5, 0.8, log_r0],
        [mu0, np.arctanh(0.6), alpha0 + 0.5, 0.3, log_r0],
    ]

    def loss(p):
        mu, phi_raw, alpha, gamma, log_r = p

        phi = np.tanh(phi_raw)
        r = np.exp(log_r)

        total_loss = 0.0

        for x in paths:
            x_prev = x[:-1]
            x_next = x[1:]

            z = np.abs(x_prev - mu)
            log_q = np.clip(alpha + gamma * z, -20, 10)
            q = np.exp(log_q)

            pred = mu + phi * (x_prev - mu)
            state_resid = x_next - pred

            state_nll = 0.5 * np.sum(log_q + (state_resid ** 2) / q)

            obs_resid = y - x
            obs_nll = 0.5 * np.sum(np.log(r) + (obs_resid ** 2) / r)

            total_loss += state_nll + obs_nll

        return total_loss / paths.shape[0]

    fits = [
        minimize(loss, init, method="L-BFGS-B", bounds=bounds)
        for init in inits
    ]

    fit = min(fits, key=lambda f: f.fun)

    mu_hat, phi_raw_hat, alpha_hat, gamma_hat, log_r_hat = fit.x

    return {
        "mu_est": mu_hat,
        "phi_est": np.tanh(phi_raw_hat),
        "alpha_est": alpha_hat,
        "gamma_est": gamma_hat,
        "r_est": np.exp(log_r_hat),
        "fun": fit.fun,
        "success": bool(fit.success),
    }


def particle_em_joint_recovery(y, max_iter=12, N=1200, M=200, seed=124, damping=0.8):
    y = np.asarray(y)

    mu = np.mean(y)
    phi = 0.6
    alpha = np.log(np.var(np.diff(y)) + 1e-6)
    gamma = 0.3
    r = np.var(y) * 0.1 + 1e-6

    hist = []

    for i in range(max_iter):
        pf = particle_filter_scsv(
            y=y,
            mu=mu,
            phi=phi,
            alpha=alpha,
            gamma=gamma,
            r=r,
            N=N,
            seed=seed + i,
        )

        paths = sample_particle_paths(
            pf,
            M=M,
            seed=seed + 10_000 + i,
        )

        fit = estimate_joint_params_from_paths(paths, y)

        mu = (1 - damping) * mu + damping * fit["mu_est"]
        phi = (1 - damping) * phi + damping * fit["phi_est"]
        alpha = (1 - damping) * alpha + damping * fit["alpha_est"]
        gamma = (1 - damping) * gamma + damping * fit["gamma_est"]
        r = (1 - damping) * r + damping * fit["r_est"]

        hist.append({
            "iter": i,
            "mu": mu,
            "phi": phi,
            "alpha": alpha,
            "gamma": gamma,
            "r": r,
            "m_step_fun": fit["fun"],
            "pf_nll": -pf["loglik"],
            "success": fit["success"],
        })

    return {
        "mu_est": mu,
        "phi_est": phi,
        "alpha_est": alpha,
        "gamma_est": gamma,
        "r_est": r,
        "history": pd.DataFrame(hist),
    }


def fit_joint_particle_em(y,N, seed=123):
    fit = particle_em_joint_recovery(
        y=y,
        max_iter=12,
        N=N,
        M=200,
        damping=0.8,
        seed=seed,
    )

    pf_final = particle_filter_scsv(
        y=y,
        mu=fit["mu_est"],
        phi=fit["phi_est"],
        alpha=fit["alpha_est"],
        gamma=fit["gamma_est"],
        r=fit["r_est"],
        N=N,
        seed=seed + 999,
    )

    return {
        "method": "joint_particle_em",
        "mu_est": fit["mu_est"],
        "phi_est": fit["phi_est"],
        "alpha_est": fit["alpha_est"],
        "gamma_est": fit["gamma_est"],
        "r_est": fit["r_est"],
        "nll": -pf_final["loglik"],
        "success": True,
        "error": "",
    }
