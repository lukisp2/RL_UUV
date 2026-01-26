# -*- coding: utf-8 -*-
"""
uuv_compare_experiment_tables.py
================================

End-to-end evaluation runner + paper outputs for the UUV Doppler-only
relative localization + formation environment.

✅ What it does (single run)
  - Runs N evaluation episodes for ALL controllers:
      (A) RL policy (SB3 SAC/PPO) loaded from .zip (optionally using VecNormalize stats)
      (B) Baselines: PID, PID+Excitation, Random
  - Logs step-by-step traces to CSV/NPZ for each episode and controller
  - Computes FIM/CRLB online from TRUTH (requires env to expose truth in info via cfg.log_truth_diagnostics=True)
  - Produces publication-friendly summary tables (one row per controller)
  - Produces comparison plots (one figure per metric) with ALL controllers on the same axes
  - Writes a ready-to-\\input LaTeX "plot grid" table containing the key comparison plots

Output structure:
    out_dir/
      rl/
        traces/ep001.csv, ep001.npz, ...
        summary.json, summary.csv, meta.json
      baseline_pid/
        ...
      baseline_pid_exc/
        ...
      baseline_random/
        ...
      compare/
        compare_summary.csv                 (wide, per-episode)
        plot_*.png                         (multi-controller)
        paper_table.csv / .tex / .md / .json
        plot_grid.tex                      (LaTeX table* with the plots)

✅ Requirements
  pip install numpy matplotlib stable-baselines3[extra] gymnasium

IMPORTANT:
  - This script IMPORTS your environment from a python module (default: uuv_rl_patched.py),
    expecting that module defines:
        - class UUVRelPosBankEnv
        - dataclass UUVConfig

Example:
  python uuv_compare_experiment_tables.py run \
    --algo sac \
    --model models/best_model.zip \
    --models-dir models \
    --baseline all \
    --episodes 30 \
    --seed 42 \
    --out-dir results_cmp \
    --fim-window 30
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Small helpers
# ---------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def wrap360(a_deg: float) -> float:
    a = float(a_deg) % 360.0
    if a < 0:
        a += 360.0
    return a

def wrap180(a_deg: float) -> float:
    """Wrap angle to [-180, 180]."""
    a = (float(a_deg) + 180.0) % 360.0 - 180.0
    return a

def deg2rad(a: float) -> float:
    return float(a) * math.pi / 180.0

def heading_from_cos_sin(c: float, s: float) -> float:
    return wrap360(math.degrees(math.atan2(float(s), float(c))))

def vel_vec2(speed: float, heading_deg: float) -> np.ndarray:
    h = deg2rad(heading_deg)
    return np.array([float(speed) * math.cos(h), float(speed) * math.sin(h)], dtype=float)

def rot90(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(2)
    return np.array([-v[1], v[0]], dtype=float)

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def timestamp() -> str:
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_log10(x: float, floor: float = 1e-16) -> float:
    x = float(x)
    if not math.isfinite(x):
        return float("nan")
    return math.log10(max(floor, x))


# ---------------------------
# VecNormalize stats loader (manual normalization)
# ---------------------------

class ObsNormalizer:
    """
    Minimal observation normalizer compatible with SB3 VecNormalize(obs).
    Loads obs_rms.mean/var and applies:
        (obs - mean) / sqrt(var + eps), then clip to [-clip_obs, clip_obs]
    """
    def __init__(self, mean: np.ndarray, var: np.ndarray, eps: float, clip_obs: float) -> None:
        self.mean = np.asarray(mean, dtype=float)
        self.var = np.asarray(var, dtype=float)
        self.eps = float(eps)
        self.clip_obs = float(clip_obs)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=float)
        x = (x - self.mean) / np.sqrt(self.var + self.eps)
        x = np.clip(x, -self.clip_obs, self.clip_obs)
        return x.astype(np.float32)

def try_load_vecnormalize(models_dir: str) -> Optional[ObsNormalizer]:
    """
    Attempts to load SB3 VecNormalize stats from models_dir/vecnormalize.pkl.
    Returns ObsNormalizer or None.
    """
    vn_path = os.path.join(models_dir, "vecnormalize.pkl")
    if not os.path.exists(vn_path):
        return None

    try:
        with open(vn_path, "rb") as f:
            vn = pickle.load(f)
        mean = np.asarray(vn.obs_rms.mean, dtype=float)
        var = np.asarray(vn.obs_rms.var, dtype=float)
        eps = float(getattr(vn, "epsilon", 1e-8))
        clip_obs = float(getattr(vn, "clip_obs", 10.0))
        return ObsNormalizer(mean=mean, var=var, eps=eps, clip_obs=clip_obs)
    except Exception as e:
        print("[WARN] Could not load vecnormalize.pkl for manual obs normalization:", e)
        return None


# ---------------------------
# Baseline controllers
# ---------------------------

def baseline_action_from_obs_raw(
    obs_raw: np.ndarray,
    cfg: Any,
    controller: str,
    kp: float,
    kexc: float,
    std_trigger_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Baseline controller uses ONLY estimator outputs encoded in the raw observation
    (same information RL gets).

    obs layout (from your env docstring; keep indices consistent with env):
      0-1: e_mix = (r_mix - r_des) / pos_scale
      2-3: r_mix / pos_scale
      4-5: std_x, std_y / std_scale
      7:   speed_F / vel_scale
      8-9: cos(hF), sin(hF)
      11:  leader_speed / leader_speed_max
      12-13: cos(hL), sin(hL)

    Returns action in [-1,1]^2: [a_speed, a_turn]
    """
    obs = np.asarray(obs_raw, dtype=float).reshape(-1)
    assert obs.size >= 14, f"Expected obs dim 14+, got {obs.size}"

    if controller == "random":
        return rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)

    pos_scale = float(cfg.pos_scale)
    std_scale = float(cfg.std_scale)
    vel_scale = float(cfg.vel_scale)

    # Decode from obs
    e_mix = obs[0:2] * pos_scale
    r_mix = obs[2:4] * pos_scale

    std_x = abs(float(obs[4]) * std_scale)
    std_y = abs(float(obs[5]) * std_scale)
    std_max = max(std_x, std_y)

    speed_F = float(obs[7]) * vel_scale
    heading_F = heading_from_cos_sin(obs[8], obs[9])

    leader_speed = float(obs[11]) * float(cfg.leader_speed_max)
    heading_L = heading_from_cos_sin(obs[12], obs[13])

    vL = vel_vec2(leader_speed, heading_L)

    # Velocity-space formation hold:
    # dr/dt = vL - vF,  e = r - r_des  => choose vF = vL + kp*e  => de/dt = -kp*e
    vF_cmd = vL + float(kp) * e_mix

    # Optional excitation to increase Doppler information when uncertainty is high.
    if controller == "pid_exc":
        trigger = float(std_trigger_m) if float(std_trigger_m) > 0.0 else float(cfg.tol_std_hard)
        if std_max > trigger:
            rho = float(np.linalg.norm(r_mix))
            if rho > 1e-6:
                rhat = r_mix / rho
                tang = rot90(rhat)  # tangential unit direction

                # Saturating scaling with std_excess
                std_excess = float(std_max - trigger)
                scale = std_excess / (std_excess + trigger + 1e-9)

                # Excitation velocity magnitude [m/s]
                v_exc = float(kexc) * scale
                vF_cmd = vF_cmd + v_exc * tang

                # Keep-away if too close (soft)
                rho_soft = float(cfg.rho_soft_min)
                if rho < rho_soft:
                    push = (rho_soft - rho) / max(rho_soft, 1e-6)
                    vF_cmd = vF_cmd - (0.5 * float(kexc) * push) * rhat

    # Convert desired velocity vector to desired speed + heading
    vF_cmd_speed = float(np.linalg.norm(vF_cmd))
    if vF_cmd_speed < 1e-6:
        vF_cmd_speed = float(cfg.f_min_speed)

    vF_cmd_speed = float(clamp(vF_cmd_speed, float(cfg.f_min_speed), float(cfg.f_max_speed)))
    vF_cmd_heading = wrap360(math.degrees(math.atan2(vF_cmd[1], vF_cmd[0])))

    # Map (desired speed, desired heading) to action in [-1,1]
    action_dt = float(cfg.action_dt)

    acc_from_step = float(cfg.rl_speed_delta_per_step) / max(action_dt, 1e-6)
    acc_limit = min(float(cfg.max_accel_m_s2), acc_from_step)

    turn_rate_from_step = float(cfg.rl_turn_per_step_deg) / max(action_dt, 1e-6)
    turn_rate_limit = min(float(cfg.max_turn_rate_deg_s), turn_rate_from_step)

    d_heading = wrap180(vF_cmd_heading - heading_F)
    a_turn = d_heading / max(turn_rate_limit * action_dt, 1e-6)
    a_turn = clamp(a_turn, -1.0, 1.0)

    d_speed = (vF_cmd_speed - speed_F)
    a_speed = d_speed / max(acc_limit * action_dt, 1e-6)
    a_speed = clamp(a_speed, -1.0, 1.0)

    return np.array([a_speed, a_turn], dtype=np.float32)


# ---------------------------
# FIM / CRLB tracker
# ---------------------------

def doppler_H(r: np.ndarray, v_rel: np.ndarray) -> Optional[np.ndarray]:
    """
    For s = - (r/||r||) · v_rel, Jacobian w.r.t r is:
        H = -(v_rel - proj*rhat)/rho
    """
    r = np.asarray(r, dtype=float).reshape(2)
    v_rel = np.asarray(v_rel, dtype=float).reshape(2)
    rho = float(np.linalg.norm(r))
    if rho <= 1e-12:
        return None
    rhat = r / rho
    proj = float(np.dot(rhat, v_rel))
    H = -(v_rel - proj * rhat) / rho
    return H

def v_perp_mag(r: np.ndarray, v_rel: np.ndarray) -> float:
    r = np.asarray(r, dtype=float).reshape(2)
    v_rel = np.asarray(v_rel, dtype=float).reshape(2)
    rho = float(np.linalg.norm(r))
    if rho <= 1e-12:
        return float("nan")
    rhat = r / rho
    proj = float(np.dot(rhat, v_rel))
    vnorm = float(np.linalg.norm(v_rel))
    return math.sqrt(max(0.0, vnorm * vnorm - proj * proj))

class FIMTracker:
    """
    Tracks total and windowed FIM and CRLB.

    FIM increment per Doppler measurement:
        I += H^T R^{-1} H,   R = sigma_s^2

    In practice per RL-step we may have M Doppler samples; we multiply the increment by M.
    """
    def __init__(
        self,
        sigma_s: float,
        window_s: float,
        reg_eps: float = 1e-9,
        use_gating: bool = True,
        rho_min: float = 20.0,
        v_min: float = 0.05,
        v_perp_min: float = 0.25,
    ) -> None:
        self.sigma_s = float(max(1e-9, sigma_s))
        self.Rinv = 1.0 / (self.sigma_s ** 2)
        self.window_s = float(max(0.0, window_s))
        self.reg_eps = float(max(0.0, reg_eps))
        self.use_gating = bool(use_gating)

        self.rho_min = float(rho_min)
        self.v_min = float(v_min)
        self.v_perp_min = float(v_perp_min)

        self.I_tot = np.zeros((2, 2), dtype=float)
        self._win: List[Tuple[float, np.ndarray]] = []  # list of (t, contrib)
        self.I_win = np.zeros((2, 2), dtype=float)

    def step(
        self,
        t: float,
        r_true: np.ndarray,
        v_rel_true: np.ndarray,
        meas_count: int,
    ) -> Dict[str, float]:
        t = float(t)
        meas_count = int(max(0, meas_count))

        r_true = np.asarray(r_true, dtype=float).reshape(2)
        v_rel_true = np.asarray(v_rel_true, dtype=float).reshape(2)

        rho = float(np.linalg.norm(r_true))
        vnorm = float(np.linalg.norm(v_rel_true))
        vperp = float(v_perp_mag(r_true, v_rel_true))

        H = doppler_H(r_true, v_rel_true)
        Hnorm = float(np.linalg.norm(H)) if H is not None else float("nan")

        add = (H is not None) and (meas_count > 0)
        if self.use_gating and add:
            if (rho <= self.rho_min) or (vnorm <= self.v_min) or (vperp <= self.v_perp_min):
                add = False

        if add:
            contrib = float(meas_count) * self.Rinv * np.outer(H, H)
            self.I_tot = self.I_tot + contrib

            if self.window_s > 0.0:
                self._win.append((t, contrib))
                self.I_win = self.I_win + contrib

        # pop old from window
        if self.window_s > 0.0:
            t0 = t - self.window_s
            while self._win and self._win[0][0] < t0:
                _, old_c = self._win.pop(0)
                self.I_win = self.I_win - old_c

        Iw = 0.5 * (self.I_win + self.I_win.T)
        It = 0.5 * (self.I_tot + self.I_tot.T)

        def eig_min_max(M: np.ndarray) -> Tuple[float, float]:
            try:
                w = np.linalg.eigvalsh(M)
                return float(w[0]), float(w[1])
            except Exception:
                return float("nan"), float("nan")

        wmin_w, wmax_w = eig_min_max(Iw)
        wmin_t, wmax_t = eig_min_max(It)

        def crlb_trace_and_stds(M: np.ndarray) -> Tuple[float, float, float]:
            try:
                C = np.linalg.inv(M + self.reg_eps * np.eye(2))
                tr = float(np.trace(C))
                sx = float(math.sqrt(max(C[0, 0], 0.0)))
                sy = float(math.sqrt(max(C[1, 1], 0.0)))
                return tr, sx, sy
            except Exception:
                return float("nan"), float("nan"), float("nan")

        crlb_w_tr, crlb_w_sx, crlb_w_sy = crlb_trace_and_stds(Iw)
        crlb_t_tr, crlb_t_sx, crlb_t_sy = crlb_trace_and_stds(It)

        return {
            "rho_true": rho,
            "vrel_norm_true": vnorm,
            "vperp_true": vperp,
            "Hnorm_true": Hnorm,

            "fim_win_xx": float(Iw[0, 0]),
            "fim_win_xy": float(Iw[0, 1]),
            "fim_win_yy": float(Iw[1, 1]),
            "fim_win_eig_min": wmin_w,
            "fim_win_eig_max": wmax_w,
            "fim_win_trace": float(np.trace(Iw)),

            "fim_tot_eig_min": wmin_t,
            "fim_tot_eig_max": wmax_t,
            "fim_tot_trace": float(np.trace(It)),

            "crlb_win_trace": crlb_w_tr,
            "crlb_win_std_x": crlb_w_sx,
            "crlb_win_std_y": crlb_w_sy,

            "crlb_tot_trace": crlb_t_tr,
            "crlb_tot_std_x": crlb_t_sx,
            "crlb_tot_std_y": crlb_t_sy,

            "fim_added": float(int(add)),
            "fim_meas_count": float(meas_count),
        }


# ---------------------------
# Logging helpers
# ---------------------------

TRACE_FIELDS = [
    "controller", "episode", "step",
    "t", "reward", "terminated", "truncated", "term_reason",
    "a_speed", "a_turn",

    # estimator quantities
    "r_des_x", "r_des_y",
    "r_mix_x", "r_mix_y",
    "Pmix_xx", "Pmix_xy", "Pmix_yy",
    "err_est", "std_x", "std_y", "std_max",
    "sens_avg",
    "nis_mix", "nis_excess",
    "w_max", "ess", "rejuv_count_ep",
    "doppler_meas_step", "doppler_updates_step", "doppler_updated_any",

    # truth
    "r_true_x", "r_true_y",
    "pL_x", "pL_y",
    "pF_x", "pF_y",

    # true errors
    "err_true_form",
    "err_true_rel_est",

    # FIM/CRLB
    "rho_true", "vrel_norm_true", "vperp_true", "Hnorm_true",
    "fim_win_eig_min", "fim_win_eig_max", "fim_win_trace",
    "crlb_win_trace", "crlb_win_std_x", "crlb_win_std_y",
    "fim_added", "fim_meas_count",
]

def write_trace_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr = {}
            for k in TRACE_FIELDS:
                v = r.get(k, "")
                if isinstance(v, (np.floating, np.integer, np.bool_)):
                    v = v.item()
                rr[k] = v
            w.writerow(rr)

def write_trace_npz(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    data = {}
    for k in TRACE_FIELDS:
        data[k] = np.array([r.get(k, np.nan) for r in rows], dtype=object)
    for k, arr in list(data.items()):
        try:
            data[k] = arr.astype(np.float64)
        except Exception:
            pass
    np.savez_compressed(path, **data)

def write_json(path: str, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------------------------
# Controller runner
# ---------------------------

def run_one_controller(
    *,
    controller_name: str,
    env_module: str,
    cfg_kwargs: Dict[str, Any],
    episodes: int,
    seed0: int,
    out_dir: str,
    render: bool,
    fim_window_s: float,
    fim_use_gating: bool,
    fim_reg_eps: float,
    baseline_kind: str,
    baseline_kp: float,
    baseline_kexc: float,
    baseline_std_trigger: float,
    algo: str,
    model_path: str,
    models_dir: str,
    device: str,
) -> Dict[str, Any]:
    # Import environment
    try:
        mod = __import__(env_module, fromlist=["UUVRelPosBankEnv", "UUVConfig"])
        UUVRelPosBankEnv = getattr(mod, "UUVRelPosBankEnv")
        UUVConfig = getattr(mod, "UUVConfig")
    except Exception as e:
        raise RuntimeError(
            f"Could not import environment from module '{env_module}'. "
            f"Put this script next to {env_module}.py or pass --env-module."
        ) from e

    # Load SB3 model if RL
    model = None
    obs_norm = None
    if controller_name == "rl":
        try:
            from stable_baselines3 import SAC, PPO
        except Exception as e:
            raise RuntimeError("Install stable-baselines3[extra]") from e

        algo_l = algo.lower()
        if algo_l == "sac":
            model = SAC.load(model_path, device=device)
        elif algo_l == "ppo":
            model = PPO.load(model_path, device=device)
        else:
            raise ValueError("algo must be sac or ppo")

        obs_norm = try_load_vecnormalize(models_dir)

    rng = np.random.default_rng(int(seed0) + 12345)

    ctrl_dir = os.path.join(out_dir, controller_name if controller_name == "rl" else f"baseline_{baseline_kind}")
    traces_dir = os.path.join(ctrl_dir, "traces")
    ensure_dir(traces_dir)

    meta = {
        "timestamp": timestamp(),
        "controller_name": controller_name,
        "baseline_kind": baseline_kind,
        "baseline_kp": float(baseline_kp),
        "baseline_kexc": float(baseline_kexc),
        "baseline_std_trigger": float(baseline_std_trigger),
        "algo": algo,
        "model_path": model_path if controller_name == "rl" else "",
        "models_dir": models_dir,
        "device": device,
        "obs_normalization": bool(obs_norm is not None),
        "fim_window_s": float(fim_window_s),
        "fim_use_gating": bool(fim_use_gating),
        "fim_reg_eps": float(fim_reg_eps),
        "episodes": int(episodes),
        "seed0": int(seed0),
        "env_module": env_module,
        "cfg_kwargs": dict(cfg_kwargs),
    }
    write_json(os.path.join(ctrl_dir, "meta.json"), meta)

    eps_summary: List[Dict[str, Any]] = []
    series_cache: List[Dict[str, Any]] = []

    for ep in range(1, int(episodes) + 1):
        ep_seed = int(seed0) + ep - 1

        cfg = UUVConfig(**cfg_kwargs)
        try:
            cfg.log_truth_diagnostics = True
        except Exception:
            pass

        env = UUVRelPosBankEnv(cfg=cfg, render_mode=("human" if render else "none"))
        try:
            env.set_difficulty(1.0)
        except Exception:
            pass

        obs, info = env.reset(seed=ep_seed)

        fim = FIMTracker(
            sigma_s=float(getattr(cfg, "sigma_s_true", 0.015)),
            window_s=float(fim_window_s),
            reg_eps=float(fim_reg_eps),
            use_gating=bool(fim_use_gating),
            rho_min=float(getattr(cfg, "doppler_rho_min", 20.0)),
            v_min=float(getattr(cfg, "doppler_v_min", 0.05)),
            v_perp_min=float(getattr(cfg, "doppler_v_perp_min_hard", 0.25)),
        )

        rows: List[Dict[str, Any]] = []
        ep_return = 0.0
        done = False
        step_idx = 0

        t_list, sens_list, fim_eig_list, crlb_tr_list, err_form_list = [], [], [], [], []

        while not done:
            # Action
            if controller_name == "rl":
                obs_in = obs
                if obs_norm is not None:
                    obs_in = obs_norm.normalize(obs_in)
                action, _ = model.predict(obs_in, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(-1)
            else:
                action = baseline_action_from_obs_raw(
                    obs_raw=obs,
                    cfg=cfg,
                    controller=baseline_kind,
                    kp=float(baseline_kp),
                    kexc=float(baseline_kexc),
                    std_trigger_m=float(baseline_std_trigger),
                    rng=rng,
                )

            obs2, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            done = bool(terminated) or bool(truncated)

            if render:
                try:
                    env.render()
                except Exception:
                    pass

            def g(k, default=np.nan):
                try:
                    return info.get(k, default)
                except Exception:
                    return default

            t = float(g("t", np.nan))
            term_reason = str(g("term_reason", "running"))

            r_true = np.array([float(g("r_true_x", np.nan)), float(g("r_true_y", np.nan))], dtype=float)
            pF_true = np.array([float(g("pF_x", np.nan)), float(g("pF_y", np.nan))], dtype=float)

            r_des = np.array([float(g("r_des_x", np.nan)), float(g("r_des_y", np.nan))], dtype=float)
            r_mix = np.array([float(g("r_mix_x", np.nan)), float(g("r_mix_y", np.nan))], dtype=float)

            err_true_form = float(np.linalg.norm(r_true - r_des)) if np.all(np.isfinite(r_true)) else float("nan")
            err_true_rel_est = float(np.linalg.norm(r_mix - r_true)) if np.all(np.isfinite(r_true)) else float("nan")

            vL = vel_vec2(float(g("leader_speed", np.nan)), float(g("heading_L_deg", np.nan)))
            vF = vel_vec2(float(g("speed_F", np.nan)), float(g("heading_F_deg", np.nan)))
            v_rel_true = vL - vF

            meas_count = int(round(float(g("doppler_meas_step", 0.0))))
            fim_metrics = fim.step(t=t, r_true=r_true, v_rel_true=v_rel_true, meas_count=meas_count)

            row = {
                "controller": ("RL" if controller_name == "rl" else f"BASELINE:{baseline_kind}"),
                "episode": ep,
                "step": step_idx,
                "t": t,
                "reward": float(reward),
                "terminated": int(bool(terminated)),
                "truncated": int(bool(truncated)),
                "term_reason": term_reason,
                "a_speed": float(action[0]) if action.size > 0 else 0.0,
                "a_turn": float(action[1]) if action.size > 1 else 0.0,

                "r_des_x": float(r_des[0]),
                "r_des_y": float(r_des[1]),
                "r_mix_x": float(r_mix[0]),
                "r_mix_y": float(r_mix[1]),
                "Pmix_xx": float(g("Pmix_xx", np.nan)),
                "Pmix_xy": float(g("Pmix_xy", np.nan)),
                "Pmix_yy": float(g("Pmix_yy", np.nan)),
                "err_est": float(g("err_est", np.nan)),
                "std_x": float(g("std_x", np.nan)),
                "std_y": float(g("std_y", np.nan)),
                "std_max": float(g("std_max", np.nan)),
                "sens_avg": float(g("sens_avg", np.nan)),
                "nis_mix": float(g("nis_mix", np.nan)),
                "nis_excess": float(g("nis_excess", np.nan)),
                "w_max": float(g("w_max", np.nan)),
                "ess": float(g("ess", np.nan)),
                "rejuv_count_ep": float(g("rejuv_count_ep", np.nan)),
                "doppler_meas_step": float(g("doppler_meas_step", np.nan)),
                "doppler_updates_step": float(g("doppler_updates_step", np.nan)),
                "doppler_updated_any": float(g("doppler_updated_any", np.nan)),

                "r_true_x": float(r_true[0]),
                "r_true_y": float(r_true[1]),
                "pL_x": float(g("pL_x", np.nan)),
                "pL_y": float(g("pL_y", np.nan)),
                "pF_x": float(pF_true[0]),
                "pF_y": float(pF_true[1]),

                "err_true_form": err_true_form,
                "err_true_rel_est": err_true_rel_est,
            }
            row.update(fim_metrics)

            rows.append(row)

            t_list.append(t)
            sens_list.append(float(row["sens_avg"]))
            fim_eig_list.append(float(row["fim_win_eig_min"]))
            crlb_tr_list.append(float(row["crlb_win_trace"]))
            err_form_list.append(float(row["err_true_form"]))

            obs = obs2
            step_idx += 1

        last = rows[-1] if rows else {}
        eps_summary.append({
            "episode": ep,
            "seed": ep_seed,
            "term_reason": str(last.get("term_reason", "")),
            "t_end": float(last.get("t", float("nan"))),
            "steps": int(step_idx),
            "return": float(ep_return),

            "err_true_form_end": float(last.get("err_true_form", float("nan"))),
            "err_true_rel_est_end": float(last.get("err_true_rel_est", float("nan"))),
            "err_est_end": float(last.get("err_est", float("nan"))),

            "fim_win_eig_min_end": float(last.get("fim_win_eig_min", float("nan"))),
            "crlb_win_trace_end": float(last.get("crlb_win_trace", float("nan"))),

            "fim_tot_trace_end": float(last.get("fim_tot_trace", float("nan"))),
            "crlb_tot_trace_end": float(last.get("crlb_tot_trace", float("nan"))),
        })

        series_cache.append({
            "episode": ep,
            "term_reason": str(last.get("term_reason", "")),
            "t": np.asarray(t_list, dtype=float),
            "sens_avg": np.asarray(sens_list, dtype=float),
            "fim_win_eig_min": np.asarray(fim_eig_list, dtype=float),
            "crlb_win_trace": np.asarray(crlb_tr_list, dtype=float),
            "err_true_form": np.asarray(err_form_list, dtype=float),
        })

        base = os.path.join(traces_dir, f"ep{ep:03d}")
        write_trace_csv(base + ".csv", rows)
        write_trace_npz(base + ".npz", rows)

        env.close()

        print(f"[{controller_name.upper()}:{baseline_kind}] ep {ep}/{episodes} seed={ep_seed} term={eps_summary[-1]['term_reason']} "
              f"return={ep_return:.2f} err_true_form_end={eps_summary[-1]['err_true_form_end']:.2f}")

    summary = {"meta": meta, "episodes": eps_summary}
    write_json(os.path.join(ctrl_dir, "summary.json"), summary)

    summary_csv_path = os.path.join(ctrl_dir, "summary.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        fields = list(eps_summary[0].keys()) if eps_summary else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in eps_summary:
            w.writerow(r)

    # store series cache
    np.savez_compressed(os.path.join(ctrl_dir, "series_cache.npz"), **{
        f"ep{d['episode']:03d}_t": d["t"] for d in series_cache
    }, **{
        f"ep{d['episode']:03d}_sens": d["sens_avg"] for d in series_cache
    }, **{
        f"ep{d['episode']:03d}_fim": d["fim_win_eig_min"] for d in series_cache
    }, **{
        f"ep{d['episode']:03d}_crlb": d["crlb_win_trace"] for d in series_cache
    }, **{
        f"ep{d['episode']:03d}_err": d["err_true_form"] for d in series_cache
    })

    summary["_series_cache_path"] = "series_cache.npz"
    return summary


# ---------------------------
# Comparison plots + paper tables (multi-controller)
# ---------------------------

@dataclass(frozen=True)
class ControllerRun:
    key: str         # short id used in column names (e.g., "rl", "pid")
    label: str       # displayed label in plots/tables (e.g., "RL", "PID")
    ctrl_dir: str    # directory containing traces/summary/series_cache
    summary: Dict[str, Any]

def _ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys

def _interp_to_grid(t: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size == 0:
        return np.full_like(grid, np.nan, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    if np.sum(m) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    t2 = t[m]; y2 = y[m]
    idx = np.argsort(t2)
    t2 = t2[idx]; y2 = y2[idx]
    out = np.interp(grid, t2, y2, left=np.nan, right=np.nan)
    out[grid < t2[0]] = np.nan
    out[grid > t2[-1]] = np.nan
    return out

def _load_series(ctrl_dir: str):
    sc_path = os.path.join(ctrl_dir, "series_cache.npz")
    z = np.load(sc_path)
    episodes = sorted(set([k.split("_")[0] for k in z.files]))
    series = []
    for ep in episodes:
        t = z[f"{ep}_t"]
        fim = z[f"{ep}_fim"]
        crlb = z[f"{ep}_crlb"]
        err = z[f"{ep}_err"]
        sens = z[f"{ep}_sens"]
        series.append((t, fim, crlb, err, sens))
    return series

def _median_curve(series_list, grid, idx):
    """
    idx: 1=fim, 2=crlb, 3=err, 4=sens
    """
    curves = []
    for (t, fim, crlb, err, sens) in series_list:
        y = [fim, crlb, err, sens][idx - 1]
        curves.append(_interp_to_grid(t, y, grid))
    M = np.vstack(curves) if curves else np.empty((0, grid.size))
    med = np.nanmedian(M, axis=0) if M.size else np.full_like(grid, np.nan)
    q25 = np.nanpercentile(M, 25, axis=0) if M.size else np.full_like(grid, np.nan)
    q75 = np.nanpercentile(M, 75, axis=0) if M.size else np.full_like(grid, np.nan)
    return med, q25, q75

def _read_trace_points(traces_dir: str, max_points: int = 8000) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    files = sorted([p for p in Path(traces_dir).glob("ep*.csv")])
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    x = float(row.get("sens_avg", "nan"))
                    y = float(row.get("fim_win_eig_min", "nan"))
                    if np.isfinite(x) and np.isfinite(y):
                        xs.append(x); ys.append(max(y, 1e-16))
                except Exception:
                    pass
        if len(xs) >= max_points:
            break
    xs = np.array(xs[:max_points], dtype=float)
    ys = np.array(ys[:max_points], dtype=float)
    return xs, ys

def _summarize_from_traces(traces_dir: str, rho_min: float = 20.0) -> Dict[str, float]:
    """
    Step-level metrics aggregated over all episodes.
    Uses rho_true>rho_min mask to avoid near-singular geometry.
    """
    fim_vals = []
    crlb_vals = []
    sens_vals = []
    rho_vals = []

    # degeneracy thresholds
    fim_deg_thr = 1e-12
    sens_zero_thr = 1e-3

    fim_deg_count = 0
    sens_zero_count = 0
    total_count = 0

    for p in sorted(Path(traces_dir).glob("ep*.csv")):
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rho = float(row.get("rho_true", "nan"))
                    fim = float(row.get("fim_win_eig_min", "nan"))
                    crlb = float(row.get("crlb_win_trace", "nan"))
                    sens = float(row.get("sens_avg", "nan"))
                except Exception:
                    continue
                if not (math.isfinite(rho) and rho > float(rho_min)):
                    continue
                if not math.isfinite(fim) or not math.isfinite(crlb) or not math.isfinite(sens):
                    continue
                fim = max(fim, 1e-16)
                crlb = max(crlb, 1e-16)

                fim_vals.append(fim)
                crlb_vals.append(crlb)
                sens_vals.append(sens)
                rho_vals.append(rho)

                total_count += 1
                if fim < fim_deg_thr:
                    fim_deg_count += 1
                if abs(sens) < sens_zero_thr:
                    sens_zero_count += 1

    fim_arr = np.asarray(fim_vals, dtype=float)
    crlb_arr = np.asarray(crlb_vals, dtype=float)
    sens_arr = np.asarray(sens_vals, dtype=float)
    rho_arr = np.asarray(rho_vals, dtype=float)

    out: Dict[str, float] = {}
    out["steps_used"] = float(total_count)
    out["sens_avg_mean"] = float(np.mean(sens_arr)) if sens_arr.size else float("nan")
    out["sens_avg_median"] = float(np.median(sens_arr)) if sens_arr.size else float("nan")

    out["fim_win_eig_min_median"] = float(np.median(fim_arr)) if fim_arr.size else float("nan")
    out["fim_win_eig_min_log10_median"] = float(np.median(np.log10(fim_arr))) if fim_arr.size else float("nan")

    out["crlb_win_trace_median"] = float(np.median(crlb_arr)) if crlb_arr.size else float("nan")
    out["crlb_win_trace_log10_median"] = float(np.median(np.log10(crlb_arr))) if crlb_arr.size else float("nan")

    out["fim_deg_frac"] = float(fim_deg_count / total_count) if total_count > 0 else float("nan")
    out["sens_zero_frac"] = float(sens_zero_count / total_count) if total_count > 0 else float("nan")

    out["rho_true_median"] = float(np.median(rho_arr)) if rho_arr.size else float("nan")
    return out

def _summarize_from_episode_summary(eps: List[Dict[str, Any]]) -> Dict[str, float]:
    term = np.array([str(d.get("term_reason", "")) for d in eps], dtype=object)
    succ = (term == "success")

    def arr(key):
        return np.array([float(d.get(key, np.nan)) for d in eps], dtype=float)

    out: Dict[str, float] = {}
    out["episodes"] = float(len(eps))
    out["success_rate"] = float(np.mean(succ)) if succ.size else float("nan")

    # success-only metrics
    err_form = arr("err_true_form_end")
    err_rel = arr("err_true_rel_est_end")
    t_end = arr("t_end")

    if np.any(succ):
        out["t_end_median_success"] = float(np.nanmedian(t_end[succ]))
        out["err_true_form_end_median_success"] = float(np.nanmedian(err_form[succ]))
        out["err_true_form_end_p95_success"] = float(np.nanpercentile(err_form[succ], 95))
        out["err_true_rel_est_end_median_success"] = float(np.nanmedian(err_rel[succ]))
    else:
        out["t_end_median_success"] = float("nan")
        out["err_true_form_end_median_success"] = float("nan")
        out["err_true_form_end_p95_success"] = float("nan")
        out["err_true_rel_est_end_median_success"] = float("nan")

    # all episodes (robust)
    out["err_true_form_end_median_all"] = float(np.nanmedian(err_form)) if err_form.size else float("nan")
    out["t_end_median_all"] = float(np.nanmedian(t_end)) if t_end.size else float("nan")
    return out

def write_paper_tables_multi(
    *,
    compare_dir: str,
    runs: List[ControllerRun],
    rho_min_for_step_metrics: float = 20.0,
) -> None:
    """
    Writes one table (one row per controller) and machine-readable variants.
    """
    ensure_dir(compare_dir)

    rows = []
    for run in runs:
        ep = _summarize_from_episode_summary(run.summary["episodes"])
        step = _summarize_from_traces(os.path.join(run.ctrl_dir, "traces"), rho_min=float(rho_min_for_step_metrics))

        r: Dict[str, Any] = {"controller": run.label, "key": run.key}
        r.update(ep)
        r.update(step)
        r["fim_log10_med"] = r.get("fim_win_eig_min_log10_median", float("nan"))
        r["crlb_log10_med"] = r.get("crlb_win_trace_log10_median", float("nan"))
        rows.append(r)

    # deterministic order by input order
    fields = [
        "controller",
        "episodes", "success_rate",
        "t_end_median_success",
        "err_true_form_end_median_success",
        "err_true_form_end_p95_success",
        "err_true_rel_est_end_median_success",

        # step-level
        "sens_avg_mean",
        "fim_log10_med",
        "crlb_log10_med",
        "fim_deg_frac",
        "sens_zero_frac",
        "steps_used",
        "rho_true_median",
    ]

    # CSV
    csv_path = os.path.join(compare_dir, "paper_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    # JSON
    json_path = os.path.join(compare_dir, "paper_table.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "fields": fields}, f, indent=2)

    # Markdown
    md_path = os.path.join(compare_dir, "paper_table.md")

    def fmt(x, nd=3):
        try:
            v = float(x)
            if not math.isfinite(v):
                return "—"
            return f"{v:.{nd}f}"
        except Exception:
            return str(x)

    lines = []
    lines.append("| Controller | Success [%] | t_end median (succ) [s] | e_form median (succ) [m] | e_form p95 (succ) [m] | sens_avg mean | log10 λmin(FIM_w) med | log10 tr(CRLB_w) med | FIM-degen frac | sens≈0 frac |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['controller']} | {fmt(100*r['success_rate'],1)} | {fmt(r['t_end_median_success'],1)} | "
            f"{fmt(r['err_true_form_end_median_success'],2)} | {fmt(r['err_true_form_end_p95_success'],2)} | "
            f"{fmt(r['sens_avg_mean'],3)} | {fmt(r['fim_log10_med'],3)} | {fmt(r['crlb_log10_med'],3)} | "
            f"{fmt(r['fim_deg_frac'],3)} | {fmt(r['sens_zero_frac'],3)} |"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # LaTeX (plain tabular)
    tex_path = os.path.join(compare_dir, "paper_table.tex")
    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(
        r"\caption{Comparison of controllers on the Doppler-only formation task. "
        r"Step-level information metrics computed for $\rho_{\mathrm{true}}>%.0f\,\mathrm{m}$ (masking near-singular geometry).}"
        % float(rho_min_for_step_metrics)
    )
    tex.append(r"\label{tab:controllers_compare}")
    tex.append(r"\begin{tabular}{lrrrrrrrrr}")
    tex.append(r"\hline")
    tex.append(r"Controller & Succ.\,[\%] & $t_{\mathrm{end}}$ (med) & $e_{\mathrm{form}}$ (med) & $e_{\mathrm{form}}$ (p95) & $\overline{s}_{\mathrm{ens}}$ & $\log_{10}\lambda_{\min}(I_w)$ & $\log_{10}\mathrm{tr}(\mathrm{CRLB}_w)$ & degen.\,frac & $s_{\mathrm{ens}}\!\approx\!0$ frac \\")
    tex.append(r"\hline")
    for r in rows:
        def num(x, nd=3):
            try:
                v = float(x)
                if not math.isfinite(v):
                    return "--"
                return f"{v:.{nd}f}"
            except Exception:
                return "--"
        tex.append(
            f"{r['controller']} & "
            f"{(100*float(r['success_rate'])):.1f} & "
            f"{num(r['t_end_median_success'],1)} & "
            f"{num(r['err_true_form_end_median_success'],2)} & "
            f"{num(r['err_true_form_end_p95_success'],2)} & "
            f"{num(r['sens_avg_mean'],3)} & "
            f"{num(r['fim_log10_med'],3)} & "
            f"{num(r['crlb_log10_med'],3)} & "
            f"{num(r['fim_deg_frac'],3)} & "
            f"{num(r['sens_zero_frac'],3)} \\\\"
        )
    tex.append(r"\hline")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")

    print("[OK] Paper tables saved:")
    print("  ", csv_path)
    print("  ", md_path)
    print("  ", tex_path)
    print("  ", json_path)

def write_plot_grid_tex(compare_dir: str) -> None:
    """
    Writes a LaTeX table* that arranges the key comparison plots in a 2x3 grid.

    NOTE:
      The generated paths assume you copy the whole 'compare/' directory into your LaTeX project root
      and include with:
          \\input{compare/plot_grid.tex}
    """
    tex_path = os.path.join(compare_dir, "plot_grid.tex")
    tex = []
    tex.append(r"% Auto-generated by uuv_compare_experiment_tables.py")
    tex.append(r"% Assumes the folder 'compare/' is available in your LaTeX project root.")
    tex.append(r"\begin{table*}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Comparison plots for RL and baseline controllers (PID, PID+exc, Random) on the Doppler-only formation task.}")
    tex.append(r"\label{tab:compare_plots}")
    tex.append(r"\setlength{\tabcolsep}{2pt}")
    tex.append(r"\renewcommand{\arraystretch}{1.0}")
    tex.append(r"\begin{tabular}{ccc}")
    tex.append(r"\includegraphics[width=0.32\textwidth]{compare/plot_success_rate.png} &"
               r"\includegraphics[width=0.32\textwidth]{compare/plot_ecdf_err_true_form_end.png} &"
               r"\includegraphics[width=0.32\textwidth]{compare/plot_ecdf_time_to_success.png} \\")
    tex.append(r"\includegraphics[width=0.32\textwidth]{compare/plot_fim_eigmin_median_over_time.png} &"
               r"\includegraphics[width=0.32\textwidth]{compare/plot_crlb_trace_median_over_time.png} &"
               r"\includegraphics[width=0.32\textwidth]{compare/plot_err_true_form_median_over_time.png} \\")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table*}")
    tex.append(r"% Optional (often used as a separate figure):")
    tex.append(r"% \begin{figure}[t]\centering")
    tex.append(r"% \includegraphics[width=0.65\linewidth]{compare/plot_scatter_sens_vs_fim.png}")
    tex.append(r"% \caption{sens\_avg vs $\lambda_{\min}(I_w)$ scatter (sampled points).}\end{figure}")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print("[OK] LaTeX plot grid saved:", tex_path)

def make_compare_outputs_multi(
    *,
    out_dir: str,
    runs: List[ControllerRun],
) -> None:
    compare_dir = os.path.join(out_dir, "compare")
    ensure_dir(compare_dir)

    # fixed colors for consistency across plots
    color_map = {
        "rl": "C0",
        "pid": "C1",
        "pid_exc": "C2",
        "random": "C3",
    }

    # ---------------------------
    # compare_summary.csv (wide, per-episode)
    # ---------------------------
    E = min(len(run.summary.get("episodes", [])) for run in runs) if runs else 0

    fields = ["episode", "seed"]
    for run in runs:
        k = run.key
        fields += [
            f"{k}_term",
            f"{k}_t_end",
            f"{k}_err_true_form_end",
            f"{k}_fim_win_eig_min_end",
            f"{k}_crlb_win_trace_end",
        ]

    compare_csv = os.path.join(compare_dir, "compare_summary.csv")
    with open(compare_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(E):
            row = {"episode": i + 1, "seed": runs[0].summary["episodes"][i].get("seed", "") if runs else ""}
            for run in runs:
                ep = run.summary["episodes"][i]
                k = run.key
                row[f"{k}_term"] = ep.get("term_reason", "")
                row[f"{k}_t_end"] = ep.get("t_end", np.nan)
                row[f"{k}_err_true_form_end"] = ep.get("err_true_form_end", np.nan)
                row[f"{k}_fim_win_eig_min_end"] = ep.get("fim_win_eig_min_end", np.nan)
                row[f"{k}_crlb_win_trace_end"] = ep.get("crlb_win_trace_end", np.nan)
            w.writerow(row)

    # ---------------------------
    # episode arrays
    # ---------------------------
    def arr(eps: List[Dict[str, Any]], key: str) -> np.ndarray:
        return np.array([float(d.get(key, np.nan)) for d in eps[:E]], dtype=float)

    ctrl_data = {}
    for run in runs:
        eps = run.summary["episodes"][:E]
        term = np.array([str(d.get("term_reason", "")) for d in eps], dtype=object)
        succ = (term == "success")
        ctrl_data[run.key] = {
            "label": run.label,
            "color": color_map.get(run.key, None),
            "term": term,
            "succ": succ,
            "t_end": arr(eps, "t_end"),
            "err_form_end": arr(eps, "err_true_form_end"),
            "fim_end": arr(eps, "fim_win_eig_min_end"),
            "crlb_end": arr(eps, "crlb_win_trace_end"),
        }

    # ---------------------------
    # Plot 1: success rate
    # ---------------------------
    fig, ax = plt.subplots()
    keys = [r.key for r in runs]
    labels = [ctrl_data[k]["label"] for k in keys]
    colors = [ctrl_data[k]["color"] if ctrl_data[k]["color"] is not None else f"C{i}" for i, k in enumerate(keys)]
    success_rates = [float(np.mean(ctrl_data[k]["succ"])) if ctrl_data[k]["succ"].size else 0.0 for k in keys]

    ax.bar(np.arange(len(keys)), success_rates, color=colors)
    ax.set_xticks(np.arange(len(keys)), labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success rate")
    ax.set_title("Success rate (term_reason == success)")
    for i, v in enumerate(success_rates):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(compare_dir, "plot_success_rate.png"), dpi=170)
    plt.close(fig)

    # ---------------------------
    # Plot 2: ECDF final true formation error (success only)
    # ---------------------------
    fig, ax = plt.subplots()
    for k in keys:
        succ = ctrl_data[k]["succ"]
        xs, ys = _ecdf(ctrl_data[k]["err_form_end"][succ])
        if xs.size == 0:
            continue
        ax.step(xs, ys, where="post", label=f"{ctrl_data[k]['label']} (success)", color=ctrl_data[k]["color"])
    ax.set_xlabel("err_true_form_end = ||r_true - r_des|| [m]")
    ax.set_ylabel("ECDF")
    ax.set_title("Distribution of final TRUE formation error (success episodes)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(compare_dir, "plot_ecdf_err_true_form_end.png"), dpi=170)
    plt.close(fig)

    # ---------------------------
    # Plot 3: ECDF time-to-termination (success only)
    # ---------------------------
    fig, ax = plt.subplots()
    for k in keys:
        succ = ctrl_data[k]["succ"]
        xs, ys = _ecdf(ctrl_data[k]["t_end"][succ])
        if xs.size == 0:
            continue
        ax.step(xs, ys, where="post", label=f"{ctrl_data[k]['label']} (success)", color=ctrl_data[k]["color"])
    ax.set_xlabel("t_end [s]")
    ax.set_ylabel("ECDF")
    ax.set_title("Distribution of time-to-success/termination (success episodes)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(compare_dir, "plot_ecdf_time_to_success.png"), dpi=170)
    plt.close(fig)

    # ---------------------------
    # Load series for all controllers
    # ---------------------------
    series_by = {run.key: _load_series(run.ctrl_dir) for run in runs}

    # Determine time grid from data (robust)
    dt_candidates = []
    max_t = 0.0
    for k, series_list in series_by.items():
        for (t, _, _, _, _) in series_list:
            if t.size >= 2:
                dts = np.diff(t)
                dts = dts[np.isfinite(dts) & (dts > 0)]
                if dts.size:
                    dt_candidates.append(float(np.median(dts)))
            if t.size and np.isfinite(t[-1]):
                max_t = max(max_t, float(np.nanmax(t)))
    dt_grid = float(np.median(dt_candidates)) if dt_candidates else 1.0
    grid = np.arange(0.0, max_t + 1e-9, dt_grid)

    # ---------------------------
    # Plot 4: FIM median curve (log)
    # ---------------------------
    fig, ax = plt.subplots()
    for k in keys:
        med, q25, q75 = _median_curve(series_by[k], grid, idx=1)
        # clip for log-scale safety
        med = np.clip(med, 1e-16, np.inf)
        q25 = np.clip(q25, 1e-16, np.inf)
        q75 = np.clip(q75, 1e-16, np.inf)
        ax.plot(grid, med, label=f"{ctrl_data[k]['label']} median", color=ctrl_data[k]["color"])
        ax.fill_between(grid, q25, q75, color=ctrl_data[k]["color"], alpha=0.18)
    ax.set_yscale("log")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("λ_min(FIM_window) (log)")
    ax.set_title("Information growth: median λ_min(FIM_window) over time (IQR shaded)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(compare_dir, "plot_fim_eigmin_median_over_time.png"), dpi=170)
    plt.close(fig)

    # ---------------------------
    # Plot 5: CRLB median curve (log)
    # ---------------------------
    fig, ax = plt.subplots()
    for k in keys:
        med, q25, q75 = _median_curve(series_by[k], grid, idx=2)
        med = np.clip(med, 1e-16, np.inf)
        q25 = np.clip(q25, 1e-16, np.inf)
        q75 = np.clip(q75, 1e-16, np.inf)
        ax.plot(grid, med, label=f"{ctrl_data[k]['label']} median", color=ctrl_data[k]["color"])
        ax.fill_between(grid, q25, q75, color=ctrl_data[k]["color"], alpha=0.18)
    ax.set_yscale("log")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("trace(CRLB_window) [m²] (log)")
    ax.set_title("Lower bound: median trace(CRLB_window) over time (IQR shaded)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(compare_dir, "plot_crlb_trace_median_over_time.png"), dpi=170)
    plt.close(fig)

    # ---------------------------
    # Plot 6: TRUE formation error curve (median + IQR)
    # ---------------------------
    fig, ax = plt.subplots()
    for k in keys:
        med, q25, q75 = _median_curve(series_by[k], grid, idx=3)
        ax.plot(grid, med, label=f"{ctrl_data[k]['label']} median", color=ctrl_data[k]["color"])
        ax.fill_between(grid, q25, q75, color=ctrl_data[k]["color"], alpha=0.18)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("err_true_form = ||r_true - r_des|| [m]")
    ax.set_title("TRUE formation error over time (median + IQR)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(compare_dir, "plot_err_true_form_median_over_time.png"), dpi=170)
    plt.close(fig)

    # ---------------------------
    # Plot 7: scatter sens vs FIM (sample)
    # ---------------------------
    fig, ax = plt.subplots()
    for k in keys:
        xs, ys = _read_trace_points(os.path.join(os.path.join(out_dir, "rl" if k == "rl" else f"baseline_{k}"), "traces"))
        if xs.size == 0:
            continue
        ax.scatter(xs, ys, s=10, alpha=0.35, label=ctrl_data[k]["label"], color=ctrl_data[k]["color"])
    ax.set_yscale("log")
    ax.set_xlabel("sens_avg [m/s] (proxy for v_perp)")
    ax.set_ylabel("λ_min(FIM_window) (log)")
    ax.set_title("sens_avg vs λ_min(FIM_window) (sampled points from traces)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(compare_dir, "plot_scatter_sens_vs_fim.png"), dpi=170)
    plt.close(fig)

    # Paper tables (multi-controller)
    write_paper_tables_multi(compare_dir=compare_dir, runs=runs, rho_min_for_step_metrics=20.0)

    # LaTeX plot grid (2x3)
    write_plot_grid_tex(compare_dir)

    print("[OK] Comparison outputs saved to:", compare_dir)
    print("[OK] compare_summary.csv:", compare_csv)


# ---------------------------
# CLI
# ---------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run ALL controllers and generate comparison outputs (plots + tables)")
    pr.add_argument("--env-module", type=str, default="uuv_rl_patched",
                    help="Python module that defines UUVRelPosBankEnv and UUVConfig (default: uuv_rl_patched)")
    pr.add_argument("--out-dir", type=str, default="cmp_results")

    pr.add_argument("--episodes", type=int, default=30)
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--render", action="store_true")

    # RL model
    pr.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    pr.add_argument("--model", type=str, required=True)
    pr.add_argument("--models-dir", type=str, default="models")
    pr.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")

    # Baseline selection
    pr.add_argument("--baseline", choices=["all", "pid", "pid_exc", "random"], default="all",
                    help="Which baselines to run. Default 'all' runs pid+pid_exc+random.")
    pr.add_argument("--baseline-kp", type=float, default=0.02)
    pr.add_argument("--baseline-kexc", type=float, default=0.8)
    pr.add_argument("--baseline-std-trigger", type=float, default=0.0,
                    help="If 0 -> use cfg.tol_std_hard; else explicit meters")

    # FIM/CRLB params
    pr.add_argument("--fim-window", type=float, default=30.0)
    pr.add_argument("--fim-reg-eps", type=float, default=1e-9)
    pr.add_argument("--fim-no-gating", action="store_true",
                    help="If set, FIM uses all meas_count samples (no rho/v/v_perp gating)")

    # Env cfg overrides (optional)
    pr.add_argument("--action-dt", type=float, default=None,
                    help="Override cfg.action_dt (use with caution; should match training)")
    pr.add_argument("--max-steps", type=int, default=None,
                    help="Override cfg.max_steps")

    return p

def main():
    args = build_parser().parse_args()

    device = args.device
    if device == "auto":
        device = "cpu"

    out_dir = str(args.out_dir)
    ensure_dir(out_dir)

    cfg_kwargs: Dict[str, Any] = {}
    if args.action_dt is not None:
        cfg_kwargs["action_dt"] = float(args.action_dt)
    if args.max_steps is not None:
        cfg_kwargs["max_steps"] = int(args.max_steps)

    if args.cmd == "run":
        # always run RL (as requested)
        rl_summary = run_one_controller(
            controller_name="rl",
            env_module=str(args.env_module),
            cfg_kwargs=cfg_kwargs,
            episodes=int(args.episodes),
            seed0=int(args.seed),
            out_dir=out_dir,
            render=bool(args.render),
            fim_window_s=float(args.fim_window),
            fim_use_gating=(not bool(args.fim_no_gating)),
            fim_reg_eps=float(args.fim_reg_eps),
            baseline_kind="n/a",
            baseline_kp=float(args.baseline_kp),
            baseline_kexc=float(args.baseline_kexc),
            baseline_std_trigger=float(args.baseline_std_trigger),
            algo=str(args.algo),
            model_path=str(args.model),
            models_dir=str(args.models_dir),
            device=str(device),
        )

        runs: List[ControllerRun] = [
            ControllerRun(
                key="rl",
                label="RL",
                ctrl_dir=os.path.join(out_dir, "rl"),
                summary=rl_summary,
            )
        ]

        baseline_kinds = ["pid", "pid_exc", "random"] if str(args.baseline) == "all" else [str(args.baseline)]
        baseline_labels = {"pid": "PID", "pid_exc": "PID+exc", "random": "Random"}

        for kind in baseline_kinds:
            base_summary = run_one_controller(
                controller_name="baseline",
                env_module=str(args.env_module),
                cfg_kwargs=cfg_kwargs,
                episodes=int(args.episodes),
                seed0=int(args.seed),
                out_dir=out_dir,
                render=bool(args.render),
                fim_window_s=float(args.fim_window),
                fim_use_gating=(not bool(args.fim_no_gating)),
                fim_reg_eps=float(args.fim_reg_eps),
                baseline_kind=kind,
                baseline_kp=float(args.baseline_kp),
                baseline_kexc=float(args.baseline_kexc),
                baseline_std_trigger=float(args.baseline_std_trigger),
                algo=str(args.algo),
                model_path=str(args.model),
                models_dir=str(args.models_dir),
                device=str(device),
            )
            runs.append(
                ControllerRun(
                    key=kind,
                    label=baseline_labels.get(kind, f"Baseline:{kind}"),
                    ctrl_dir=os.path.join(out_dir, f"baseline_{kind}"),
                    summary=base_summary,
                )
            )

        make_compare_outputs_multi(out_dir=out_dir, runs=runs)

        print("\nDONE.")
        print("Results directory:", out_dir)
        for r in runs:
            print(f" - {r.label:8s} summary:", os.path.join(r.ctrl_dir, "summary.csv"))
        print("Compare outputs:", os.path.join(out_dir, "compare"))

    else:
        raise ValueError("Unknown cmd")

if __name__ == "__main__":
    main()
