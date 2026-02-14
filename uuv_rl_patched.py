
# -*- coding: utf-8 -*-
"""
UUV RL Evaluation – FIM/CRLB + Truth Metrics (Doppler-only)
==========================================================

Co dostajesz:
  • Self-contained środowisko Gymnasium (2D) zgodne z Twoim treningiem:
      - follower sterowany (akcja 2D),
      - leader jedzie ze stałym v i heading w epizodzie,
      - Doppler (range-rate) próbkowany co s_meas_period,
      - bank EKF (multi-hypothesis) + rejuvenation.
  • TRYB EWALUACJI: automatycznie liczy i loguje:
      - metryki na prawdzie (true formation error, true rel-pos error),
      - informację Fishera (FIM) z Dopplera i odpowiadający CRLB,
        zarówno w wersji "cumulative" (od startu) jak i "windowed" (ostatnie T sekund).
  • Eksport trace: CSV + NPZ + meta.json + summary.json.

Uruchomienie (na istniejącym modelu SB3):
  pip install numpy gymnasium stable-baselines3[extra] pygame

  python uuv_eval_fim.py eval \
      --algo sac \
      --model models/best_model.zip \
      --models-dir models \
      --episodes 30 \
      --seed 42 \
      --out-dir eval_fim_logs \
      --fim-window 30 \
      --render

Uwagi:
  • FIM/CRLB jest liczone na Prawdzie (r_true, v_rel_true) z modelu pomiaru Dopplera:
        s = - rhat · v_rel + noise,   noise ~ N(0, sigma_s_true^2)
    Jacobian: H = ∂s/∂r = -(v_rel - (rhat·v_rel) rhat)/||r||.
    Dla okna (window) trzymamy wkłady z ostatnich T sekund.

  • Jeśli FIM jest osobliwa (brak pobudzenia), CRLB liczymy z regularizacją:
        CRLB ≈ (FIM + eps*I)^(-1)

  • Kody nie używają prawdy do nagrody ani do sukcesu (to dalej jest po estymacie),
    ale w ewaluacji LOGUJEMY prawdę i możesz policzyć własne kryteria offline.

Autor: wygenerowane dla Twojego pipeline (Bank EKF + SAC/PPO).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
except Exception:
    pygame = None


# ===========================
# Helpers
# ===========================

def deg2rad(a: float) -> float:
    return a * math.pi / 180.0

def wrap360(a: float) -> float:
    a = a % 360.0
    if a < 0:
        a += 360.0
    return a

def vel_vec2(speed: float, heading_deg: float) -> np.ndarray:
    """
    2D:
      0° = +x (E), 90° = +y (N)
    """
    h = deg2rad(heading_deg)
    return np.array([speed * math.cos(h), speed * math.sin(h)], dtype=float)

def radial_speed(r_vec: np.ndarray, v_rel: np.ndarray) -> float:
    """
    Doppler / range-rate proxy:
      s = - rhat · v_rel
    """
    rho = float(np.linalg.norm(r_vec))
    if rho <= 1e-12:
        return 0.0
    rhat = r_vec / rho
    return -float(np.dot(rhat, v_rel))

def rot2(heading_deg: float) -> np.ndarray:
    c = math.cos(deg2rad(heading_deg))
    s = math.sin(deg2rad(heading_deg))
    return np.array([[c, -s], [s, c]], dtype=float)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def lerp(a: float, b: float, t: float) -> float:
    t = float(clamp(t, 0.0, 1.0))
    return (1.0 - t) * a + t * b

def sat_ratio(x: float, x0: float) -> float:
    """Saturating ratio in [0,1): x/(x+x0)."""
    x = float(max(0.0, x))
    x0 = float(max(1e-12, x0))
    return x / (x + x0)

def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Systematic resampling for particle filters.
    weights: shape (N,), sum to 1.
    Returns indices shape (N,)
    """
    N = int(weights.shape[0])
    positions = (rng.random() + np.arange(N)) / N
    cumulative_sum = np.cumsum(weights)
    indices = np.zeros(N, dtype=int)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
            if j >= N:
                j = N - 1
    return indices

def cov_ellipse_points(P: np.ndarray, k: float = 2.4477, n_pts: int = 48) -> np.ndarray:
    """
    Generate points of covariance ellipse for 2x2 covariance P.
    k ~ sqrt(chi2_0.95(df=2)) ≈ 2.4477 for 95%.
    Returns array (n_pts,2)
    """
    P = np.array(P, dtype=float)
    try:
        vals, vecs = np.linalg.eigh(P)
    except np.linalg.LinAlgError:
        return np.zeros((0, 2), dtype=float)
    vals = np.maximum(vals, 1e-12)
    axes = k * np.sqrt(vals)
    ang = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False)
    circle = np.stack([np.cos(ang) * axes[0], np.sin(ang) * axes[1]], axis=1)
    pts = circle @ vecs.T
    return pts

def doppler_H(r_vec: np.ndarray, v_rel: np.ndarray) -> np.ndarray:
    """
    Jacobian H = ∂s/∂r for s = -(r/||r||)·v_rel.
    Returns shape (1,2).
    """
    r_vec = np.array(r_vec, dtype=float)
    v_rel = np.array(v_rel, dtype=float)
    rho = float(np.linalg.norm(r_vec))
    if rho <= 1e-12:
        return np.zeros((1, 2), dtype=float)
    rhat = r_vec / rho
    proj = float(np.dot(rhat, v_rel))
    H = -(v_rel - proj * rhat) / rho
    return H.reshape(1, 2)



# ===========================
# Baseline controllers (for fair comparison vs RL)
# ===========================

def wrap180(a_deg: float) -> float:
    """Wrap degrees to [-180, 180)."""
    a = (float(a_deg) + 180.0) % 360.0 - 180.0
    return a

def _heading_from_cos_sin(c: float, s: float) -> float:
    """Return heading in degrees [0,360) from cos/sin."""
    ang = math.degrees(math.atan2(float(s), float(c)))
    return wrap360(ang)

def baseline_action_from_obs_raw(
    obs_raw_flat: np.ndarray,
    cfg: "UUVConfig",
    *,
    controller: str,
    kp: float,
    kexc: float,
    std_trigger_m: float,
) -> np.ndarray:
    """
    Baseline action computed ONLY from the same info the RL policy gets (obs).

    controller:
      - 'pid'     : formation P-controller in velocity space (tracks r_des using r_mix estimate)
      - 'pid_exc' : same + lateral excitation when uncertainty is high (to increase Doppler information)
      - 'random'  : uniform random action in [-1,1]^2

    Returns action (2,) in SB3 scaling: a_speed, a_turn in [-1,1].
    """
    controller = str(controller).lower().strip()

    # Ensure shape (14,)
    x = np.asarray(obs_raw_flat, dtype=float).reshape(-1)
    if x.size != 14:
        raise ValueError(f"Expected obs_raw dim=14, got {x.size}")
# 12332
    if controller == "random":
        return np.asarray([np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)], dtype=np.float32)

    # --- unpack observation (raw, unnormalized) ---
    pos_scale = float(cfg.pos_scale)
    std_scale = float(cfg.std_scale)
    vel_scale = float(cfg.vel_scale)

    e_mix = x[0:2] * pos_scale            # r_mix - r_des_world
    r_mix = x[2:4] * pos_scale            # estimate of r = pL - pF

    std_x = float(x[4] * std_scale)
    std_y = float(x[5] * std_scale)
    std_max = max(std_x, std_y)

    speed_F = float(x[7] * vel_scale)
    heading_F = _heading_from_cos_sin(x[8], x[9])

    leader_speed = float(x[11] * max(cfg.leader_speed_max, 1e-6))
    heading_L = _heading_from_cos_sin(x[12], x[13])

    vL = vel_vec2(leader_speed, heading_L)

    # --- formation hold in velocity space ---
    # r_dot = vL - vF ;   error e = r - r_des
    # Choose vF_cmd   = vL + k_p   * e  => e_dot = -k_p * e (stable)
    kp = float(max(0.0, kp))
    vF_cmd = vL + kp * e_mix

    # --- excitation to avoid Doppler degeneracy (low v_perp / weak observability) ---
    if controller == "pid_exc":
        std_thr = float(std_trigger_m) if float(std_trigger_m) > 0.0 else float(cfg.tol_std_hard)
        gain_info = sat_ratio(std_max - std_thr, max(std_thr, 1e-6))  # 0..1

        rho = float(np.linalg.norm(r_mix))
        if rho > 1e-6 and gain_info > 0.0:
            # Lateral component around leader: unit vector perpendicular to r_mix
            u_perp = np.array([-r_mix[1], r_mix[0]], dtype=float) / rho
            vF_cmd = vF_cmd + (float(kexc) * gain_info) * u_perp

        # Soft keep-away from leader (safety): push outward if too close
        rho_min = float(cfg.rho_soft_min)
        if rho > 1e-6 and rho < rho_min:
            gain_close = sat_ratio(rho_min - rho, rho_min)
            u_rad = r_mix / rho
            vF_cmd = vF_cmd + (float(kexc) * gain_close) * u_rad

    # --- map desired velocity vector -> desired (speed, heading) ---
    v_des = float(np.linalg.norm(vF_cmd))
    if v_des > 1e-6:
        heading_des = wrap360(math.degrees(math.atan2(float(vF_cmd[1]), float(vF_cmd[0]))))
    else:
        v_des = speed_F
        heading_des = heading_F

    v_des = float(np.clip(v_des, cfg.f_min_speed, cfg.f_max_speed))

    # --- map desired speed/heading to SB3 action scaling used in training ---
    # in env.step(): speed changes by rl_speed_delta_per_step per RL step when |a_speed|=1
    a_speed = clamp((v_des - speed_F) / max(cfg.rl_speed_delta_per_step, 1e-9), -1.0, 1.0)

    dh = wrap180(heading_des - heading_F)  # shortest signed angle
    a_turn = clamp(dh / max(cfg.rl_turn_per_step_deg, 1e-9), -1.0, 1.0)

    return np.asarray([a_speed, a_turn], dtype=np.float32)


# ===========================
# FIM / CRLB tracker
# ===========================

class FIMTracker2D:
    """
    Tracks Fisher Information Matrix for 2D state (r).
    Supports cumulative FIM and sliding-window FIM over last `window_s` seconds.
    """

    def __init__(self, window_s: Optional[float], reg_eps: float = 1e-9):
        self.window_s = None if window_s is None else float(max(0.0, window_s))
        self.reg_eps = float(max(0.0, reg_eps))

        self.I_total = np.zeros((2, 2), dtype=float)
        self.I_win = np.zeros((2, 2), dtype=float)
        self._events: List[Tuple[float, np.ndarray]] = []

    def reset(self) -> None:
        self.I_total[:] = 0.0
        self.I_win[:] = 0.0
        self._events = []

    def add_I(self, t_meas: float, I_inc: np.ndarray) -> None:
        t_meas = float(t_meas)
        I_inc = np.array(I_inc, dtype=float).reshape(2, 2)
        self.I_total += I_inc

        if self.window_s is None or self.window_s <= 0.0:
            self.I_win = self.I_total.copy()
            return

        # sliding window
        self._events.append((t_meas, I_inc))
        self.I_win += I_inc

        t_min = t_meas - self.window_s
        # prune old events
        while self._events and self._events[0][0] < t_min:
            _, I_old = self._events.pop(0)
            self.I_win -= I_old

        self.I_win = 0.5 * (self.I_win + self.I_win.T)

    def crlb(self, use_window: bool) -> np.ndarray:
        I = self.I_win if use_window else self.I_total
        I = 0.5 * (I + I.T)
        I_reg = I + self.reg_eps * np.eye(2, dtype=float)
        try:
            C = np.linalg.inv(I_reg)
        except np.linalg.LinAlgError:
            C = np.full((2, 2), np.nan, dtype=float)
        return 0.5 * (C + C.T)

    @staticmethod
    def eig_stats(I: np.ndarray) -> Tuple[float, float, float]:
        """
        Returns: (eig_min, eig_max, cond)
        """
        I = 0.5 * (I + I.T)
        try:
            eig = np.linalg.eigvalsh(I)
            eig_min = float(max(eig[0], 0.0))
            eig_max = float(max(eig[1], 0.0))
        except np.linalg.LinAlgError:
            return float("nan"), float("nan"), float("nan")

        if eig_min <= 1e-18:
            cond = float("inf") if eig_max > 0 else float("nan")
        else:
            cond = float(eig_max / eig_min)
        return eig_min, eig_max, cond


# ===========================
# EKF 2D
# ===========================

class EKFRelPos2D:
    """
    State: r = pL - pF (2D)
    Predict: r_{k+1} = r_k + dt * v_rel
    Measurement: s = -rhat · v_rel (1D)
    """
    def __init__(
        self,
        r0: np.ndarray,
        pos_std0: float,
        q_std: float,
        sigma_s_filter: float,
    ) -> None:
        self.r_hat = np.array(r0, dtype=float)
        self.P = np.diag([pos_std0**2, pos_std0**2]).astype(float)

        self.q_var = float(q_std**2)
        self.set_sigma_s_filter(float(sigma_s_filter))

        self.last_innov: float = float("nan")
        self.last_S: float = float("nan")
        self.last_nis: float = float("nan")
        self.last_h: float = float("nan")
        self.last_z: float = float("nan")

    def set_sigma_s_filter(self, sigma_s_filter: float) -> None:
        sigma = float(max(1e-9, sigma_s_filter))
        self.Rs = np.array([[sigma**2]], dtype=float)

    def clone(self) -> "EKFRelPos2D":
        c = EKFRelPos2D(
            r0=self.r_hat.copy(),
            pos_std0=1.0,
            q_std=math.sqrt(self.q_var),
            sigma_s_filter=float(math.sqrt(self.Rs[0, 0])),
        )
        c.P = self.P.copy()
        c.q_var = float(self.q_var)
        c.Rs = self.Rs.copy()

        c.last_innov = float(self.last_innov)
        c.last_S = float(self.last_S)
        c.last_nis = float(self.last_nis)
        c.last_h = float(self.last_h)
        c.last_z = float(self.last_z)
        return c

    def predict(self, v_rel: np.ndarray, dt: float) -> None:
        v_rel = np.array(v_rel, dtype=float)
        dt = float(dt)

        self.r_hat = self.r_hat + dt * v_rel
        Q = self.q_var * max(dt, 1e-3) * np.eye(2, dtype=float)
        self.P = self.P + Q

    def _innovation(self, v_rel: np.ndarray, s_meas: float) -> Tuple[float, float, float, np.ndarray]:
        v_rel = np.array(v_rel, dtype=float)
        rho = float(np.linalg.norm(self.r_hat))
        if rho <= 1e-12:
            return float("nan"), float("nan"), float("nan"), np.zeros((1, 2), dtype=float)

        rhat = self.r_hat / rho
        proj = float(np.dot(rhat, v_rel))
        h = -proj
        H = -(v_rel - proj * rhat) / rho
        H = H.reshape(1, 2)

        z = float(s_meas)
        y = z - h
        S = float((H @ self.P @ H.T + self.Rs)[0, 0])
        return y, S, h, H

    def update_s(
        self,
        v_rel: np.ndarray,
        s_meas: float,
        rho_min: float,
        v_min: float,
        v_perp_min: float,
    ) -> bool:
        v_rel = np.array(v_rel, dtype=float)
        vnorm = float(np.linalg.norm(v_rel))
        rho = float(np.linalg.norm(self.r_hat))
        if rho <= rho_min or vnorm <= v_min:
            self.last_innov = float("nan")
            self.last_S = float("nan")
            self.last_nis = float("nan")
            self.last_h = float("nan")
            self.last_z = float("nan")
            return False

        rhat = self.r_hat / rho
        proj = float(np.dot(rhat, v_rel))
        v_perp = math.sqrt(max(0.0, vnorm * vnorm - proj * proj))
        if v_perp <= v_perp_min:
            self.last_innov = float("nan")
            self.last_S = float("nan")
            self.last_nis = float("nan")
            self.last_h = float("nan")
            self.last_z = float("nan")
            return False

        y, S, h, H = self._innovation(v_rel=v_rel, s_meas=s_meas)
        if not np.isfinite(S) or S <= 1e-12:
            return False

        nis = (y * y) / S

        self.last_innov = float(y)
        self.last_S = float(S)
        self.last_nis = float(nis)
        self.last_h = float(h)
        self.last_z = float(s_meas)

        Sinv = 1.0 / S
        K = (self.P @ H.T) * Sinv
        self.r_hat = self.r_hat + (K.flatten() * y)

        I = np.eye(2)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.Rs @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True


# ===========================
# Bank EKF + rejuvenation
# ===========================

@dataclass
class BankMetrics:
    updated: int = 0
    nis_mix: float = float("nan")
    sens_vperp: float = 0.0
    w_max: float = 0.0
    ess: float = 0.0
    rejuv: int = 0

class BankEKFRelPos2D:
    def __init__(
        self,
        num_hyp: int,
        q_std: float,
        sigma_s_filter: float,
        pos_std0: float,
        rng: np.random.Generator,
    ) -> None:
        self.N = int(num_hyp)
        self.rng = rng

        self.filters: List[EKFRelPos2D] = []
        for _ in range(self.N):
            r0 = np.zeros(2, dtype=float)
            self.filters.append(EKFRelPos2D(r0=r0, pos_std0=pos_std0, q_std=q_std, sigma_s_filter=sigma_s_filter))

        self.w = np.ones(self.N, dtype=float) / self.N

        self.r_mix = np.zeros(2, dtype=float)
        self.P_mix = np.eye(2, dtype=float) * (pos_std0**2)

        self.last_updated: int = 0
        self.last_nis_mix: float = float("nan")
        self.last_sens_vperp: float = 0.0

        self.rejuv_count_ep: int = 0
        self._updates_since_rejuv: int = 0

        self._recompute_mixture()

    def set_sigma_s_filter(self, sigma_s_filter: float) -> None:
        for f in self.filters:
            f.set_sigma_s_filter(sigma_s_filter)

    def reset_hypotheses(self, r0_list: List[np.ndarray], pos_std0: float) -> None:
        assert len(r0_list) == self.N
        for i, r0 in enumerate(r0_list):
            self.filters[i].r_hat = np.array(r0, dtype=float)
            self.filters[i].P = np.diag([pos_std0**2, pos_std0**2]).astype(float)
            self.filters[i].last_innov = float("nan")
            self.filters[i].last_S = float("nan")
            self.filters[i].last_nis = float("nan")
            self.filters[i].last_h = float("nan")
            self.filters[i].last_z = float("nan")

        self.w[:] = 1.0 / self.N
        self.rejuv_count_ep = 0
        self._updates_since_rejuv = 0
        self._recompute_mixture()

    def predict(self, v_rel: np.ndarray, dt: float) -> None:
        for f in self.filters:
            f.predict(v_rel=v_rel, dt=dt)
        self._recompute_mixture()

    def _recompute_mixture(self) -> None:
        w = self.w
        r_stack = np.stack([f.r_hat for f in self.filters], axis=0)
        r_mix = np.sum(r_stack * w[:, None], axis=0)
        P_mix = np.zeros((2, 2), dtype=float)
        for i, f in enumerate(self.filters):
            dr = (f.r_hat - r_mix).reshape(2, 1)
            P_mix += w[i] * (f.P + (dr @ dr.T))
        self.r_mix = r_mix
        self.P_mix = 0.5 * (P_mix + P_mix.T)

    def _bank_stats(self) -> Tuple[float, float]:
        w_max = float(np.max(self.w))
        ess = float(1.0 / np.sum(np.square(self.w)))
        return w_max, ess

    def doppler_update(
        self,
        v_rel: np.ndarray,
        s_meas: float,
        *,
        rho_min: float,
        v_min: float,
        v_perp_min: float,
        likelihood_temp: float,
        loglik_clip: float,
        weight_floor: float,
        weight_forget: float,
        rejuv_enabled: bool,
        rejuv_ess_frac: float,
        rejuv_wmax: float,
        rejuv_std_trigger_min: float,
        rejuv_jitter_r_m: float,
        rejuv_inflate_P_m: float,
        rejuv_keep_best: int,
        rejuv_min_updates_between: int,
    ) -> BankMetrics:
        v_rel = np.array(v_rel, dtype=float)
        vnorm = float(np.linalg.norm(v_rel))
        rho_mix = float(np.linalg.norm(self.r_mix))
        if rho_mix <= rho_min or vnorm <= v_min:
            self.last_updated = 0
            self.last_nis_mix = float("nan")
            self.last_sens_vperp = 0.0
            self._recompute_mixture()
            w_max, ess = self._bank_stats()
            return BankMetrics(updated=0, nis_mix=float("nan"), sens_vperp=0.0, w_max=w_max, ess=ess, rejuv=0)

        rhat_mix = self.r_mix / max(rho_mix, 1e-12)
        proj_mix = float(np.dot(rhat_mix, v_rel))
        v_perp_mix = math.sqrt(max(0.0, vnorm * vnorm - proj_mix * proj_mix))
        self.last_sens_vperp = float(v_perp_mix)

        if v_perp_mix <= v_perp_min:
            self.last_updated = 0
            self.last_nis_mix = float("nan")
            self._recompute_mixture()
            w_max, ess = self._bank_stats()
            return BankMetrics(updated=0, nis_mix=float("nan"), sens_vperp=float(v_perp_mix), w_max=w_max, ess=ess, rejuv=0)

        loglik = np.zeros(self.N, dtype=float)
        nis_vals = np.full(self.N, np.nan, dtype=float)

        for i, f in enumerate(self.filters):
            y, S, h, H = f._innovation(v_rel=v_rel, s_meas=s_meas)
            if not np.isfinite(S) or S <= 1e-12:
                loglik[i] = 0.0
                continue
            ll = -0.5 * (math.log(2.0 * math.pi * S) + (y * y) / S)
            ll = float(clamp(ll, -loglik_clip, loglik_clip))
            loglik[i] = ll
            nis_vals[i] = float((y * y) / S)

        temp = float(max(1e-6, likelihood_temp))
        logw = np.log(np.maximum(self.w, 1e-12)) + (loglik / temp)
        logw -= np.max(logw)
        w_new = np.exp(logw)
        w_new_sum = float(np.sum(w_new))
        if w_new_sum <= 1e-12:
            w_new = np.ones(self.N, dtype=float) / self.N
        else:
            w_new /= w_new_sum

        floor = float(max(0.0, weight_floor))
        if floor > 0.0:
            w_new = np.maximum(w_new, floor)
            w_new /= float(np.sum(w_new))

        forget = float(clamp(weight_forget, 0.0, 1.0))
        if forget > 0.0:
            w_new = (1.0 - forget) * w_new + forget * (1.0 / self.N)
            w_new /= float(np.sum(w_new))

        self.w = w_new

        updated_flags = np.zeros(self.N, dtype=int)
        for i, f in enumerate(self.filters):
            did = f.update_s(v_rel=v_rel, s_meas=s_meas, rho_min=rho_min, v_min=v_min, v_perp_min=v_perp_min)
            updated_flags[i] = 1 if did else 0

        any_update = int(np.any(updated_flags == 1))
        self.last_updated = any_update
        self._updates_since_rejuv += any_update

        nis_mix = float("nan")
        valid = np.isfinite(nis_vals)
        if np.any(valid):
            nis_mix = float(np.sum(self.w[valid] * nis_vals[valid]) / np.sum(self.w[valid]))
        self.last_nis_mix = nis_mix

        self._recompute_mixture()

        rejuv_flag = 0
        if rejuv_enabled and any_update == 1:
            rejuv_flag = self._maybe_rejuvenate(
                ess_frac=rejuv_ess_frac,
                wmax_thr=rejuv_wmax,
                std_trigger_min=rejuv_std_trigger_min,
                jitter_r_m=rejuv_jitter_r_m,
                inflate_P_m=rejuv_inflate_P_m,
                keep_best=rejuv_keep_best,
                min_updates_between=rejuv_min_updates_between,
            )

        w_max, ess = self._bank_stats()
        return BankMetrics(updated=any_update, nis_mix=nis_mix, sens_vperp=float(v_perp_mix), w_max=w_max, ess=ess, rejuv=rejuv_flag)

    def _maybe_rejuvenate(
        self,
        *,
        ess_frac: float,
        wmax_thr: float,
        std_trigger_min: float,
        jitter_r_m: float,
        inflate_P_m: float,
        keep_best: int,
        min_updates_between: int,
    ) -> int:
        N = self.N
        w = self.w
        w_max = float(np.max(w))
        ess = float(1.0 / np.sum(np.square(w)))
        ess_thr = float(clamp(ess_frac, 0.0, 1.0)) * N

        std_x = float(math.sqrt(max(self.P_mix[0, 0], 1e-12)))
        std_y = float(math.sqrt(max(self.P_mix[1, 1], 1e-12)))
        std_max_mix = max(std_x, std_y)

        if self._updates_since_rejuv < int(max(1, min_updates_between)):
            return 0
        if std_max_mix < float(std_trigger_min):
            return 0
        if not (ess < ess_thr or w_max > float(wmax_thr)):
            return 0

        idx = systematic_resample(w, self.rng)

        keep_best = int(max(0, min(keep_best, N)))
        best_idx = int(np.argmax(w))
        if keep_best > 0:
            idx[:keep_best] = best_idx

        new_filters: List[EKFRelPos2D] = []
        for k in range(N):
            src = self.filters[int(idx[k])]
            ekf_new = src.clone()

            if not (keep_best > 0 and k == 0 and int(idx[k]) == best_idx):
                jitter = self.rng.normal(0.0, float(jitter_r_m), size=(2,))
                ekf_new.r_hat = ekf_new.r_hat + jitter
                infl = float(inflate_P_m)
                ekf_new.P = ekf_new.P + (infl * infl) * np.eye(2)

            new_filters.append(ekf_new)

        self.filters = new_filters
        self.w = np.ones(N, dtype=float) / N
        self._recompute_mixture()

        self.rejuv_count_ep += 1
        self._updates_since_rejuv = 0
        return 1


# ===========================
# Config
# ===========================

@dataclass
class UUVConfig:
    # ===== render =====
    screen_w: int = 1100
    screen_h: int = 800
    render_fps: int = 30

    # ===== follower limits =====
    f_min_speed: float = 0.2
    f_max_speed: float = 4.0
    max_turn_rate_deg_s: float = 80.0
    max_accel_m_s2: float = 1.0

    # RL action scaling (kept same as your training)
    rl_turn_per_step_deg: float = 10.0
    rl_speed_delta_per_step: float = 0.2

    # manual control
    manual_turn_per_step_deg: float = 5.0
    manual_speed_delta_per_step: float = 0.05
    manual_fine_scale: float = 0.25

    # ===== leader randomization =====
    leader_speed_min: float = 1.0
    leader_speed_max: float = 3.0
    leader_heading_min_deg: float = 0.0
    leader_heading_max_deg: float = 360.0

    leader_speed_min_easy: float = 1.8
    leader_speed_max_easy: float = 2.2
    leader_heading_min_easy: float = 0.0
    leader_heading_max_easy: float = 60.0

    # ===== initial relative start =====
    start_rho_min: float = 80.0
    start_rho_max: float = 220.0
    start_rho_min_easy: float = 90.0
    start_rho_max_easy: float = 110.0

    # desired relative position in leader body frame
    r_des_body: Tuple[float, float] = (100.0, 0.0)

    # ===== timing =====
    action_dt: float = 2.0
    sub_dt: float = 0.1
    max_steps: int = 200

    # ===== measurement =====
    sigma_s_true: float = 0.015
    s_meas_period: float = 1.0

    sigma_heading_deg: float = 0.5
    sigma_speed: float = 0.01

    # ===== EKF bank =====
    ekf_q_std: float = 0.025
    ekf_init_pos_std_hard: float = 300.0
    ekf_init_pos_std_easy: float = 30.0
    ekf_init_rho_guess: float = 150.0
    sigma_s_filter_mult_hard: float = 2.5
    sigma_s_filter_mult_easy: float = 1.0

    # Doppler gating (for the filter)
    doppler_rho_min: float = 20.0
    doppler_v_min: float = 0.05
    doppler_v_perp_min_easy: float = 0.0
    doppler_v_perp_min_hard: float = 0.25

    # success conditions (estimator-based)
    tol_pos_est_easy: float = 30.0
    tol_pos_est_hard: float = 5.0
    tol_std_easy: float = 80.0
    tol_std_hard: float = 6.0
    success_hold_steps: int = 3

    # reward shaping (unchanged)
    w_pos: float = 2.0
    w_info: float = 1.0
    w_sens: float = 0.3
    w_nis: float = 0.5
    w_close: float = 0.5
    terminal_bonus: float = 5.0

    pos_err0_m: float = 50.0
    unc_red0_m: float = 0.5
    sens0_m_s: float = 0.5
    nis_excess0: float = 10.0
    close0_m: float = 5.0

    rho_soft_min: float = 15.0

    nis_95: float = 3.84
    nis_99: float = 6.63

    pos_scale: float = 200.0
    std_scale: float = 200.0
    vel_scale: float = 4.0
    s_scale: float = 3.0

    bank_num_hyp: int = 8
    bank_likelihood_temp: float = 1.5
    bank_loglik_clip: float = 60.0
    bank_weight_floor: float = 1e-4
    bank_weight_forget: float = 0.002

    bank_rejuv_enabled: bool = True
    bank_rejuv_ess_frac: float = 0.55
    bank_rejuv_wmax: float = 0.90
    bank_rejuv_std_trigger_min: float = 15.0
    bank_rejuv_jitter_r_m: float = 6.0
    bank_rejuv_inflate_P_m: float = 10.0
    bank_rejuv_keep_best: int = 1
    bank_rejuv_min_updates_between: int = 3

    init_known_noise_easy: float = 5.0
    init_known_noise_hard: float = 30.0

    cov_ellipse_k: float = 2.4477
    cov_plot_hist_len: int = 300

    log_truth_diagnostics: bool = False
    curriculum_steps: int = 3_000_000

    # Patch 2: stall penalty
    w_stall: float = 0.20
    stall0_steps: float = 5.0

    # Patch 3: boost domykania
    close_goal_pos_mult: float = 2.0
    info_sens_boost: float = 3.0

    # ===== NEW: FIM/CRLB evaluation =====
    fim_window_s: float = 30.0      # sliding window length for FIM/CRLB
    fim_reg_eps: float = 1e-9       # regularization for CRLB = inv(FIM + eps I)
    fim_use_gating: bool = True     # count only measurements that pass Doppler gating


# ===========================
# Camera (render)
# ===========================

class Camera:
    def __init__(self, w: int, h: int, scale_init: float = 2.0, scale_min: float = 0.4, scale_max: float = 8.0):
        self.w = int(w)
        self.h = int(h)
        self.center = np.zeros(2, dtype=float)
        self.scale = float(scale_init)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

    def update(self, points: List[np.ndarray]) -> None:
        pts = np.vstack(points)
        center = pts.mean(axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        d_max = float(max(dists.max(), 10.0))
        target_px = 0.40 * min(self.w, self.h)
        target_scale = target_px / d_max
        target_scale = max(self.scale_min, min(self.scale_max, target_scale))

        alpha = 0.15
        self.center = (1 - alpha) * self.center + alpha * center
        self.scale = (1 - alpha) * self.scale + alpha * target_scale

    def world_to_screen(self, pos: np.ndarray) -> Tuple[int, int]:
        rel = np.array(pos, dtype=float) - self.center
        sx = self.w / 2 + rel[0] * self.scale
        sy = self.h / 2 - rel[1] * self.scale
        return int(sx), int(sy)


# ===========================
# Env
# ===========================

class UUVRelPosBankEnv(gym.Env):
    """
    Observation (14D) – zgodne z Twoim treningiem:
      0-1: e_mix = (r_mix - r_des_world) / pos_scale
      2-3: r_mix / pos_scale
      4-5: std_x, std_y from P_mix / std_scale
      6:   Pxy from P_mix / std_scale^2
      7:   speed_F / vel_scale
      8-9: cos(heading_F), sin(heading_F)
      10:  s_meas_last / s_scale
      11:  leader_speed / leader_speed_max(hard)
      12-13: cos(heading_L), sin(heading_L)
    """
    metadata = {"render_modes": ["human", "none"], "render_fps": 30}

    def __init__(self, cfg: Optional[UUVConfig] = None, render_mode: str = "none"):
        super().__init__()
        self.cfg = cfg or UUVConfig()
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

        self.rng = np.random.default_rng(0)

        self._difficulty_global = 1.0
        self._difficulty_ep = 1.0

        self.t = 0.0
        self.step_count = 0
        self.success_streak = 0

        self.pL = np.zeros(2, dtype=float)
        self.pF = np.zeros(2, dtype=float)

        self.leader_speed = 2.0
        self.heading_L = 0.0

        self.heading_F = 0.0
        self.speed_F = 2.0

        self.r_des_world = np.array(self.cfg.r_des_body, dtype=float)

        sigma_s_filter0 = self.cfg.sigma_s_true * self.cfg.sigma_s_filter_mult_hard
        self.bank = BankEKFRelPos2D(
            num_hyp=self.cfg.bank_num_hyp,
            q_std=self.cfg.ekf_q_std,
            sigma_s_filter=sigma_s_filter0,
            pos_std0=self.cfg.ekf_init_pos_std_hard,
            rng=self.rng,
        )

        self.next_s_time = self.cfg.s_meas_period
        self.s_meas_last = 0.0

        self.prev_unc_metric: Optional[float] = None

        # Patch 1: sens avg gęsto
        self.sens_accum: float = 0.0
        self.sens_count: int = 0

        # Patch 2: stall / brak update Dopplera
        self.no_update_steps: int = 0
        self.doppler_meas_step: int = 0
        self.doppler_updates_step: int = 0

        # ===== NEW: FIM tracking =====
        self.fim_total = FIMTracker2D(window_s=None, reg_eps=self.cfg.fim_reg_eps)
        self.fim_win = FIMTracker2D(window_s=self.cfg.fim_window_s, reg_eps=self.cfg.fim_reg_eps)
        self.fim_step_I = np.zeros((2, 2), dtype=float)
        self.fim_step_meas = 0
        self.fim_step_used = 0

        # manual override / UI
        self.manual_override = False
        self._request_reset = False
        self._request_quit = False

        self._pygame_inited = False
        self._screen = None
        self._clock = None
        self._font = None
        self._cam = Camera(self.cfg.screen_w, self.cfg.screen_h)

        self._std_hist_x: List[float] = []
        self._std_hist_y: List[float] = []

    def set_difficulty(self, d: float) -> None:
        self._difficulty_global = float(clamp(d, 0.0, 1.0))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
            self.bank.rng = self.rng

        self._difficulty_ep = float(self._difficulty_global)

        self.t = 0.0
        self.step_count = 0
        self.success_streak = 0

        self._request_reset = False
        self._request_quit = False

        # reset FIM trackers
        self.fim_total.reset()
        self.fim_win.reset()
        self.fim_step_I[:] = 0.0
        self.fim_step_meas = 0
        self.fim_step_used = 0

        d = self._difficulty_ep

        vL_min = lerp(self.cfg.leader_speed_min_easy, self.cfg.leader_speed_min, d)
        vL_max = lerp(self.cfg.leader_speed_max_easy, self.cfg.leader_speed_max, d)
        hL_min = lerp(self.cfg.leader_heading_min_easy, self.cfg.leader_heading_min_deg, d)
        hL_max = lerp(self.cfg.leader_heading_max_easy, self.cfg.leader_heading_max_deg, d)

        self.leader_speed = float(self.rng.uniform(vL_min, vL_max))
        self.heading_L = float(self.rng.uniform(hL_min, hL_max))

        self.r_des_world = rot2(self.heading_L) @ np.array(self.cfg.r_des_body, dtype=float)

        self.pL[:] = 0.0

        rho0_min = lerp(self.cfg.start_rho_min_easy, self.cfg.start_rho_min, d)
        rho0_max = lerp(self.cfg.start_rho_max_easy, self.cfg.start_rho_max, d)
        rho0 = float(self.rng.uniform(rho0_min, rho0_max))
        ang0 = float(self.rng.uniform(0.0, 2.0 * math.pi))
        r_true0 = rho0 * np.array([math.cos(ang0), math.sin(ang0)], dtype=float)

        self.pF = self.pL - r_true0

        self.heading_F = float(self.rng.uniform(0.0, 360.0))
        self.speed_F = 2.0

        sigma_mult = lerp(self.cfg.sigma_s_filter_mult_easy, self.cfg.sigma_s_filter_mult_hard, d)
        sigma_s_filter = float(self.cfg.sigma_s_true * sigma_mult)
        self.bank.set_sigma_s_filter(sigma_s_filter)

        pos_std0 = lerp(self.cfg.ekf_init_pos_std_easy, self.cfg.ekf_init_pos_std_hard, d)

        p_known = lerp(1.0, 0.0, d)
        init_known_used = 1 if (self.rng.random() < p_known) else 0
        noise_std = lerp(self.cfg.init_known_noise_easy, self.cfg.init_known_noise_hard, d)

        r0_list: List[np.ndarray] = []
        if init_known_used == 1:
            for _ in range(self.cfg.bank_num_hyp):
                r0 = r_true0 + self.rng.normal(0.0, noise_std, size=(2,))
                r0_list.append(r0.astype(float))
        else:
            rho_guess = float(self.cfg.ekf_init_rho_guess)
            base_ang = float(self.rng.uniform(0.0, 2.0 * math.pi))
            for i in range(self.cfg.bank_num_hyp):
                ang = base_ang + (2.0 * math.pi) * (i / self.cfg.bank_num_hyp)
                r0 = rho_guess * np.array([math.cos(ang), math.sin(ang)], dtype=float)
                r0_list.append(r0.astype(float))

        self.bank.reset_hypotheses(r0_list=r0_list, pos_std0=float(pos_std0))

        self.next_s_time = float(self.cfg.s_meas_period)
        self.s_meas_last = 0.0

        std_x, std_y = self._stds_from_Pmix()
        self.prev_unc_metric = float(std_x + std_y)

        self.sens_accum = 0.0
        self.sens_count = 0

        self.no_update_steps = 0
        self.doppler_meas_step = 0
        self.doppler_updates_step = 0

        self._std_hist_x = []
        self._std_hist_y = []

        obs = self._get_obs()
        info = self._get_info(extra={
            "difficulty": float(d),
            "init_known_used": int(init_known_used),
            "sigma_s_filter": float(sigma_s_filter),
            "pos_std0": float(pos_std0),
        })
        return obs, info

    def step(self, action: np.ndarray):
        if self.render_mode == "human" and self._pygame_inited and pygame is not None:
            self._process_pygame_events()

        if self._request_quit:
            self._request_quit = False
            obs = self._get_obs()
            info = self._get_info(extra={"term_reason": "quit"})
            return obs, 0.0, False, True, info

        if self._request_reset:
            self._request_reset = False
            obs = self._get_obs()
            info = self._get_info(extra={"term_reason": "manual_reset"})
            return obs, 0.0, False, True, info

        # reset per-step FIM accumulators
        self.fim_step_I[:] = 0.0
        self.fim_step_meas = 0
        self.fim_step_used = 0

        # sterowanie
        if self.render_mode == "human" and self.manual_override and self._pygame_inited and pygame is not None:
            a = self._keyboard_action()
            acc_cmd = float(a[0]) * (self.cfg.manual_speed_delta_per_step / max(self.cfg.action_dt, 1e-6))
            turn_cmd = float(a[1]) * (self.cfg.manual_turn_per_step_deg / max(self.cfg.action_dt, 1e-6))
        else:
            a = np.asarray(action, dtype=np.float32)
            a = np.clip(a, -1.0, 1.0)

            acc_from_step = float(self.cfg.rl_speed_delta_per_step) / max(self.cfg.action_dt, 1e-6)
            acc_limit = min(float(self.cfg.max_accel_m_s2), acc_from_step)
            acc_cmd = float(a[0]) * acc_limit

            turn_rate_from_step = float(self.cfg.rl_turn_per_step_deg) / max(self.cfg.action_dt, 1e-6)
            turn_rate_limit = min(float(self.cfg.max_turn_rate_deg_s), turn_rate_from_step)
            turn_cmd = float(a[1]) * turn_rate_limit

        n_sub = max(1, int(round(self.cfg.action_dt / self.cfg.sub_dt)))
        dt = float(self.cfg.action_dt / n_sub)

        # per-krok RL
        self.sens_accum = 0.0
        self.sens_count = 0
        self.doppler_meas_step = 0
        self.doppler_updates_step = 0

        bank_metrics_last: Optional[BankMetrics] = None
        for _ in range(n_sub):
            bank_metrics_last = self._sim_substep(acc_cmd=acc_cmd, turn_cmd=turn_cmd, dt=dt)

        self.step_count += 1

        obs = self._get_obs()
        reward, terms = self._compute_reward(bank_metrics_last)
        terminated, truncated, reason = self._check_done()

        if terminated and reason == "success":
            reward += float(self.cfg.terminal_bonus)
            terms["terminal_bonus"] = float(self.cfg.terminal_bonus)
        else:
            terms["terminal_bonus"] = 0.0

        info = self._get_info(extra=terms | {"term_reason": reason, "manual_override": int(self.manual_override)})
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _sim_substep(self, acc_cmd: float, turn_cmd: float, dt: float) -> BankMetrics:
        self.speed_F = float(np.clip(self.speed_F + acc_cmd * dt, self.cfg.f_min_speed, self.cfg.f_max_speed))
        self.heading_F = wrap360(self.heading_F + turn_cmd * dt)

        vL_true = vel_vec2(self.leader_speed, self.heading_L)
        vF_true = vel_vec2(self.speed_F, self.heading_F)

        self.pL = self.pL + vL_true * dt
        self.pF = self.pF + vF_true * dt

        self.t += dt

        # nawigacyjne błędy followera
        heading_meas = self.heading_F + float(self.rng.normal(0.0, self.cfg.sigma_heading_deg))
        speed_meas = self.speed_F + float(self.rng.normal(0.0, self.cfg.sigma_speed))
        vF_ekf = vel_vec2(speed_meas, heading_meas)
        v_rel_ekf = vL_true - vF_ekf

        self.bank.predict(v_rel=v_rel_ekf, dt=dt)

        bank_metrics = BankMetrics(updated=0, nis_mix=float("nan"), sens_vperp=0.0, w_max=0.0, ess=0.0, rejuv=0)

        d = self._difficulty_ep
        v_perp_min_eff = lerp(self.cfg.doppler_v_perp_min_easy, self.cfg.doppler_v_perp_min_hard, d)

        while self.t + 1e-12 >= self.next_s_time:
            t_meas = float(self.next_s_time)

            # prawda dla pomiaru Dopplera (symulator)
            r_true = self.pL - self.pF
            v_rel_true = vL_true - vF_true
            s_true = radial_speed(r_true, v_rel_true)
            s_meas = s_true + float(self.rng.normal(0.0, self.cfg.sigma_s_true))
            self.s_meas_last = float(s_meas)

            # === NEW: FIM increment on TRUTH ===
            H = doppler_H(r_true, v_rel_true)  # (1,2)
            I_inc = (H.T @ H) / max(self.cfg.sigma_s_true**2, 1e-18)  # (2,2)

            # gating (optional) for FIM_used
            rho = float(np.linalg.norm(r_true))
            vnorm = float(np.linalg.norm(v_rel_true))
            proj = 0.0
            v_perp = 0.0
            if rho > 1e-12 and vnorm > 1e-12:
                rhat = r_true / rho
                proj = float(np.dot(rhat, v_rel_true))
                v_perp = math.sqrt(max(0.0, vnorm * vnorm - proj * proj))

            gate_ok = (rho > float(self.cfg.doppler_rho_min)) and (vnorm > float(self.cfg.doppler_v_min)) and (v_perp > float(v_perp_min_eff))

            self.fim_step_meas += 1
            if (not self.cfg.fim_use_gating) or gate_ok:
                self.fim_total.add_I(t_meas, I_inc)
                self.fim_win.add_I(t_meas, I_inc)
                self.fim_step_I += I_inc
                self.fim_step_used += 1

            # === filter update ===
            bank_metrics = self.bank.doppler_update(
                v_rel=v_rel_ekf,
                s_meas=s_meas,
                rho_min=float(self.cfg.doppler_rho_min),
                v_min=float(self.cfg.doppler_v_min),
                v_perp_min=float(v_perp_min_eff),
                likelihood_temp=float(self.cfg.bank_likelihood_temp),
                loglik_clip=float(self.cfg.bank_loglik_clip),
                weight_floor=float(self.cfg.bank_weight_floor),
                weight_forget=float(self.cfg.bank_weight_forget),
                rejuv_enabled=bool(self.cfg.bank_rejuv_enabled),
                rejuv_ess_frac=float(self.cfg.bank_rejuv_ess_frac),
                rejuv_wmax=float(self.cfg.bank_rejuv_wmax),
                rejuv_std_trigger_min=float(self.cfg.bank_rejuv_std_trigger_min),
                rejuv_jitter_r_m=float(self.cfg.bank_rejuv_jitter_r_m),
                rejuv_inflate_P_m=float(self.cfg.bank_rejuv_inflate_P_m),
                rejuv_keep_best=int(self.cfg.bank_rejuv_keep_best),
                rejuv_min_updates_between=int(self.cfg.bank_rejuv_min_updates_between),
            )

            # Patch 2: liczniki “czy był update”
            self.doppler_meas_step += 1
            self.doppler_updates_step += int(bank_metrics.updated)

            # Patch 1: sens avg GĘSTE — zawsze dodaj v_perp
            self.sens_accum += float(bank_metrics.sens_vperp)
            self.sens_count += 1

            self.next_s_time += float(self.cfg.s_meas_period)

        return bank_metrics

    def _stds_from_Pmix(self) -> Tuple[float, float]:
        P = self.bank.P_mix
        std_x = float(math.sqrt(max(P[0, 0], 1e-12)))
        std_y = float(math.sqrt(max(P[1, 1], 1e-12)))
        return std_x, std_y

    def _get_obs(self) -> np.ndarray:
        r_mix = self.bank.r_mix
        e_mix = r_mix - self.r_des_world

        P = self.bank.P_mix
        std_x, std_y = self._stds_from_Pmix()
        pxy = float(P[0, 1])

        obs = np.array([
            e_mix[0] / self.cfg.pos_scale,
            e_mix[1] / self.cfg.pos_scale,
            r_mix[0] / self.cfg.pos_scale,
            r_mix[1] / self.cfg.pos_scale,
            std_x / self.cfg.std_scale,
            std_y / self.cfg.std_scale,
            pxy / (self.cfg.std_scale ** 2),
            self.speed_F / self.cfg.vel_scale,
            math.cos(deg2rad(self.heading_F)),
            math.sin(deg2rad(self.heading_F)),
            self.s_meas_last / self.cfg.s_scale,
            self.leader_speed / max(self.cfg.leader_speed_max, 1e-6),
            math.cos(deg2rad(self.heading_L)),
            math.sin(deg2rad(self.heading_L)),
        ], dtype=np.float32)
        return obs

    def _compute_reward(self, bank_metrics_last: Optional[BankMetrics]) -> Tuple[float, Dict[str, float]]:
        r_mix = self.bank.r_mix
        err_est = float(np.linalg.norm(r_mix - self.r_des_world))

        std_x, std_y = self._stds_from_Pmix()
        std_max = max(std_x, std_y)

        unc_metric = float(std_x + std_y)
        prev_unc = float(self.prev_unc_metric) if self.prev_unc_metric is not None else unc_metric
        unc_reduction = float(prev_unc - unc_metric)
        unc_reduction_pos = float(max(0.0, unc_reduction))
        self.prev_unc_metric = unc_metric

        # Patch 1: sens avg gęste
        sens_avg = float(self.sens_accum / max(self.sens_count, 1))

        nis_mix = float(self.bank.last_nis_mix) if np.isfinite(self.bank.last_nis_mix) else float("nan")
        nis_excess = 0.0
        if np.isfinite(nis_mix):
            nis_excess = max(0.0, float(nis_mix) - float(self.cfg.nis_95))

        rho_mix = float(np.linalg.norm(r_mix))
        close_pen = 0.0
        if rho_mix < self.cfg.rho_soft_min:
            close_pen = float(self.cfg.rho_soft_min - rho_mix)

        r_pos = -self.cfg.w_pos * sat_ratio(err_est, self.cfg.pos_err0_m)
        r_info = self.cfg.w_info * sat_ratio(unc_reduction_pos, self.cfg.unc_red0_m)
        r_sens = self.cfg.w_sens * sat_ratio(sens_avg, self.cfg.sens0_m_s)
        r_nis = -self.cfg.w_nis * sat_ratio(nis_excess, self.cfg.nis_excess0)
        r_close = -self.cfg.w_close * sat_ratio(close_pen, self.cfg.close0_m)

        # Patch 3: boost domykania niepewności
        d = self._difficulty_ep
        tol_pos = lerp(self.cfg.tol_pos_est_easy, self.cfg.tol_pos_est_hard, d)
        tol_std = lerp(self.cfg.tol_std_easy, self.cfg.tol_std_hard, d)

        close_to_goal = (err_est < float(self.cfg.close_goal_pos_mult) * float(tol_pos))
        need_info = (std_max > float(tol_std))

        boost = float(self.cfg.info_sens_boost) if (close_to_goal and need_info) else 1.0
        r_info *= boost
        r_sens *= boost

        # Patch 2: stall penalty — tylko gdy nadal niepewny i w kroku był pomiar Dopplera
        doppler_any_update = int(self.doppler_updates_step > 0)

        if need_info and (self.doppler_meas_step > 0):
            if doppler_any_update == 0:
                self.no_update_steps += 1
            else:
                self.no_update_steps = 0
        else:
            self.no_update_steps = 0

        r_stall = 0.0
        if need_info and (self.doppler_meas_step > 0):
            r_stall = -float(self.cfg.w_stall) * sat_ratio(float(self.no_update_steps), float(self.cfg.stall0_steps))

        reward = float(r_pos + r_info + r_sens + r_nis + r_close + r_stall)

        w_max, ess = self.bank._bank_stats()

        terms = {
            "err_est": err_est,
            "std_x": std_x,
            "std_y": std_y,
            "std_max": std_max,
            "unc_metric": unc_metric,
            "unc_reduction_pos": unc_reduction_pos,
            "sens_avg": sens_avg,
            "nis_mix": nis_mix if np.isfinite(nis_mix) else float("nan"),
            "nis_excess": float(nis_excess),
            "rho_mix": rho_mix,
            "w_max": w_max,
            "ess": ess,
            "rejuv_count_ep": float(self.bank.rejuv_count_ep),

            "r_pos": float(r_pos),
            "r_info": float(r_info),
            "r_sens": float(r_sens),
            "r_nis": float(r_nis),
            "r_close": float(r_close),
            "r_stall": float(r_stall),

            "close_to_goal": float(int(close_to_goal)),
            "need_info": float(int(need_info)),
            "boost": float(boost),

            "no_update_steps": float(self.no_update_steps),
            "doppler_meas_step": float(self.doppler_meas_step),
            "doppler_updates_step": float(self.doppler_updates_step),
            "doppler_updated_any": float(doppler_any_update),
        }

        # zachowujemy starą nazwę "doppler_updated" jako "czy był update w tym kroku RL"
        terms["doppler_updated"] = float(doppler_any_update)

        if bank_metrics_last is not None:
            terms["doppler_vperp_last"] = float(bank_metrics_last.sens_vperp)
            terms["doppler_rejuv_last"] = float(bank_metrics_last.rejuv)

        # optional truth diagnostics (still not used in reward)
        if self.cfg.log_truth_diagnostics:
            r_true = self.pL - self.pF
            err_true_form = float(np.linalg.norm(r_true - self.r_des_world))
            terms["err_true_form"] = err_true_form

        return reward, terms

    def _check_done(self) -> Tuple[bool, bool, str]:
        d = self._difficulty_ep
        tol_pos = lerp(self.cfg.tol_pos_est_easy, self.cfg.tol_pos_est_hard, d)
        tol_std = lerp(self.cfg.tol_std_easy, self.cfg.tol_std_hard, d)

        r_mix = self.bank.r_mix
        err_est = float(np.linalg.norm(r_mix - self.r_des_world))

        std_x, std_y = self._stds_from_Pmix()
        std_max = max(std_x, std_y)

        success_now = (err_est < tol_pos) and (std_max < tol_std)

        if success_now:
            self.success_streak += 1
        else:
            self.success_streak = 0

        terminated = self.success_streak >= int(self.cfg.success_hold_steps)
        truncated = self.step_count >= int(self.cfg.max_steps)

        if terminated:
            return True, False, "success"
        if truncated:
            return False, True, "max_steps"
        return False, False, "running"

    def _get_info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        r_mix = self.bank.r_mix.copy()
        P_mix = self.bank.P_mix.copy()
        pF_hat = (self.pL - r_mix).copy()
        pF_des = (self.pL - self.r_des_world).copy()

        info: Dict[str, Any] = {
            "t": float(self.t),
            "step": int(self.step_count),
            "difficulty": float(self._difficulty_ep),
            "leader_speed": float(self.leader_speed),
            "heading_L_deg": float(self.heading_L),
            "speed_F": float(self.speed_F),
            "heading_F_deg": float(self.heading_F),
            "s_meas_last": float(self.s_meas_last),

            "r_des_x": float(self.r_des_world[0]),
            "r_des_y": float(self.r_des_world[1]),
            "r_mix_x": float(r_mix[0]),
            "r_mix_y": float(r_mix[1]),
            "Pmix_xx": float(P_mix[0, 0]),
            "Pmix_xy": float(P_mix[0, 1]),
            "Pmix_yy": float(P_mix[1, 1]),

            "pL_x": float(self.pL[0]),
            "pL_y": float(self.pL[1]),
            "pFhat_x": float(pF_hat[0]),
            "pFhat_y": float(pF_hat[1]),
            "pFdes_x": float(pF_des[0]),
            "pFdes_y": float(pF_des[1]),

            "w_max": float(np.max(self.bank.w)),
            "ess": float(1.0 / np.sum(np.square(self.bank.w))),
            "rejuv_count_ep": float(self.bank.rejuv_count_ep),

            # per-step FIM counts
            "fim_step_meas": float(self.fim_step_meas),
            "fim_step_used": float(self.fim_step_used),
        }

        # ===== Truth metrics (always log in this eval-oriented script) =====
        r_true = self.pL - self.pF
        err_true_form = float(np.linalg.norm(r_true - self.r_des_world))  # formation error on truth
        err_true_rel_est = float(np.linalg.norm(r_mix - r_true))          # estimation error of r

        info.update({
            "pF_x": float(self.pF[0]),
            "pF_y": float(self.pF[1]),
            "r_true_x": float(r_true[0]),
            "r_true_y": float(r_true[1]),
            "err_true_form": float(err_true_form),
            "err_true_rel_est": float(err_true_rel_est),
        })

        # ===== FIM/CRLB metrics =====
        I_tot = self.fim_total.I_total
        I_win = self.fim_win.I_win

        C_tot = self.fim_total.crlb(use_window=False)
        C_win = self.fim_win.crlb(use_window=True)

        eigmin_tot, eigmax_tot, cond_tot = self.fim_total.eig_stats(I_tot)
        eigmin_win, eigmax_win, cond_win = self.fim_win.eig_stats(I_win)

        info.update({
            # FIM
            "fim_tot_xx": float(I_tot[0, 0]),
            "fim_tot_xy": float(I_tot[0, 1]),
            "fim_tot_yy": float(I_tot[1, 1]),
            "fim_tot_eig_min": float(eigmin_tot),
            "fim_tot_eig_max": float(eigmax_tot),
            "fim_tot_cond": float(cond_tot),

            "fim_win_xx": float(I_win[0, 0]),
            "fim_win_xy": float(I_win[0, 1]),
            "fim_win_yy": float(I_win[1, 1]),
            "fim_win_eig_min": float(eigmin_win),
            "fim_win_eig_max": float(eigmax_win),
            "fim_win_cond": float(cond_win),

            # CRLB (regularized inverse)
            "crlb_tot_xx": float(C_tot[0, 0]),
            "crlb_tot_xy": float(C_tot[0, 1]),
            "crlb_tot_yy": float(C_tot[1, 1]),
            "crlb_tot_std_x": float(math.sqrt(max(C_tot[0, 0], 0.0))) if np.isfinite(C_tot[0, 0]) else float("nan"),
            "crlb_tot_std_y": float(math.sqrt(max(C_tot[1, 1], 0.0))) if np.isfinite(C_tot[1, 1]) else float("nan"),
            "crlb_tot_trace": float(np.trace(C_tot)) if np.all(np.isfinite(C_tot)) else float("nan"),

            "crlb_win_xx": float(C_win[0, 0]),
            "crlb_win_xy": float(C_win[0, 1]),
            "crlb_win_yy": float(C_win[1, 1]),
            "crlb_win_std_x": float(math.sqrt(max(C_win[0, 0], 0.0))) if np.isfinite(C_win[0, 0]) else float("nan"),
            "crlb_win_std_y": float(math.sqrt(max(C_win[1, 1], 0.0))) if np.isfinite(C_win[1, 1]) else float("nan"),
            "crlb_win_trace": float(np.trace(C_win)) if np.all(np.isfinite(C_win)) else float("nan"),
        })

        if extra:
            info.update(extra)
        return info

    # =========== render (optional) ===========
    def render(self):
        if self.render_mode != "human":
            return
        if pygame is None:
            raise RuntimeError("pygame nie jest zainstalowany. Zainstaluj: pip install pygame")

        if not self._pygame_inited:
            pygame.init()
            self._screen = pygame.display.set_mode((self.cfg.screen_w, self.cfg.screen_h))
            pygame.display.set_caption("UUV Eval – FIM/CRLB (press M for manual)")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("consolas", 16)
            self._pygame_inited = True

        assert self._screen is not None and self._clock is not None and self._font is not None
        screen = self._screen
        font = self._font

        self._process_pygame_events()
        if self._request_quit:
            return

        screen.fill((10, 10, 25))

        r_mix = self.bank.r_mix
        pF_hat = self.pL - r_mix
        pF_des = self.pL - self.r_des_world

        vL = vel_vec2(self.leader_speed, self.heading_L)
        vF = vel_vec2(self.speed_F, self.heading_F)

        hyp_positions = [self.pL, self.pF, pF_hat, pF_des]
        for f in self.bank.filters:
            hyp_positions.append(self.pL - f.r_hat)
        self._cam.update(hyp_positions)

        L_scr = self._cam.world_to_screen(self.pL)
        F_scr = self._cam.world_to_screen(self.pF)
        Fhat_scr = self._cam.world_to_screen(pF_hat)
        Fdes_scr = self._cam.world_to_screen(pF_des)

        for r_m in (50, 100, 200, 400):
            rad_px = int(r_m * self._cam.scale)
            if 5 < rad_px < max(self.cfg.screen_w, self.cfg.screen_h):
                pygame.draw.circle(screen, (35, 40, 70), L_scr, rad_px, 1)

        pygame.draw.circle(screen, (80, 150, 255), L_scr, 8)
        pygame.draw.circle(screen, (80, 230, 130), F_scr, 7)
        pygame.draw.circle(screen, (255, 90, 90), Fhat_scr, 5, 2)
        pygame.draw.circle(screen, (255, 210, 100), Fdes_scr, 6, 2)

        pygame.draw.line(screen, (150, 150, 200), L_scr, F_scr, 1)
        pygame.draw.line(screen, (255, 210, 100), L_scr, Fhat_scr, 1)

        def draw_arrow(pos_world: np.ndarray, v_world: np.ndarray, color, length_scale: float = 10.0, width: int = 3):
            start = self._cam.world_to_screen(pos_world)
            end_world = pos_world + v_world * float(length_scale)
            end = self._cam.world_to_screen(end_world)
            pygame.draw.line(screen, color, start, end, width)

        draw_arrow(self.pL, vL, (120, 190, 255), length_scale=10.0, width=3)
        draw_arrow(self.pF, vF, (80, 230, 130), length_scale=10.0, width=3)

        for i, f in enumerate(self.bank.filters):
            pFi_hat = self.pL - f.r_hat
            pi = self._cam.world_to_screen(pFi_hat)
            w = float(self.bank.w[i])
            rad = int(2 + 8 * math.sqrt(max(0.0, w)))
            pygame.draw.circle(screen, (180, 110, 110), pi, rad, 1)

        P_mix = self.bank.P_mix
        pts = cov_ellipse_points(P_mix, k=self.cfg.cov_ellipse_k, n_pts=48)
        if pts.shape[0] > 0:
            poly = []
            for p in pts:
                wpos = pF_hat + p
                poly.append(self._cam.world_to_screen(wpos))
            pygame.draw.lines(screen, (255, 120, 120), True, poly, 1)

        std_x = float(math.sqrt(max(P_mix[0, 0], 1e-12)))
        std_y = float(math.sqrt(max(P_mix[1, 1], 1e-12)))

        self._std_hist_x.append(std_x)
        self._std_hist_y.append(std_y)
        if len(self._std_hist_x) > self.cfg.cov_plot_hist_len:
            self._std_hist_x.pop(0)
            self._std_hist_y.pop(0)

        plot_w, plot_h = 360, 130
        margin = 10
        px0 = self.cfg.screen_w - plot_w - margin
        py0 = self.cfg.screen_h - plot_h - margin
        pygame.draw.rect(screen, (15, 15, 40), (px0, py0, plot_w, plot_h))
        pygame.draw.rect(screen, (90, 90, 140), (px0, py0, plot_w, plot_h), 1)

        max_std = max(max(self._std_hist_x + self._std_hist_y), 10.0)
        max_std = min(max_std, 500.0)
        n = len(self._std_hist_x)
        if n > 1:
            def make_pts(hist: List[float]):
                pts2 = []
                for i, v in enumerate(hist):
                    x = px0 + 20 + (plot_w - 40) * (i / (n - 1))
                    frac = max(0.0, min(v / max_std, 1.0))
                    y = py0 + plot_h - 20 - frac * (plot_h - 40)
                    pts2.append((int(x), int(y)))
                return pts2
            pygame.draw.lines(screen, (120, 240, 140), False, make_pts(self._std_hist_x), 2)
            pygame.draw.lines(screen, (240, 140, 140), False, make_pts(self._std_hist_y), 2)

        def draw_text(text: str, x: int, y: int, color=(230, 230, 230)):
            surf = font.render(text, True, color)
            screen.blit(surf, (x, y))

        err_est = float(np.linalg.norm(self.bank.r_mix - self.r_des_world))
        err_true = float(np.linalg.norm((self.pL - self.pF) - self.r_des_world))
        w_max, ess = self.bank._bank_stats()
        nis_mix = self.bank.last_nis_mix
        mode = "MANUAL" if self.manual_override else "POLICY"

        eigmin_win, eigmax_win, _ = self.fim_win.eig_stats(self.fim_win.I_win)
        crlb_win = self.fim_win.crlb(use_window=True)
        crlb_tr = float(np.trace(crlb_win)) if np.all(np.isfinite(crlb_win)) else float("nan")

        y0 = 8
        draw_text(f"t={self.t:6.1f}s step={self.step_count:4d} d={self._difficulty_ep:.2f} mode={mode} [M]", 10, y0); y0 += 18
        draw_text(f"Leader: v={self.leader_speed:4.2f} m/s hdg={self.heading_L:6.1f} deg", 10, y0); y0 += 18
        draw_text(f"Follower: v={self.speed_F:4.2f} m/s hdg={self.heading_F:6.1f} deg  (arrows, SHIFT fine)", 10, y0); y0 += 18
        draw_text(f"err_est={err_est:7.2f} m  err_true={err_true:7.2f} m", 10, y0); y0 += 18
        draw_text(f"std_x={std_x:6.1f} std_y={std_y:6.1f}  NIS_mix={nis_mix:6.2f}", 10, y0); y0 += 18
        draw_text(f"FIM(win) eig_min={eigmin_win:8.3e} eig_max={eigmax_win:8.3e}  CRLB_trace(win)={crlb_tr:8.2f}", 10, y0); y0 += 18
        draw_text(f"w_max={w_max:5.3f} ESS={ess:5.2f} rejuv_count_ep={self.bank.rejuv_count_ep}", 10, y0); y0 += 18
        draw_text("BACKSPACE reset, ESC/Q quit", 10, y0, (200, 200, 140)); y0 += 18

        draw_text(f"std_x/std_y history (green/red), max~{max_std:5.1f} m", px0 + 10, py0 + 5, (220, 220, 220))

        pygame.display.flip()
        self._clock.tick(self.cfg.render_fps)

    def close(self):
        if self._pygame_inited and pygame is not None:
            pygame.quit()
        self._pygame_inited = False
        self._screen = None
        self._clock = None
        self._font = None

    def _process_pygame_events(self) -> None:
        if pygame is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._request_quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self._request_quit = True
                elif event.key == pygame.K_BACKSPACE:
                    self._request_reset = True
                elif event.key == pygame.K_m:
                    self.manual_override = not self.manual_override
        pygame.event.pump()

    def _keyboard_action(self) -> np.ndarray:
        if pygame is None:
            return np.zeros(2, dtype=np.float32)
        keys = pygame.key.get_pressed()
        a_speed = 0.0
        if keys[pygame.K_UP]:
            a_speed += 1.0
        if keys[pygame.K_DOWN]:
            a_speed -= 1.0

        a_turn = 0.0
        if keys[pygame.K_LEFT]:
            a_turn += 1.0
        if keys[pygame.K_RIGHT]:
            a_turn -= 1.0

        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            a_speed *= float(self.cfg.manual_fine_scale)
            a_turn *= float(self.cfg.manual_fine_scale)

        return np.array([clamp(a_speed, -1.0, 1.0), clamp(a_turn, -1.0, 1.0)], dtype=np.float32)


# ===========================
# SB3 utils / Evaluation
# ===========================

def make_env(seed: int, cfg: UUVConfig, render: bool):
    def _init():
        env = UUVRelPosBankEnv(cfg=cfg, render_mode=("human" if render else "none"))
        env.reset(seed=seed)
        return env
    return _init


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def _flatten_obs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x.reshape(-1)

def _vec_unnormalize_obs(vecnorm, obs_norm: np.ndarray) -> np.ndarray:
    try:
        return vecnorm.unnormalize_obs(obs_norm)
    except Exception:
        return obs_norm


def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def save_npz(path: str, data: Dict[str, Any]) -> None:
    arrays = {}
    for k, v in data.items():
        try:
            arrays[k] = np.asarray(v)
        except Exception:
            pass
    np.savez_compressed(path, **arrays)


def cmd_eval(args: argparse.Namespace) -> None:
    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except Exception as e:
        raise RuntimeError("Zainstaluj stable-baselines3[extra].") from e

    # cfg: force truth diag in eval (does NOT affect obs/reward)
    cfg = UUVConfig(
        action_dt=float(args.action_dt),
        fim_window_s=float(args.fim_window),
        fim_reg_eps=float(args.fim_reg_eps),
        fim_use_gating=bool(args.fim_use_gating),
        log_truth_diagnostics=True,  # always on for eval logs
    )

    base_env = DummyVecEnv([make_env(int(args.seed), cfg, render=bool(args.render))])

    # VecNormalize (optional)
    vn_path = os.path.join(args.models_dir, "vecnormalize.pkl")
    vecnorm = None
    env = base_env
    if os.path.exists(vn_path):
        env = VecNormalize.load(vn_path, env)
        env.training = False
        env.norm_reward = False
        vecnorm = env

    # load model (only if controller == 'rl')
    controller = str(getattr(args, "controller", "rl")).lower().strip()

    algo = args.algo.lower()
    device = str(args.device)
    if device == "auto":
        device = "cpu"

    model = None
    if controller == "rl":
        if not str(args.model).strip():
            raise ValueError("For --controller rl you must provide --model path/to/model.zip")
        if algo == "ppo":
            model = PPO.load(args.model, env=env, device=device)
        elif algo == "sac":
            model = SAC.load(args.model, env=env, device=device)
        else:
            raise ValueError("algo must be ppo or sac")


    # output dirs
    out_dir = str(args.out_dir)
    _ensure_dir(out_dir)
    ts0 = _timestamp()

    meta = {
        "timestamp": ts0,
        "algo": algo,
        "controller": controller,
        "model": str(args.model),
        "baseline_kp": float(getattr(args, "baseline_kp", 0.0)),
        "baseline_kexc": float(getattr(args, "baseline_kexc", 0.0)),
        "baseline_std_trigger": float(getattr(args, "baseline_std_trigger", 0.0)),
        "vecnormalize": bool(os.path.exists(vn_path)),
        "cfg": asdict(cfg),
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "render": bool(args.render),
        "fim_window": float(args.fim_window),
        "fim_reg_eps": float(args.fim_reg_eps),
        "fim_use_gating": bool(args.fim_use_gating),
    }
    meta_path = os.path.join(out_dir, f"eval_meta_{ts0}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("[EVAL] meta:", meta_path)

    summary_rows: List[Dict[str, Any]] = []

    # helper: render inside VecEnv
    def _render_vecenv(venv):
        try:
            if hasattr(venv, "envs") and len(getattr(venv, "envs")) > 0:
                venv.envs[0].render()
                return
        except Exception:
            pass
        try:
            if hasattr(venv, "venv") and hasattr(venv.venv, "envs") and len(getattr(venv.venv, "envs")) > 0:
                venv.venv.envs[0].render()
                return
        except Exception:
            pass

    # run episodes
    for ep in range(int(args.episodes)):
        obs = env.reset()
        done = False
        ep_return = 0.0

        rows: List[Dict[str, Any]] = []
        npz: Dict[str, Any] = {
            "t": [], "reward": [], "done": [],
            "action": [], "obs_norm": [], "obs_raw": [],
        }

        step_idx = 0
        term_reason = ""
        t_end = float("nan")
        err_true_end = float("nan")
        err_est_end = float("nan")
        fim_eig_min_end = float("nan")
        crlb_trace_end = float("nan")

        while not done:
            if controller == "rl":
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Baseline/random action uses *raw* observation (unnormalized)
                obs_norm_tmp = np.array(obs).copy()
                obs_raw_tmp = obs_norm_tmp
                if vecnorm is not None:
                    obs_raw_tmp = _vec_unnormalize_obs(vecnorm, obs_norm_tmp)
                obs_raw_flat_tmp = _flatten_obs(obs_raw_tmp)
                act = baseline_action_from_obs_raw(
                    obs_raw_flat_tmp,
                    cfg,
                    controller=controller,
                    kp=float(getattr(args, "baseline_kp", 0.02)),
                    kexc=float(getattr(args, "baseline_kexc", 0.8)),
                    std_trigger_m=float(getattr(args, "baseline_std_trigger", 0.0)),
                )
                action = act.reshape(1, 2)

            obs2, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            ep_return += float(reward[0])

            if args.render:
                _render_vecenv(env)

            info0 = infos[0] if infos else {}
            term_reason = str(info0.get("term_reason", "")) if done else term_reason
            t_end = float(info0.get("t", np.nan))
            err_true_end = float(info0.get("err_true_form", np.nan))
            err_est_end = float(info0.get("err_est", np.nan))
            fim_eig_min_end = float(info0.get("fim_win_eig_min", np.nan))
            crlb_trace_end = float(info0.get("crlb_win_trace", np.nan))

            # log step
            obs_norm = np.array(obs).copy()
            obs_raw = obs_norm
            if vecnorm is not None:
                obs_raw = _vec_unnormalize_obs(vecnorm, obs_norm)

            obs_norm_flat = _flatten_obs(obs_norm)
            obs_raw_flat = _flatten_obs(obs_raw)
            act_flat = _flatten_obs(action)

            row: Dict[str, Any] = {
                "episode": ep + 1,
                "step": step_idx,
                "t": float(info0.get("t", np.nan)),
                "reward": float(reward[0]),
                "done": int(done),
                "term_reason": str(info0.get("term_reason", "")),
                "a_speed": float(act_flat[0]) if act_flat.size > 0 else 0.0,
                "a_turn": float(act_flat[1]) if act_flat.size > 1 else 0.0,
                "controller": controller,
            }

            # keep all scalar floats from info (best for research)
            for k, v in info0.items():
                if k in row:
                    continue
                try:
                    fv = float(v)
                    if np.isfinite(fv):
                        row[k] = fv
                    else:
                        row[k] = fv
                except Exception:
                    # keep strings like term_reason too
                    row[k] = v

            # add obs raw/norm
            for i, v in enumerate(obs_raw_flat):
                row[f"obs_raw_{i}"] = float(v)
            for i, v in enumerate(obs_norm_flat):
                row[f"obs_norm_{i}"] = float(v)

            rows.append(row)

            npz["t"].append(float(info0.get("t", np.nan)))
            npz["reward"].append(float(reward[0]))
            npz["done"].append(int(done))
            npz["action"].append(act_flat.astype(np.float32))
            npz["obs_norm"].append(obs_norm_flat.astype(np.float32))
            npz["obs_raw"].append(obs_raw_flat.astype(np.float32))

            obs = obs2
            step_idx += 1

        # save episode trace
        ts = _timestamp()
        base = os.path.join(out_dir, f"eval_trace_{ts}_seed{int(args.seed)}_ep{ep+1:03d}")
        csv_path = base + ".csv"
        npz_path = base + ".npz"
        save_csv(csv_path, rows)
        save_npz(npz_path, npz)

        print(f"[EVAL] ({controller}) ep {ep+1}/{args.episodes} return={ep_return:.2f} term={term_reason} "
              f"t_end={t_end:.1f}s err_true_end={err_true_end:.2f} err_est_end={err_est_end:.2f} "
              f"eigmin(FIM_win)={fim_eig_min_end:.3e} crlb_trace_win={crlb_trace_end:.2f}")
        print("       csv:", csv_path)
        print("       npz:", npz_path)

        summary_rows.append({
            "episode": ep + 1,
            "seed": int(args.seed),
            "return": float(ep_return),
            "term_reason": term_reason,
            "t_end": float(t_end),
            "err_true_end": float(err_true_end),
            "err_est_end": float(err_est_end),
            "fim_win_eig_min_end": float(fim_eig_min_end),
            "crlb_win_trace_end": float(crlb_trace_end),
            "trace_csv": os.path.basename(csv_path),
            "trace_npz": os.path.basename(npz_path),
            "controller": controller,
        })

    # save summary
    summary_path = os.path.join(out_dir, f"eval_summary_{ts0}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "episodes": summary_rows}, f, indent=2)
    print("[EVAL] summary:", summary_path)

    env.close()


# ===========================
# CLI
# ===========================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="Evaluate an existing SB3 model and log FIM/CRLB + truth errors.")
    pe.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    pe.add_argument("--controller", choices=["rl", "pid", "pid_exc", "random"], default="rl",
                    help="Controller to evaluate: rl=SB3 policy, pid=formation P, pid_exc=formation+excitation, random=random actions.")
    pe.add_argument("--baseline-kp", type=float, default=0.02, help="(pid/pid_exc) Proportional gain [1/s] in vF_cmd = vL + kp*(r_hat-r_des).")
    pe.add_argument("--baseline-kexc", type=float, default=0.8, help="(pid_exc) Max lateral excitation speed [m/s].")
    pe.add_argument("--baseline-std-trigger", type=float, default=0.0, help="(pid_exc) Trigger excitation when std_max > this [m]. 0 => use cfg.tol_std_hard.")
    pe.add_argument("--model", type=str, default="", help="Path to SB3 model .zip (required if --controller rl).")
    pe.add_argument("--models-dir", type=str, default="models", help="Directory with vecnormalize.pkl (if used).")
    pe.add_argument("--episodes", type=int, default=10)
    pe.add_argument("--seed", type=int, default=42)
    pe.add_argument("--action-dt", type=float, default=2.0)
    pe.add_argument("--render", action="store_true")
    pe.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")

    pe.add_argument("--out-dir", type=str, default="eval_fim_logs")

    pe.add_argument("--fim-window", type=float, default=30.0, help="Sliding window for FIM/CRLB [s].")
    pe.add_argument("--fim-reg-eps", type=float, default=1e-9, help="Regularization epsilon for CRLB.")
    pe.add_argument("--fim-use-gating", action="store_true", help="Count only Doppler samples that pass gating (default).")
    pe.add_argument("--fim-no-gating", dest="fim_use_gating", action="store_false", help="Count all Doppler samples.")
    pe.set_defaults(fim_use_gating=True)

    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "eval":
        cmd_eval(args)
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
