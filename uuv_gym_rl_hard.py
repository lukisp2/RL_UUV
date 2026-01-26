
# -*- coding: utf-8 -*-
"""
UUV Active Relative Localization + Formation (HARD) – Gymnasium + SB3 PPO/SAC + TensorBoard
=========================================================================================

Wersja "hard":
  - nagroda i warunek sukcesu NIE używają prawdy (r_true).
  - agent dostaje tylko to, co "na pokładzie": EKF (r_hat, P), własny speed/heading,
    oraz znany ruch Leadera (speed+heading, losowane per epizod).

Pomiary:
  - Doppler / range-rate:  s = - r̂ · v_rel
  - sampling co S_MEAS_PERIOD (domyślnie 1 s)
  - Doppler-only (brak range)

Domain randomization:
  - leader_speed i leader_heading losowane na początku epizodu (stałe w epizodzie).

Zakończenie:
  - sukces: (||r_hat - r_des|| < tol_pos_est) AND (std_max < tol_std) przez N kroków
  - albo limit kroków

Metryki konsystencji (wyjaśnienie):
  - innovation (residual) w EKF: y = z - h(x^-)
  - innovation covariance:      S = H P^- H^T + R

  NIS (Normalized Innovation Squared):
      NIS = y^T S^{-1} y
    * nie wymaga prawdy (good for real-world)
    * dla m=1 (1D pomiar) ma rozkład chi^2(dof=1) przy konsystentnym filtrze
      - 95% próg ~ 3.84
      - 99% próg ~ 6.63
    * jeśli NIS często jest duży -> filtr/model jest niespójny (za małe P/R, model mismatch, itp.)

  NEES (Normalized Estimation Error Squared):
      NEES = (x_hat - x_true)^T P^{-1} (x_hat - x_true)
    * wymaga prawdy -> w realu niepoliczalne
    * w symulatorze świetne do walidacji (diagnostyka)

W tej wersji:
  - w reward dajemy karę tylko za NIS powyżej progu 95% (żeby nie "minimalizować NIS do zera").
  - NEES liczymy tylko diagnostycznie (opcjonalnie) i nie wpływa na reward/sukces.

NOWE (na Twoją prośbę):
  1) Tryb symulacyjny (sterowanie z klawiszy) w renderze:
     - w trybie render (human) naciśnij [M], aby przełączyć manual override ON/OFF
     - w manual:
         ↑/↓  przyspieszaj/hamuj (akceleracja)
         ←/→  skręcaj (turn rate)
         BACKSPACE reset epizodu
         ESC/Q wyjście
     - działa zarówno w komendzie "sim", jak i w "play" (możesz przejąć sterowanie nad RL).

  2) Render: wizualizacja wektorów prędkości (strzałki) dla Leadera i Followera
     (jak w Twoim pierwotnym kodzie – tylko zamiast samego headingu, rysujemy v).

Wymagania:
  pip install numpy gymnasium pygame tensorboard stable-baselines3[extra]

Uruchomienia:

  # trening PPO
  python uuv_gym_rl_hard.py train --algo ppo --tb runs/uuv_hard --steps 800000 --n-envs 8 --action-dt 2.0

  # trening SAC
  python uuv_gym_rl_hard.py train --algo sac --tb runs/uuv_hard --steps 800000 --n-envs 8 --action-dt 2.0

  # TensorBoard
  tensorboard --logdir runs/uuv_hard

  # symulator ręczny (klawisze)
  python uuv_gym_rl_hard.py sim --render --action-dt 2.0 --log-truth-diag

  # test render: losowa polityka (bez manual override domyślnie)
  python uuv_gym_rl_hard.py random --episodes 1 --render --action-dt 2.0

  # render z wytrenowanym modelem + możliwość przejęcia sterowania klawiszem [M]
  python uuv_gym_rl_hard.py play --algo ppo --model models/ppo_uuv_hard.zip --episodes 3 --render --action-dt 2.0
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np

import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
except Exception:
    pygame = None


# ===================== Pomocnicze ===========================================

def deg2rad(a: float) -> float:
    return a * math.pi / 180.0

def wrap360(a: float) -> float:
    a = a % 360.0
    if a < 0:
        a += 360.0
    return a

def vel_vec2(speed: float, heading_deg: float) -> np.ndarray:
    """
    2D konwencja:
      0° = +x (E), 90° = +y (N)
    """
    h = deg2rad(heading_deg)
    return np.array([speed * math.cos(h), speed * math.sin(h)], dtype=float)

def radial_speed(r_vec: np.ndarray, v_rel: np.ndarray) -> float:
    """
    s = - r̂ · v_rel
    """
    rho = float(np.linalg.norm(r_vec))
    if rho <= 1e-9:
        return 0.0
    rhat = r_vec / rho
    return -float(np.dot(rhat, v_rel))

def rot2(heading_deg: float) -> np.ndarray:
    """Macierz obrotu 2D dla heading w stopniach (zgodna z vel_vec2)."""
    c = math.cos(deg2rad(heading_deg))
    s = math.sin(deg2rad(heading_deg))
    return np.array([[c, -s],
                     [s,  c]], dtype=float)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ===================== EKF ===================================================

class EKFRelPos2D:
    """
    Stan: r = pL - pF (2D)
    Predykcja: r_{k+1} = r_k + dt * v_rel
    Pomiar Dopplera: s = - rhat · v_rel
    """
    def __init__(
        self,
        r0: np.ndarray,
        pos_std0: float = 300.0,
        q_std: float = 0.025,
        sigma_s: float = 0.015,
    ) -> None:
        self.r_hat = np.array(r0, dtype=float)
        self.P = np.diag([pos_std0**2, pos_std0**2]).astype(float)

        # Model procesu: dyfuzja pozycji (na krok ~ q^2 * dt)
        self.q_var = float(q_std**2)

        # Szum pomiaru Dopplera (1D)
        self.Rs = np.array([[sigma_s**2]], dtype=float)

        # Diagnostyka innowacji
        self.last_innov: float = float("nan")     # y
        self.last_S: float = float("nan")         # S (skalar)
        self.last_nis: float = float("nan")       # y^2 / S
        self.last_h: float = float("nan")         # h(x^-)
        self.last_z: float = float("nan")         # z

    def predict(self, v_rel: np.ndarray, dt: float) -> None:
        v_rel = np.array(v_rel, dtype=float)
        dt = float(dt)

        self.r_hat = self.r_hat + dt * v_rel

        Q = self.q_var * max(dt, 1e-3) * np.eye(2, dtype=float)
        self.P = self.P + Q

    def update_s(self, v_rel: np.ndarray, s_meas: float, rho_min: float = 20.0, v_min: float = 0.05) -> None:
        v_rel = np.array(v_rel, dtype=float)
        vnorm = float(np.linalg.norm(v_rel))
        rho = float(np.linalg.norm(self.r_hat))

        # gating dla stabilności i sensownej obserwowalności
        if rho <= rho_min or vnorm <= v_min:
            self.last_innov = float("nan")
            self.last_S = float("nan")
            self.last_nis = float("nan")
            self.last_h = float("nan")
            self.last_z = float("nan")
            return

        rhat = self.r_hat / rho
        proj = float(np.dot(rhat, v_rel))
        h = -proj

        # Jacobian ∂h/∂r
        H = -(v_rel - proj * rhat) / rho
        H = H.reshape(1, 2)

        z = float(s_meas)
        y = z - h

        S = float((H @ self.P @ H.T + self.Rs)[0, 0])
        if S <= 1e-12:
            return

        nis = (y * y) / S

        # Diagnostyka
        self.last_innov = float(y)
        self.last_S = float(S)
        self.last_nis = float(nis)
        self.last_h = float(h)
        self.last_z = float(z)

        # Update
        Sinv = 1.0 / S
        K = (self.P @ H.T) * Sinv  # (2,1)

        self.r_hat = self.r_hat + (K.flatten() * y)

        # Joseph form
        I = np.eye(2)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.Rs @ K.T
        self.P = 0.5 * (self.P + self.P.T)


# ===================== Parametry środowiska ==================================

@dataclass
class UUVConfig:
    # render
    screen_w: int = 1100
    screen_h: int = 800

    # Leader – losowane per epizod
    leader_speed_min: float = 1.0
    leader_speed_max: float = 3.0
    leader_heading_min_deg: float = 0.0
    leader_heading_max_deg: float = 360.0

    # Follower ograniczenia
    f_min_speed: float = 0.2
    f_max_speed: float = 4.0
    max_turn_rate_deg_s: float = 80.0
    max_accel_m_s2: float = 1.0


    # Manual control (render/sim) – żeby "kliknięcie" nie dawało ogromnego skrętu.
    # Te parametry ograniczają ZMIANĘ na krok env.step (czyli na action_dt sekund symulacji),
    # niezależnie od max_turn_rate_deg_s.
    rl_turn_per_step_deg: float = 10.0              # [deg] maks. zmiana kursu na 1 krok dla RL (|Δhdg|<=10°/krok)
    rl_speed_delta_per_step: float = 0.2           # [m/s] maks. zmiana prędkości na 1 krok dla RL (|Δv|<=0.2 m/s/krok)
    manual_turn_per_step_deg: float = 5.0          # [deg] maks. zmiana kursu na 1 krok (przy wciśniętym ←/→)
    manual_speed_delta_per_step: float = 0.05      # [m/s] maks. zmiana prędkości na 1 krok (↑/↓)
    manual_fine_scale: float = 0.25                # SHIFT: mnożnik (np. 0.25 -> 4x wolniej)
    # Pomiary
    sigma_s: float = 0.015
    s_meas_period: float = 1.0

    # błędy nawigacji Followera (w EKF)
    sigma_heading_deg: float = 0.5
    sigma_speed: float = 0.01

    # EKF
    ekf_q_std: float = 0.025
    ekf_init_pos_std: float = 300.0
    ekf_init_rho_guess: float = 150.0  # żeby nie startować z [0,0]

    # czas
    action_dt: float = 2.0
    sub_dt: float = 0.1

    # start (losowy r_true0)
    start_rho_min: float = 80.0
    start_rho_max: float = 220.0

    # cel formacji w układzie Leadera (body)
    # r = pL - pF: jeśli chcesz, by Follower był "za" liderem o dystans D, to r_des_body = [D, 0]
    r_des_body: Tuple[float, float] = (100.0, 0.0)

    # sukces (TYLKO na estymacie i P)
    tol_pos_est: float = 5.0      # [m]
    tol_std: float = 6.0          # [m] max(std_x,std_y)
    success_hold_steps: int = 3
    max_steps: int = 1000          # kroków RL (każdy to action_dt)

    # skale normalizacji obserwacji
    pos_scale: float = 200.0
    std_scale: float = 200.0
    vel_scale: float = 4.0
    s_scale: float = 3.0

    # NIS – progi chi^2 dla dof=1
    nis_95: float = 3.84
    nis_99: float = 6.63

    # nagroda – wagi
    w_pos: float = 0.02        # kara za błąd estymaty [m]
    w_unc: float = 0.0        # kara za niepewność (std_x+std_y)
    w_info: float = 0.20       # bonus za redukcję niepewności
    w_ctrl: float = 0.0       # kara za sterowanie
    w_nis: float = 0.05        # kara za NIS powyżej progu
    w_sens: float = 0.03       # nagroda za "informacyjność" geometrii Dopplera (na bazie |v_rel_perp|) per update
    terminal_bonus: float = 20.0

    # bezpieczeństwo (soft)
    rho_soft_min: float = 15.0
    w_close: float = 1.0

    # diagnostyka (prawda tylko do logów/wyświetlania; NIE do reward/sukcesu)
    log_truth_diagnostics: bool = False


# ===================== Kamera do renderu ======================================

class Camera:
    def __init__(self, screen_w: int, screen_h: int, scale_init: float = 2.0, scale_min: float = 0.4, scale_max: float = 8.0):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.center = np.array([0.0, 0.0], dtype=float)
        self.scale = float(scale_init)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

    def update(self, points: list[np.ndarray]) -> None:
        pts = np.vstack(points)
        center = pts.mean(axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        d_max = float(max(dists.max(), 10.0))
        target_px = 0.40 * min(self.screen_w, self.screen_h)
        target_scale = target_px / d_max
        target_scale = max(self.scale_min, min(self.scale_max, target_scale))

        alpha = 0.15
        self.center = (1 - alpha) * self.center + alpha * center
        self.scale = (1 - alpha) * self.scale + alpha * target_scale

    def world_to_screen(self, pos: np.ndarray) -> Tuple[int, int]:
        rel = np.array(pos, dtype=float) - self.center
        sx = self.screen_w / 2 + rel[0] * self.scale
        sy = self.screen_h / 2 - rel[1] * self.scale
        return int(sx), int(sy)


# ===================== Gymnasium Environment =================================

class UUVRelPosHardEnv(gym.Env):
    """
    Obserwacje (14D):
      0-1: e_hat = r_hat - r_des_world / pos_scale
      2-3: r_hat / pos_scale
      4-5: std_x, std_y / std_scale
      6:   Pxy / std_scale^2
      7:   speed_F / vel_scale
      8-9: cos(heading_F), sin(heading_F)
      10:  s_meas_last / s_scale
      11:  leader_speed / leader_speed_max
      12-13: cos(heading_L), sin(heading_L)

    Akcje:
      a_speed, a_turn w [-1,1] -> acc_cmd, turn_cmd
    """
    metadata = {"render_modes": ["human", "none"], "render_fps": 30}

    def __init__(self, config: Optional[UUVConfig] = None, render_mode: str = "none") -> None:
        super().__init__()
        self.cfg = config or UUVConfig()
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

        self.np_random = np.random.default_rng(0)

        # stan
        self.t = 0.0
        self.step_count = 0
        self.success_streak = 0

        self.pL = np.zeros(2, dtype=float)
        self.pF = np.zeros(2, dtype=float)

        self.leader_speed = 2.0
        self.heading_L = 0.0

        self.heading_F = 0.0
        self.speed_F = 2.0

        # cel w świecie
        self.r_des_world = np.array(self.cfg.r_des_body, dtype=float)

        self.ekf: Optional[EKFRelPos2D] = None

        # Doppler
        self.next_s_time = self.cfg.s_meas_period
        self.s_meas_last = 0.0

        # info_gain
        self.prev_unc_metric: Optional[float] = None


        # Doppler geometry "informativeness" (accumulated over one env.step)
        self.sens_accum: float = 0.0
        self.sens_count: int = 0
        # manual control (render)
        self.manual_override: bool = False
        self._request_reset: bool = False
        self._request_quit: bool = False

        # render
        self._pygame_inited = False
        self._screen = None
        self._clock = None
        self._font = None
        self._cam = Camera(self.cfg.screen_w, self.cfg.screen_h)

    # --------- reset / step ---------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.t = 0.0
        self.step_count = 0
        self.success_streak = 0

        self._request_reset = False
        self._request_quit = False

        # leader randomization
        self.leader_speed = float(self.np_random.uniform(self.cfg.leader_speed_min, self.cfg.leader_speed_max))
        self.heading_L = float(self.np_random.uniform(self.cfg.leader_heading_min_deg, self.cfg.leader_heading_max_deg))

        # r_des: obrót z body do świata
        self.r_des_world = rot2(self.heading_L) @ np.array(self.cfg.r_des_body, dtype=float)

        # Leader na (0,0)
        self.pL[:] = np.array([0.0, 0.0], dtype=float)

        # losowy start r_true0 = pL - pF
        rho0 = float(self.np_random.uniform(self.cfg.start_rho_min, self.cfg.start_rho_max))
        ang0 = float(self.np_random.uniform(0.0, 2.0 * math.pi))
        r_true0 = rho0 * np.array([math.cos(ang0), math.sin(ang0)], dtype=float)
        self.pF = self.pL - r_true0

        # losowy heading Followera
        self.heading_F = float(self.np_random.uniform(0.0, 360.0))
        self.speed_F = 2.0

        # EKF start: losowy guess na stałym rho_guess
        rho_guess = float(self.cfg.ekf_init_rho_guess)
        ang_guess = float(self.np_random.uniform(0.0, 2.0 * math.pi))
        r0_guess = rho_guess * np.array([math.cos(ang_guess), math.sin(ang_guess)], dtype=float)

        self.ekf = EKFRelPos2D(
            r0=r0_guess,
            pos_std0=self.cfg.ekf_init_pos_std,
            q_std=self.cfg.ekf_q_std,
            sigma_s=self.cfg.sigma_s,
        )

        self.next_s_time = self.cfg.s_meas_period
        self.s_meas_last = 0.0

        std_x, std_y, _ = self._get_std_and_pxy()
        self.prev_unc_metric = float(std_x + std_y)

        obs = self._get_obs()
        info = self._get_info(extra={"r0_guess": r0_guess.copy(), "r_true0_diag": r_true0.copy()})
        return obs, info

    def step(self, action: np.ndarray):
        # obsługa eventów przy renderze (reset/quit/toggle manual)
        if self.render_mode == "human" and self._pygame_inited and pygame is not None:
            self._process_pygame_events()

        # manual reset/quit sygnalizowane klawiszami
        if self._request_quit:
            self._request_quit = False
            obs = self._get_obs()
            info = self._get_info(extra={"term_reason": "quit"})
            return obs, 0.0, False, True, info

        if self._request_reset:
            # sygnalizujemy truncation; pętla (sim/play) może wtedy zrobić reset()
            self._request_reset = False
            obs = self._get_obs()
            info = self._get_info(extra={"term_reason": "manual_reset"})
            return obs, 0.0, False, True, info

        # manual override akcji
        if self.render_mode == "human" and self.manual_override and self._pygame_inited and pygame is not None:
            # UWAGA: przy dużym action_dt (np. 2–5 s) pełne max_turn_rate dawałoby ogromny skręt na 1 krok.
            # Dlatego w manualu ograniczamy zmianę na krok: manual_turn_per_step_deg / manual_speed_delta_per_step.
            action = self._keyboard_action()
            acc_cmd = float(action[0]) * (float(self.cfg.manual_speed_delta_per_step) / max(self.cfg.action_dt, 1e-6))
            turn_cmd = float(action[1]) * (float(self.cfg.manual_turn_per_step_deg) / max(self.cfg.action_dt, 1e-6))
        else:
            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)
            acc_cmd = float(action[0]) * self.cfg.max_accel_m_s2

            # RL: limit zmiany kursu na krok (|Δhdg|action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)

            # RL: limit zmiany prędkości i kursu na 1 krok env.step (delta-per-step).
            # Speed: |Δv| <= rl_speed_delta_per_step
            acc_from_step = float(self.cfg.rl_speed_delta_per_step) / max(self.cfg.action_dt, 1e-6)
            acc_limit = min(float(self.cfg.max_accel_m_s2), acc_from_step)
            acc_cmd = float(action[0]) * acc_limit

            # Heading: |Δhdg| <= rl_turn_per_step_deg
            turn_rate_from_step = float(self.cfg.rl_turn_per_step_deg) / max(self.cfg.action_dt, 1e-6)
            turn_rate_limit = min(float(self.cfg.max_turn_rate_deg_s), turn_rate_from_step)
            turn_cmd = float(action[1]) * turn_rate_limit

        n_sub = max(1, int(round(self.cfg.action_dt / self.cfg.sub_dt)))
        dt = self.cfg.action_dt / n_sub

        # reset per-step accumulators for geometry reward
        self.sens_accum = 0.0
        self.sens_count = 0

        for _ in range(n_sub):
            self._sim_substep(acc_cmd=acc_cmd, turn_cmd=turn_cmd, dt=dt)

        self.step_count += 1

        obs = self._get_obs()

        reward, terms = self._compute_reward(action)
        terminated, truncated, reason = self._check_done()

        if terminated and reason == "success":
            reward += float(self.cfg.terminal_bonus)
            terms["terminal_bonus"] = float(self.cfg.terminal_bonus)
        else:
            terms["terminal_bonus"] = 0.0

        info = self._get_info(extra=terms | {"term_reason": reason, "manual_override": int(self.manual_override)})
        return obs, float(reward), bool(terminated), bool(truncated), info

    # --------- dynamika + pomiary ---------

    def _sim_substep(self, acc_cmd: float, turn_cmd: float, dt: float) -> None:
        # sterowanie followera
        self.speed_F = float(np.clip(self.speed_F + acc_cmd * dt, self.cfg.f_min_speed, self.cfg.f_max_speed))
        self.heading_F = wrap360(self.heading_F + turn_cmd * dt)

        vL_true = vel_vec2(self.leader_speed, self.heading_L)
        vF_true = vel_vec2(self.speed_F, self.heading_F)

        self.pL = self.pL + vL_true * dt
        self.pF = self.pF + vF_true * dt

        self.t += dt

        # v_rel do EKF: z błędami Followera
        heading_meas = self.heading_F + float(self.np_random.normal(0.0, self.cfg.sigma_heading_deg))
        speed_meas = self.speed_F + float(self.np_random.normal(0.0, self.cfg.sigma_speed))
        vF_ekf = vel_vec2(speed_meas, heading_meas)

        v_rel_ekf = vL_true - vF_ekf

        assert self.ekf is not None
        self.ekf.predict(v_rel=v_rel_ekf, dt=dt)

        # pomiar Dopplera
        while self.t + 1e-12 >= self.next_s_time:
            r_true = self.pL - self.pF
            v_rel_true = vL_true - vF_true
            s_true = radial_speed(r_true, v_rel_true)

            s_meas = s_true + float(self.np_random.normal(0.0, self.cfg.sigma_s))
            self.s_meas_last = float(s_meas)

            # Geometry reward helper (no truth): how "side-looking" is v_rel vs r_hat?
            # Use current EKF r_hat (after predict) and v_rel_ekf used by the filter.
            try:
                rho_hat = float(np.linalg.norm(self.ekf.r_hat))
                vnorm = float(np.linalg.norm(v_rel_ekf))
                if rho_hat > 20.0 and vnorm > 0.05:
                    rhat = self.ekf.r_hat / rho_hat
                    proj = float(np.dot(rhat, v_rel_ekf))
                    sin_theta = math.sqrt(max(0.0, 1.0 - (proj / vnorm) ** 2))
                    v_perp = vnorm * sin_theta  # |v_rel_perp| [m/s]
                    self.sens_accum += float(v_perp)
                    self.sens_count += 1
            except Exception:
                pass

            self.ekf.update_s(v_rel=v_rel_ekf, s_meas=s_meas)

            self.next_s_time += self.cfg.s_meas_period

    # --------- keyboard control helpers ---------

    def _process_pygame_events(self) -> None:
        """
        Obsługa zdarzeń pygame.
        Nie zatrzymuje programu – tylko ustawia flagi.
        """
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

        # odśwież stan klawiszy
        pygame.event.pump()

    def _keyboard_action(self) -> np.ndarray:
        """
        Mapowanie klawiszy na akcje [-1,1]^2:
          ↑ / ↓ : a_speed = +1 / -1
          ← / → : a_turn  = +1 / -1   (← skręt w lewo)

        Dodatkowo:
          SHIFT = "fine" (wolniej): mnożnik cfg.manual_fine_scale
        """
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

        # fine control: SHIFT zmniejsza amplitudę akcji
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            a_speed *= float(self.cfg.manual_fine_scale)
            a_turn  *= float(self.cfg.manual_fine_scale)

        return np.array([clamp(a_speed, -1.0, 1.0), clamp(a_turn, -1.0, 1.0)], dtype=np.float32)

    # --------- obserwacje / reward / done ---------

    def _get_std_and_pxy(self) -> Tuple[float, float, float]:
        assert self.ekf is not None
        P = self.ekf.P
        std_x = float(math.sqrt(max(P[0, 0], 1e-9)))
        std_y = float(math.sqrt(max(P[1, 1], 1e-9)))
        pxy = float(P[0, 1])
        return std_x, std_y, pxy

    def _get_obs(self) -> np.ndarray:
        assert self.ekf is not None
        r_hat = self.ekf.r_hat
        e_hat = r_hat - self.r_des_world

        std_x, std_y, pxy = self._get_std_and_pxy()

        pos_scale = float(self.cfg.pos_scale)
        std_scale = float(self.cfg.std_scale)
        vel_scale = float(self.cfg.vel_scale)
        s_scale = float(self.cfg.s_scale)

        obs = np.array([
            e_hat[0] / pos_scale,
            e_hat[1] / pos_scale,
            r_hat[0] / pos_scale,
            r_hat[1] / pos_scale,
            std_x / std_scale,
            std_y / std_scale,
            pxy / (std_scale**2),
            self.speed_F / vel_scale,
            math.cos(deg2rad(self.heading_F)),
            math.sin(deg2rad(self.heading_F)),
            self.s_meas_last / s_scale,
            self.leader_speed / max(self.cfg.leader_speed_max, 1e-6),
            math.cos(deg2rad(self.heading_L)),
            math.sin(deg2rad(self.heading_L)),
        ], dtype=np.float32)
        return obs

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        HARD reward: tylko (r_hat, P), info_gain, NIS, koszt sterowania.
        NIE używa r_true.
        """
        assert self.ekf is not None
        r_hat = self.ekf.r_hat
        P = self.ekf.P

        err_est = float(np.linalg.norm(r_hat - self.r_des_world))

        std_x, std_y, _ = self._get_std_and_pxy()
        unc_metric = float(std_x + std_y)

        prev_unc = float(self.prev_unc_metric) if self.prev_unc_metric is not None else unc_metric
        # Reward only when uncertainty decreases (no continuous penalty term).
        unc_reduction = float(prev_unc - unc_metric)
        unc_reduction_pos = float(max(0.0, unc_reduction))
        self.prev_unc_metric = unc_metric

        # NIS penalty tylko nad progiem
        nis = float(self.ekf.last_nis) if np.isfinite(self.ekf.last_nis) else float("nan")
        nis_excess = 0.0
        if np.isfinite(nis):
            nis_excess = max(0.0, nis - self.cfg.nis_95)

        # soft safety (na estymacie)
        rho_hat = float(np.linalg.norm(r_hat))
        close_pen = 0.0
        if rho_hat < self.cfg.rho_soft_min:
            close_pen = float(self.cfg.rho_soft_min - rho_hat)

        r_pos = -self.cfg.w_pos * err_est

        # Uncertainty reduction reward (clipped to positive improvements)
        r_info = self.cfg.w_info * unc_reduction_pos

        # Doppler geometry reward: prefer having some sideways component of v_rel wrt LOS.
        sens_avg = float(self.sens_accum / max(self.sens_count, 1))
        r_sens = self.cfg.w_sens * sens_avg

        # No control penalty (w_ctrl default 0.0). Keep for logging.
        r_ctrl = -self.cfg.w_ctrl * float(np.sum(np.square(action)))

        r_nis = -self.cfg.w_nis * nis_excess
        r_close = -self.cfg.w_close * close_pen

        reward = r_pos + r_info + r_sens + r_ctrl + r_nis + r_close

        terms: Dict[str, float] = {
            "err_est": err_est,
            "std_x": std_x,
            "std_y": std_y,
            "unc_metric": unc_metric,
            "unc_reduction": unc_reduction,
            "unc_reduction_pos": unc_reduction_pos,
            "nis": nis if np.isfinite(nis) else float("nan"),
            "nis_excess": float(nis_excess),
            "rho_hat": rho_hat,
            "sens_avg": sens_avg,
            "sens_count": float(self.sens_count),
            "r_pos": r_pos,
                        "r_info": r_info,
            "r_sens": r_sens,
            "r_ctrl": r_ctrl,
            "r_nis": r_nis,
            "r_close": r_close,
        }

        # prawda – tylko diagnostyka
        if self.cfg.log_truth_diagnostics:
            r_true = self.pL - self.pF
            err_true = float(np.linalg.norm(r_true - self.r_des_world))
            nees = float("nan")
            try:
                Pinv = np.linalg.inv(P)
                e = (r_hat - r_true).reshape(2, 1)
                nees = float((e.T @ Pinv @ e)[0, 0])
            except np.linalg.LinAlgError:
                nees = float("nan")
            terms["err_true_diag"] = err_true
            terms["nees_diag"] = nees

        return float(reward), terms

    def _check_done(self) -> Tuple[bool, bool, str]:
        # sukces: tylko na estymacie i P (bez prawdy)
        assert self.ekf is not None
        r_hat = self.ekf.r_hat
        err_est = float(np.linalg.norm(r_hat - self.r_des_world))
        std_x, std_y, _ = self._get_std_and_pxy()
        std_max = max(std_x, std_y)

        success_now = (err_est < self.cfg.tol_pos_est) and (std_max < self.cfg.tol_std)

        if success_now:
            self.success_streak += 1
        else:
            self.success_streak = 0

        terminated = self.success_streak >= self.cfg.success_hold_steps
        truncated = self.step_count >= self.cfg.max_steps

        if terminated:
            return True, False, "success"
        if truncated:
            return False, True, "max_steps"
        return False, False, "running"

    def _get_info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        assert self.ekf is not None
        info: Dict[str, Any] = {
            "t": float(self.t),
            "step": int(self.step_count),
            "leader_speed": float(self.leader_speed),
            "heading_L_deg": float(self.heading_L),
            "speed_F": float(self.speed_F),
            "heading_F_deg": float(self.heading_F),
            "r_des_world": self.r_des_world.copy(),
            "r_hat": self.ekf.r_hat.copy(),
            "P": self.ekf.P.copy(),
            "s_meas_last": float(self.s_meas_last),
            "innov": float(self.ekf.last_innov) if np.isfinite(self.ekf.last_innov) else float("nan"),
            "S": float(self.ekf.last_S) if np.isfinite(self.ekf.last_S) else float("nan"),
            "nis": float(self.ekf.last_nis) if np.isfinite(self.ekf.last_nis) else float("nan"),
        }

        if self.cfg.log_truth_diagnostics:
            info["r_true_diag"] = (self.pL - self.pF).copy()

        if extra:
            info.update(extra)
        return info

    # --------- render / close ---------

    def render(self):
        if self.render_mode != "human":
            return
        if pygame is None:
            raise RuntimeError("pygame nie jest zainstalowany. Zainstaluj: pip install pygame")

        if not self._pygame_inited:
            pygame.init()
            self._screen = pygame.display.set_mode((self.cfg.screen_w, self.cfg.screen_h))
            pygame.display.set_caption("UUV RL HARD (EKF + Doppler-only) – press [M] manual")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("consolas", 16)
            self._pygame_inited = True

        assert self._screen is not None and self._clock is not None and self._font is not None
        screen = self._screen

        # obsługa eventów również w renderze (toggle manual / reset / quit)
        self._process_pygame_events()
        if self._request_quit:
            # pozwól pętli zewnętrznej to obsłużyć
            return

        screen.fill((10, 10, 25))

        assert self.ekf is not None
        r_hat = self.ekf.r_hat.copy()
        pF_hat = self.pL - r_hat
        pF_des = self.pL - self.r_des_world

        # prędkości do wizualizacji
        vL = vel_vec2(self.leader_speed, self.heading_L)
        vF = vel_vec2(self.speed_F, self.heading_F)

        # kamera
        self._cam.update([self.pL, self.pF, pF_hat, pF_des])

        L_scr = self._cam.world_to_screen(self.pL)
        F_scr = self._cam.world_to_screen(self.pF)
        Fhat_scr = self._cam.world_to_screen(pF_hat)
        Fdes_scr = self._cam.world_to_screen(pF_des)

        # okręgi zasięgu
        for r_m in (50, 100, 200, 400):
            rad_px = int(r_m * self._cam.scale)
            if 5 < rad_px < max(self.cfg.screen_w, self.cfg.screen_h):
                pygame.draw.circle(screen, (35, 40, 70), L_scr, rad_px, 1)

        # obiekty
        pygame.draw.circle(screen, (80, 150, 255), L_scr, 8)        # Leader
        pygame.draw.circle(screen, (80, 230, 130), F_scr, 7)        # Follower true (tylko render)
        pygame.draw.circle(screen, (255, 90, 90), Fhat_scr, 5, 2)   # estymata
        pygame.draw.circle(screen, (255, 210, 100), Fdes_scr, 6, 2) # desired point

        # linie
        pygame.draw.line(screen, (150, 150, 200), L_scr, F_scr, 1)
        pygame.draw.line(screen, (255, 210, 100), L_scr, Fhat_scr, 1)

        # strzałki prędkości (jak w Twoim pierwotnym kodzie)
        def draw_velocity_arrow(pos_world: np.ndarray, v_world: np.ndarray, color, length_scale: float = 10.0, width: int = 3):
            """
            Rysuje wektor prędkości: od pos do pos + v*length_scale (w metrach).
            length_scale=10: 2m/s -> 20m strzałka.
            """
            start = self._cam.world_to_screen(pos_world)
            end_world = pos_world + v_world * float(length_scale)
            end = self._cam.world_to_screen(end_world)
            pygame.draw.line(screen, color, start, end, width)

        draw_velocity_arrow(self.pL, vL, (120, 190, 255), length_scale=10.0, width=3)
        draw_velocity_arrow(self.pF, vF, (80, 230, 130), length_scale=10.0, width=3)

        # HUD
        def draw_text(txt, x, y, color=(230, 230, 230)):
            surf = self._font.render(txt, True, color)
            screen.blit(surf, (x, y))

        std_x, std_y, _ = self._get_std_and_pxy()
        err_est = float(np.linalg.norm(r_hat - self.r_des_world))
        nis = float(self.ekf.last_nis) if np.isfinite(self.ekf.last_nis) else float("nan")
        mode = "MANUAL" if self.manual_override else "POLICY"

        y0 = 8
        draw_text(f"t={self.t:6.1f}s  step={self.step_count:4d}  action_dt={self.cfg.action_dt:.1f}s  mode={mode}  ([M] toggle)", 10, y0); y0 += 18
        draw_text(f"Leader: v={self.leader_speed:4.2f} m/s  heading={self.heading_L:6.1f} deg", 10, y0); y0 += 18
        draw_text(f"Follower: v={self.speed_F:4.2f} m/s  heading={self.heading_F:6.1f} deg  (↑/↓ accel, ←/→ turn, SHIFT=fine)", 10, y0); y0 += 18
        draw_text(f"err_est=|rhat-rdes|={err_est:6.2f} m   std_x={std_x:6.1f} std_y={std_y:6.1f} m", 10, y0); y0 += 18
        draw_text(f"s_meas_last={self.s_meas_last:6.3f} m/s   NIS={nis:6.2f} (95%~{self.cfg.nis_95})", 10, y0); y0 += 18
        draw_text("Colors: BLUE=Leader, GREEN=Follower(true), RED=estimate, YELLOW=desired", 10, y0, (200, 200, 140)); y0 += 18
        draw_text("BACKSPACE reset epizodu, ESC/Q wyjście", 10, y0, (200, 200, 140)); y0 += 18

        if self.cfg.log_truth_diagnostics:
            r_true = self.pL - self.pF
            err_true = float(np.linalg.norm(r_true - self.r_des_world))
            draw_text(f"DIAG (not used): err_true={err_true:6.2f} m", 10, y0, (200, 200, 140)); y0 += 18

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._pygame_inited and pygame is not None:
            pygame.quit()
        self._pygame_inited = False
        self._screen = None
        self._clock = None
        self._font = None


# ===================== SB3 training utilities =================================

def make_env(seed: int, render: bool, cfg: Optional[UUVConfig] = None):
    def _init():
        env = UUVRelPosHardEnv(config=cfg, render_mode=("human" if render else "none"))
        env.reset(seed=seed)
        return env
    return _init


def cmd_train(args: argparse.Namespace) -> None:
    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    except Exception as e:
        raise RuntimeError("Zainstaluj: pip install stable-baselines3[extra]") from e

    class InfoTensorboardCallback(BaseCallback):
        """
        Loguje wybrane klucze z info do TensorBoard (rollout/*).
        Działa dla VecEnv: infos to lista dictów.
        """
        def __init__(self, keys, verbose=0):
            super().__init__(verbose)
            self.keys = list(keys)

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", None)
            if infos is None:
                return True
            for k in self.keys:
                vals = []
                for info in infos:
                    if k in info:
                        try:
                            fv = float(info[k])
                            if math.isfinite(fv):
                                vals.append(fv)
                        except Exception:
                            pass
                if vals:
                    self.logger.record(f"rollout/{k}", float(np.mean(vals)))
            return True

    os.makedirs(args.models_dir, exist_ok=True)

    cfg = UUVConfig(
        action_dt=args.action_dt,
        log_truth_diagnostics=args.log_truth_diag,
        leader_speed_min=args.leader_speed_min,
        leader_speed_max=args.leader_speed_max,
        leader_heading_min_deg=args.leader_heading_min,
        leader_heading_max_deg=args.leader_heading_max,
    )

    n_envs = int(args.n_envs)
    env = DummyVecEnv([make_env(seed=args.seed + i, render=False, cfg=cfg) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(seed=args.seed + 10000, render=False, cfg=cfg)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.norm_reward = False

    run_name = args.run_name or f"{args.algo}_uuv_hard_{int(time.time())}"
    model_path = os.path.join(args.models_dir, f"{args.algo}_uuv_hard")
    tb_log = args.tb

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.models_dir,
        log_path=args.models_dir,
        eval_freq=max(4000 // n_envs, 500),
        deterministic=True,
        render=False,
    )

    info_cb = InfoTensorboardCallback(keys=[
        "err_est", "std_x", "std_y", "unc_metric", "unc_reduction_pos", "sens_avg",
        "nis", "nis_excess",
        "r_pos", "r_info", "r_sens", "r_ctrl", "r_nis",
        # diagnostyka (tylko jeśli --log-truth-diag)
        "err_true_diag", "nees_diag",
        # manual
        "manual_override",
    ])

    common_kwargs = dict(verbose=1, tensorboard_log=tb_log, seed=args.seed)

    if args.algo.lower() == "ppo":
        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=2048 // n_envs,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            **common_kwargs,
        )
    elif args.algo.lower() == "sac":
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            gamma=0.99,
            buffer_size=250_000,
            batch_size=256,
            tau=0.02,
            train_freq=1,
            gradient_steps=1,
            learning_starts=5000,
            **common_kwargs,
        )
    else:
        raise ValueError("algo musi być ppo albo sac")

    model.learn(total_timesteps=int(args.steps), tb_log_name=run_name, callback=[eval_cb, info_cb])

    model.save(model_path)
    env.save(os.path.join(args.models_dir, "vecnormalize.pkl"))

    print(f"[OK] Zapisano model: {model_path}.zip")
    print(f"[OK] Zapisano VecNormalize: {os.path.join(args.models_dir, 'vecnormalize.pkl')}")
    print(f"TensorBoard: tensorboard --logdir {tb_log}")


def _load_vecnormalize(models_dir: str):
    try:
        from stable_baselines3.common.vec_env import VecNormalize
    except Exception:
        return None
    vn_path = os.path.join(models_dir, "vecnormalize.pkl")
    if os.path.exists(vn_path):
        return VecNormalize.load(vn_path)
    return None


def cmd_play(args: argparse.Namespace) -> None:
    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except Exception as e:
        raise RuntimeError("Zainstaluj stable-baselines3[extra].") from e

    cfg = UUVConfig(
        action_dt=args.action_dt,
        log_truth_diagnostics=args.log_truth_diag,
        leader_speed_min=args.leader_speed_min,
        leader_speed_max=args.leader_speed_max,
        leader_heading_min_deg=args.leader_heading_min,
        leader_heading_max_deg=args.leader_heading_max,
    )

    env = DummyVecEnv([make_env(seed=args.seed, render=args.render, cfg=cfg)])

    vn_path = os.path.join(args.models_dir, "vecnormalize.pkl")
    if os.path.exists(vn_path):
        env = VecNormalize.load(vn_path, env)
        env.training = False
        env.norm_reward = False

    if args.algo.lower() == "ppo":
        model = PPO.load(args.model, env=env)
    elif args.algo.lower() == "sac":
        model = SAC.load(args.model, env=env)
    else:
        raise ValueError("algo musi być ppo albo sac")

    for ep in range(int(args.episodes)):
        obs = env.reset()
        done = False
        ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            ep_rew += float(reward[0])

            if args.render:
                # render (manual override i eventy obsługiwane wewnątrz env)
                env.envs[0].render()

            # jeśli użytkownik nacisnął BACKSPACE/ESC -> env zwróci truncated z reason w info,
            # co naturalnie zakończy epizod / pętlę.
        print(f"[PLAY] episode {ep+1}/{args.episodes} return={ep_rew:.2f}")


def cmd_random(args: argparse.Namespace) -> None:
    cfg = UUVConfig(
        action_dt=args.action_dt,
        log_truth_diagnostics=args.log_truth_diag,
        leader_speed_min=args.leader_speed_min,
        leader_speed_max=args.leader_speed_max,
        leader_heading_min_deg=args.leader_heading_min,
        leader_heading_max_deg=args.leader_heading_max,
    )

    env = UUVRelPosHardEnv(config=cfg, render_mode=("human" if args.render else "none"))
    for ep in range(int(args.episodes)):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_rew = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_rew += reward
            if args.render:
                env.render()
        print(f"[RANDOM] episode {ep+1}/{args.episodes} return={ep_rew:.2f}")
    env.close()


def cmd_sim(args: argparse.Namespace) -> None:
    """
    Interaktywny "symulator" sterowany klawiszami.
    - env.step dostaje dummy akcję, ale manual_override=True nadpisuje ją klawiaturą.
    """
    cfg = UUVConfig(
        action_dt=args.action_dt,
        log_truth_diagnostics=args.log_truth_diag,
        leader_speed_min=args.leader_speed_min,
        leader_speed_max=args.leader_speed_max,
        leader_heading_min_deg=args.leader_heading_min,
        leader_heading_max_deg=args.leader_heading_max,
    )
    env = UUVRelPosHardEnv(config=cfg, render_mode="human")
    obs, info = env.reset(seed=args.seed)
    env.manual_override = True  # start w manual
    env.render()  # zainicjalizuj pygame przed pierwszym krokiem (klawisze działają od razu)

    while True:
        # dummy action; zostanie nadpisane przez klawiaturę (manual_override=True)
        obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        env.render()

        if truncated or terminated:
            reason = info.get("term_reason", "")
            if reason == "quit":
                break
            # reset (manual_reset lub max_steps/success)
            obs, info = env.reset(seed=args.seed)
            env.manual_override = True

    env.close()


# ===================== CLI ====================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--action-dt", type=float, default=2.0, help="czas utrzymania akcji [s] (np 2-5)")
        sp.add_argument("--leader-speed-min", type=float, default=1.0)
        sp.add_argument("--leader-speed-max", type=float, default=3.0)
        sp.add_argument("--leader-heading-min", type=float, default=0.0)
        sp.add_argument("--leader-heading-max", type=float, default=360.0)
        sp.add_argument("--log-truth-diag", action="store_true", help="diagnostycznie loguj err_true/NEES (nie do reward/success)")

    # train
    pt = sub.add_parser("train")
    pt.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    pt.add_argument("--steps", type=int, default=800_000)
    pt.add_argument("--tb", type=str, default="runs/uuv_hard")
    pt.add_argument("--models-dir", type=str, default="models")
    pt.add_argument("--run-name", type=str, default="")
    pt.add_argument("--n-envs", type=int, default=8)
    add_common(pt)

    # play
    pp = sub.add_parser("play")
    pp.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    pp.add_argument("--model", type=str, required=True)
    pp.add_argument("--models-dir", type=str, default="models")
    pp.add_argument("--episodes", type=int, default=3)
    pp.add_argument("--render", action="store_true")
    add_common(pp)

    # random
    pr = sub.add_parser("random")
    pr.add_argument("--episodes", type=int, default=1)
    pr.add_argument("--render", action="store_true")
    add_common(pr)

    # sim (manual)
    ps = sub.add_parser("sim")
    ps.add_argument("--render", action="store_true", help="(zostaw) – kompatybilność; sim zawsze renderuje")
    add_common(ps)

    return p


def main():
    args = build_arg_parser().parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "play":
        cmd_play(args)
    elif args.cmd == "random":
        cmd_random(args)
    elif args.cmd == "sim":
        cmd_sim(args)
    else:
        raise ValueError("unknown cmd")


if __name__ == "__main__":
    main()
