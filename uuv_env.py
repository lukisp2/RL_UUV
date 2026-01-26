"""Minimal gymnasium environment for relative-position UUV tracking with a light
RGB render helper.

The leader moves straight with a fixed speed/heading. The agent controls the
follower (turn + acceleration) and tries to keep small range to the leader.
Observations contain noisy relative state, Doppler radial speed and (optionally)
range anchors every few seconds when ``use_range_anchors=True``.
"""

import math
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def deg2rad(angle_deg: float) -> float:
    return angle_deg * math.pi / 180.0


def wrap360(angle_deg: float) -> float:
    angle_deg = angle_deg % 360.0
    return angle_deg + 360.0 if angle_deg < 0.0 else angle_deg


def vel_vec2(speed: float, heading_deg: float) -> np.ndarray:
    heading_rad = deg2rad(heading_deg)
    return np.array(
        [
            speed * math.cos(heading_rad),
            speed * math.sin(heading_rad),
        ],
        dtype=float,
    )


def radial_speed(r_vec: np.ndarray, v_rel: np.ndarray) -> float:
    rho = float(np.linalg.norm(r_vec))
    if rho <= 1e-9:
        return 0.0
    rhat = r_vec / rho
    return -float(np.dot(rhat, v_rel))


class EKFRelPos2D:
    """Copy of the EKF logic from symulator.py (2D relative position)."""

    def __init__(self, r0, pos_std0=80.0, q_std=0.10,
                 sigma_s=0.05, sigma_r=2.5):
        self.r_hat = np.array(r0, dtype=float)
        self.P = np.diag([pos_std0**2, pos_std0**2])
        self.q_var = q_std**2
        self.Rs = np.array([[sigma_s**2]], dtype=float)
        self.sigma_r2 = sigma_r**2

    def predict(self, v_rel, dt):
        self.r_hat = self.r_hat + dt * np.array(v_rel, dtype=float)
        Q = self.q_var * max(dt, 1e-3) * np.eye(2, dtype=float)
        self.P = self.P + Q

    def update_s(self, v_rel, s_meas):
        v_rel = np.array(v_rel, dtype=float)
        rho = float(np.linalg.norm(self.r_hat))
        if rho <= 1e-6:
            return
        rhat = self.r_hat / rho
        proj = float(np.dot(rhat, v_rel))
        h = -proj
        H = -(v_rel - proj * rhat) / rho
        H = H.reshape(1, 2)
        z = np.array([[s_meas]], dtype=float)
        y = z - np.array([[h]], dtype=float)
        S = H @ self.P @ H.T + self.Rs
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        K = self.P @ H.T @ Sinv
        self.r_hat = self.r_hat + (K @ y).flatten()
        self.P = (np.eye(2) - K @ H) @ self.P

    def update_range(self, rho_meas):
        rho = float(np.linalg.norm(self.r_hat))
        if rho <= 1e-6:
            return
        rhat = self.r_hat / rho
        H = rhat.reshape(1, 2)
        h = np.array([[rho]], dtype=float)
        z = np.array([[rho_meas]], dtype=float)
        R = np.array([[self.sigma_r2]], dtype=float)
        S = H @ self.P @ H.T + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        K = self.P @ H.T @ Sinv
        self.r_hat = self.r_hat + (K @ (z - h)).flatten()
        self.P = (np.eye(2) - K @ H) @ self.P


class UUVRelPosEnv(gym.Env):
    """2D relative-position control task."""

    metadata = {"render_modes": ["rgb_array", "human", "pygame"], "render_fps": 10}

    def __init__(self, use_range_anchors: bool = True, render_mode: Optional[str] = None,
                 init_known_r0: bool = True) -> None:
        super().__init__()
        self.use_range_anchors = use_range_anchors
        self.render_mode = render_mode
        self.init_known_r0 = init_known_r0
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}, got {render_mode}")

        # Dynamics / sensor params
        # symulator.py uses ~60 Hz z time_scale=3 => ~0.15 s kroku
        self.dt = 0.15
        self.leader_speed = 2.0
        self.leader_heading = 0.0
        self.f_min_speed = 0.2
        self.f_max_speed = 4.0
        self.turn_rate_deg = 80.0
        self.accel = 1.0
        self.sigma_heading_deg = 1.0
        self.sigma_speed = 0.02
        self.sigma_s = 0.05
        self.sigma_rho = 2.5
        self.q_std = 0.10
        self.pos_std_known = 10.0
        self.pos_std_unknown = 300.0
        self.s_meas_period = 1.0
        self.anchor_period = 7.0

        # Episode logic
        self.max_steps = 2000
        self.fail_dist = 800.0
        self.target_radius = 20.0

        # RNG
        self.np_random = np.random.default_rng()
        self._last_frame = None
        # pygame state (lazy init)
        self._pg = None
        self._pg_screen = None
        self._pg_clock = None
        self._pg_size = (640, 640)

        # Gym API
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        # [r_hat_x, r_hat_y, v_rel_nav_x, v_rel_nav_y,
        #  s_meas(last), rho_meas(last anchor or 0), sin(heading_meas), cos(heading_meas),
        #  speed_meas, anchor_flag]
        obs_dim = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State placeholders
        self.pL = np.zeros(2, dtype=float)
        self.pF = np.zeros(2, dtype=float)
        self.heading_F = 0.0
        self.speed_F = 0.0
        self.time_s = 0.0
        self.step_count = 0
        self.next_anchor_time = self.anchor_period
        self.next_s_meas_time = self.s_meas_period
        self.s_meas_last = 0.0
        self.rho_meas_last = 0.0
        self.anchor_used_last = 0.0
        self._last_v_rel_nav = np.zeros(2, dtype=float)
        self._last_heading_meas = 0.0
        self._last_speed_meas = 0.0
        self.ekf: Optional[EKFRelPos2D] = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.time_s = 0.0
        self.step_count = 0
        self.next_anchor_time = self.anchor_period
        self.next_s_meas_time = self.s_meas_period
        self.s_meas_last = 0.0
        self.rho_meas_last = 0.0
        self.anchor_used_last = 0.0
        self._last_v_rel_nav = np.zeros(2, dtype=float)
        self._last_heading_meas = 0.0
        self._last_speed_meas = 0.0

        # Leader start
        self.pL = np.array([0.0, 0.0], dtype=float)

        # Follower start
        self.pF = np.array([-120.0, 60.0], dtype=float)
        self.heading_F = 30.0
        self.speed_F = 2.0

        r_true0 = self.pL - self.pF
        if self.init_known_r0:
            r0 = r_true0.copy()
            pos_std0 = self.pos_std_known
        else:
            r0 = np.array([0.0, 0.0], dtype=float)
            pos_std0 = self.pos_std_unknown
        self.ekf = EKFRelPos2D(
            r0,
            pos_std0=pos_std0,
            q_std=self.q_std,
            sigma_s=self.sigma_s,
            sigma_r=self.sigma_rho,
        )

        # initial nav-like signals (no noise to start)
        self._last_heading_meas = self.heading_F
        self._last_speed_meas = self.speed_F
        self._last_v_rel_nav = vel_vec2(self.leader_speed, self.leader_heading) - vel_vec2(self.speed_F, self.heading_F)

        obs = self._get_obs()
        info = {"rho_true": float(np.linalg.norm(self.pL - self.pF)), "time_s": 0.0}
        if self.render_mode is not None:
            if self.render_mode == "rgb_array":
                self._last_frame = self._build_frame()
            else:
                self._render_pygame()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=float)
        action = np.clip(action, -1.0, 1.0)

        self._advance_dynamics(action)
        self.step_count += 1
        self.time_s += self.dt

        # true kinematics
        vL_true = vel_vec2(self.leader_speed, self.leader_heading)
        vF_true = vel_vec2(self.speed_F, self.heading_F)
        r_true = self.pL - self.pF
        rho_true = float(np.linalg.norm(r_true))
        v_rel_true = vL_true - vF_true

        # nav measurements (noisy heading/speed)
        heading_meas = self.heading_F + self.np_random.normal(0.0, self.sigma_heading_deg)
        speed_meas = self.speed_F + self.np_random.normal(0.0, self.sigma_speed)
        self._last_heading_meas = heading_meas
        self._last_speed_meas = speed_meas
        vF_ekf = vel_vec2(speed_meas, heading_meas)
        v_rel_ekf = vL_true - vF_ekf
        self._last_v_rel_nav = v_rel_ekf

        # EKF predict
        if self.ekf is None:
            self.ekf = EKFRelPos2D(r_true, pos_std0=self.pos_std_known, q_std=self.q_std,
                                   sigma_s=self.sigma_s, sigma_r=self.sigma_rho)
        self.ekf.predict(v_rel_ekf, self.dt)

        anchor_used = False
        rho_meas = 0.0
        if self.use_range_anchors and self.time_s >= self.next_anchor_time:
            rho_meas = rho_true + self.np_random.normal(0.0, self.sigma_rho)
            self.ekf.update_range(rho_meas)
            self.next_anchor_time += self.anchor_period
            anchor_used = True

        # Doppler measurement every s_meas_period
        if self.time_s >= self.next_s_meas_time:
            s_true = radial_speed(r_true, v_rel_true)
            s_meas = s_true + self.np_random.normal(0.0, self.sigma_s)
            self.s_meas_last = s_meas
            self.ekf.update_s(v_rel_ekf, s_meas)
            self.next_s_meas_time += self.s_meas_period

        self.rho_meas_last = rho_meas if anchor_used else 0.0
        self.anchor_used_last = 1.0 if anchor_used else 0.0

        r_hat = self.ekf.r_hat
        pF_hat = self.pL - r_hat
        err_pos = float(np.linalg.norm(pF_hat - self.pF))

        obs = self._get_obs()

        reward = -0.05 * rho_true - 0.01 * float(np.sum(np.square(action)))
        if rho_true < self.target_radius:
            reward += 5.0

        terminated = rho_true > self.fail_dist
        truncated = self.step_count >= self.max_steps

        info = {
            "rho_true": rho_true,
            "time_s": self.time_s,
            "anchor_used": anchor_used,
            "err_pos": err_pos,
        }
        if self.render_mode is not None:
            if self.render_mode == "rgb_array":
                self._last_frame = self._build_frame()
            else:
                self._render_pygame()
        return obs, reward, terminated, truncated, info

    def _advance_dynamics(self, action: np.ndarray) -> None:
        turn_cmd, accel_cmd = action
        self.heading_F = wrap360(
            self.heading_F + turn_cmd * self.turn_rate_deg * self.dt
        )
        self.speed_F = float(
            np.clip(
                self.speed_F + accel_cmd * self.accel * self.dt,
                self.f_min_speed,
                self.f_max_speed,
            )
        )

        vL = vel_vec2(self.leader_speed, self.leader_heading)
        vF = vel_vec2(self.speed_F, self.heading_F)
        self.pL = self.pL + vL * self.dt
        self.pF = self.pF + vF * self.dt

    def _get_obs(self) -> np.ndarray:
        r_hat = self.ekf.r_hat if self.ekf is not None else np.zeros(2, dtype=float)
        obs = np.array(
            [
                r_hat[0],
                r_hat[1],
                self._last_v_rel_nav[0],
                self._last_v_rel_nav[1],
                self.s_meas_last,
                self.rho_meas_last,
                math.sin(deg2rad(self._last_heading_meas)),
                math.cos(deg2rad(self._last_heading_meas)),
                self._last_speed_meas,
                self.anchor_used_last,
            ],
            dtype=np.float32,
        )
        return obs

    def _world_to_px(self, pos: np.ndarray, center: np.ndarray, scale: float, w: int, h: int) -> Tuple[int, int]:
        rel = pos - center
        sx = w / 2 + rel[0] * scale
        sy = h / 2 - rel[1] * scale
        return int(round(sx)), int(round(sy))

    def _draw_disc(self, img: np.ndarray, cx: int, cy: int, r: int, color: Tuple[int, int, int]) -> None:
        h, w, _ = img.shape
        y_idx, x_idx = np.ogrid[:h, :w]
        mask = (x_idx - cx) ** 2 + (y_idx - cy) ** 2 <= r * r
        img[mask] = color

    def _build_frame(self, size: int = 400) -> np.ndarray:
        """Return a small RGB array with leader (blue) and follower (green)."""
        w = h = size
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = 12  # dark background

        pts = np.vstack([self.pL, self.pF])
        center = pts.mean(axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        dmax = float(max(dists.max(), 10.0))
        scale = 0.35 * min(w, h) / dmax

        L_px = self._world_to_px(self.pL, center, scale, w, h)
        F_px = self._world_to_px(self.pF, center, scale, w, h)
        r_hat = self.ekf.r_hat if self.ekf is not None else np.zeros(2, dtype=float)
        F_est_px = self._world_to_px(self.pL - r_hat, center, scale, w, h)

        # line leader-follower
        self._draw_line(frame, L_px, F_px, color=(80, 80, 160))

        self._draw_disc(frame, L_px[0], L_px[1], 8, (70, 140, 255))
        self._draw_disc(frame, F_px[0], F_px[1], 8, (90, 220, 130))
        self._draw_disc(frame, F_est_px[0], F_est_px[1], 6, (255, 210, 100))

        return frame

    def _draw_line(self, img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        """Simple Bresenham-style line."""
        x1, y1 = p1
        x2, y2 = p2
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        while True:
            if 0 <= x1 < img.shape[1] and 0 <= y1 < img.shape[0]:
                img[y1, x1] = color
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy

    def render(self):
        if self.render_mode == "rgb_array":
            return np.copy(self._last_frame) if self._last_frame is not None else None
        if self.render_mode in ("human", "pygame"):
            self._render_pygame()
            return None
        return None

    def close(self):
        if self._pg is not None:
            try:
                self._pg.quit()
            except Exception:
                pass
            self._pg = None
            self._pg_screen = None
            self._pg_clock = None

    def _render_pygame(self) -> None:
        """Draw a simple view using pygame (intended for 1 env, debug only)."""
        if self._pg is None:
            try:
                import pygame
            except ImportError as exc:
                raise RuntimeError("render_mode='human' or 'pygame' requires pygame installed.") from exc
            self._pg = pygame
            pygame.init()
            self._pg_screen = pygame.display.set_mode(self._pg_size)
            pygame.display.set_caption("UUVRelPosEnv (pygame)")
            self._pg_clock = pygame.time.Clock()

        pg = self._pg
        screen = self._pg_screen
        if screen is None:
            return

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                self._pg = None
                self._pg_screen = None
                return

        w, h = self._pg_size
        screen.fill((10, 10, 25))

        pts = np.vstack([self.pL, self.pF])
        center = pts.mean(axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        dmax = float(max(dists.max(), 10.0))
        scale = 0.35 * min(w, h) / dmax

        L_px = self._world_to_px(self.pL, center, scale, w, h)
        F_px = self._world_to_px(self.pF, center, scale, w, h)
        # estimated follower from EKF
        r_hat = self.ekf.r_hat if self.ekf is not None else np.zeros(2, dtype=float)
        pF_est = self.pL - r_hat
        Fest_px = self._world_to_px(pF_est, center, scale, w, h)

        # Leader-follower line
        pg.draw.line(screen, (90, 90, 160), L_px, F_px, 2)

        # Leader and follower markers
        pg.draw.circle(screen, (70, 140, 255), L_px, 9)
        pg.draw.circle(screen, (90, 220, 130), F_px, 9)
        # Estimated follower marker
        pg.draw.circle(screen, (255, 210, 100), Fest_px, 7, 2)

        pg.display.flip()
        if self._pg_clock:
            self._pg_clock.tick(self.metadata.get("render_fps", 10))
