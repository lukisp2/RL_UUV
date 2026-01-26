
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig11_example_trajectories.py

Figure 11-style *matched episode* comparison:
  RL vs 3 baseline controllers (PID, PID+exc, Random)

Key features
------------
1) Single episode -> combined figure (PNG, optional PDF)
2) Review mode -> multi-page PDF with many episodes (e.g. 300 pages)
3) NEW: Export each panel as a *separate* PNG into a folder named after the episode
   (e.g., out_root/ep176/a_trajectories.png, b_true_formation_error.png, ...)

Folder structure expected under --root
--------------------------------------
Canonical layout:
  root/rl/traces/ep001.csv
  root/baseline_pid/traces/ep001.csv
  root/baseline_pid_exc/traces/ep001.csv
  root/baseline_random/traces/ep001.csv

If your tree differs, the script falls back to recursive search:
  root/**/<controller>/**/epXXX.csv

Panels
------
Core (paper) panels:
  (a) Trajectories: p_L(t), desired follower p_{F,des}(t), p_F(t)
  (b) True formation error: e_{form,true}(t)
  (c) Estimator uncertainty: sigma_max(t) = std_max(t)
  (d) Information & excitation:
        solid  : lambda_min(J_w(t))   (log, left axis)
        dashed : vbar_perp(t) ~ sens_avg(t) (right axis)

Extra motion panels (optional / extended layout and export):
  (e) follower speed      v_F(t)
  (f) follower course     psi_F(t)   [deg]
  (g) follower yaw-rate   omega_F(t) = d psi_F / dt   [deg/s] (or rad/s)

IMPORTANT about v_F, psi_F, omega_F
-----------------------------------
Most trace CSVs produced by uuv_compare_experiment_tables.py do NOT store follower speed/heading.
To remain compatible, this script derives them from true follower positions (pF_x, pF_y) and time t.

If your CSV contains columns:
  - speed_F   (or v_F, vF)
  - heading_F_deg (or psi_F_deg, course_F_deg)
  - omega_F_deg_s (or yaw_rate_F_deg_s)
then these are preferred where possible.

Examples (Windows)
------------------
Single episode, export panels into out/ep176/:
  python fig11_example_trajectories.py ^
    --root "C:\\...\\eval_logs\\traces" ^
    --episode 176 ^
    --export-panels ^
    --export-root "out" ^
    --layout extended

Single episode, combined figure:
  python fig11_example_trajectories.py ^
    --root "C:\\...\\eval_logs\\traces" ^
    --episode 176 ^
    --out "fig11_ep176.png" ^
    --layout extended ^
    --save-pdf

Review PDF for all matched episodes (or 1-300):
  python fig11_example_trajectories.py ^
    --root "C:\\...\\eval_logs\\traces" ^
    --pdf-all ^
    --pdf-out "fig11_all_episodes.pdf" ^
    --episodes 1-300 ^
    --layout extended ^
    --title-mode review

"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


# ---------------------------
# Controller config
# ---------------------------

CTRL_ORDER = ["rl", "baseline_pid", "baseline_pid_exc", "baseline_random"]
CTRL_LABEL = {
    "rl": "RL",
    "baseline_pid": "PID",
    "baseline_pid_exc": "PID+exc",
    "baseline_random": "Random",
}
CTRL_COLOR = {
    "rl": "C0",
    "baseline_pid": "C1",
    "baseline_pid_exc": "C2",
    "baseline_random": "C3",
}

# Minimum columns required to build all panels (speed/course derived from pF_x,pF_y,t)
REQUIRED_COLS = [
    "t",
    "pL_x",
    "pL_y",
    "pF_x",
    "pF_y",
    "r_des_x",
    "r_des_y",
    "r_mix_x",
    "r_mix_y",
    "err_true_form",
    "std_max",
    "sens_avg",
    "fim_win_eig_min",
]

_EP_RE = re.compile(r"^ep(\d+)\.csv$", re.IGNORECASE)


# ---------------------------
# Math helpers
# ---------------------------

def _wrap360_deg(deg: np.ndarray) -> np.ndarray:
    d = np.asarray(deg, dtype=float)
    return np.mod(d, 360.0)

def _nan_ffill_bfill(x: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    return s.ffill().bfill().to_numpy(dtype=float)

def _nan_unwrap(rad: np.ndarray) -> np.ndarray:
    """Unwrap radians while preserving NaNs."""
    rad = np.asarray(rad, dtype=float)
    m = np.isfinite(rad)
    if np.sum(m) < 2:
        return np.full_like(rad, np.nan, dtype=float)
    rad_fill = _nan_ffill_bfill(rad)
    rad_uw = np.unwrap(rad_fill)
    rad_uw[~m] = np.nan
    return rad_uw

def _safe_gradient(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2 or x.size < 2:
        return np.full_like(y, np.nan, dtype=float)
    m = np.isfinite(y) & np.isfinite(x)
    if np.sum(m) < 2:
        return np.full_like(y, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.gradient(y, x)


# ---------------------------
# IO helpers
# ---------------------------

def _load_trace(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns {missing}: {path}")
    # sort by time just in case
    try:
        df = df.sort_values("t").reset_index(drop=True)
    except Exception:
        pass
    return df

def _derive_motion_from_positions(df: pd.DataFrame, *, speed_eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive vx, vy, speed, course_rad from pF(t)."""
    t = df["t"].to_numpy(dtype=float)
    x = df["pF_x"].to_numpy(dtype=float)
    y = df["pF_y"].to_numpy(dtype=float)

    vx = _safe_gradient(x, t)
    vy = _safe_gradient(y, t)
    speed = np.sqrt(vx * vx + vy * vy)

    course_rad = np.arctan2(vy, vx)
    bad = ~np.isfinite(speed) | (speed < float(speed_eps))
    course_rad = course_rad.astype(float)
    course_rad[bad] = np.nan
    return vx, vy, speed, course_rad

def _pick_first_existing(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used by plots."""
    out = df.copy()

    # Desired follower position: pF_des = pL - r_des  (since r = pL - pF)
    out["pFdes_x"] = out["pL_x"] - out["r_des_x"]
    out["pFdes_y"] = out["pL_y"] - out["r_des_y"]

    # Estimated follower position (from mixture estimate): pF_hat = pL - r_mix
    out["pFhat_x"] = out["pL_x"] - out["r_mix_x"]
    out["pFhat_y"] = out["pL_y"] - out["r_mix_y"]

    # Follower speed / heading / yaw-rate
    # Prefer stored columns if present, else derive from pF_x,pF_y,t
    col_speed = _pick_first_existing(out, ["speed_F", "v_F", "vF"])
    col_heading_deg = _pick_first_existing(out, ["heading_F_deg", "psi_F_deg", "course_F_deg"])
    col_omega_deg_s = _pick_first_existing(out, ["omega_F_deg_s", "yaw_rate_F_deg_s"])

    # Speed
    if col_speed is not None:
        out["speed_F"] = pd.to_numeric(out[col_speed], errors="coerce").to_numpy(dtype=float)
        # Derive course/omega from positions anyway (robust), unless explicit heading exists
        _, _, _, course_rad_d = _derive_motion_from_positions(out)
    else:
        _, _, speed_d, course_rad_d = _derive_motion_from_positions(out)
        out["speed_F"] = speed_d

    # Course (deg)
    if col_heading_deg is not None:
        # Assume deg, but keep both wrap/unwrap
        course_deg = pd.to_numeric(out[col_heading_deg], errors="coerce").to_numpy(dtype=float)
        out["psi_F_deg_wrap"] = _wrap360_deg(course_deg)
        # unwrap
        course_rad = np.radians(course_deg)
        course_rad_uw = _nan_unwrap(course_rad)
        out["psi_F_deg_unwrap"] = np.degrees(course_rad_uw)
    else:
        out["psi_F_deg_wrap"] = _wrap360_deg(np.degrees(course_rad_d))
        course_rad_uw = _nan_unwrap(course_rad_d)
        out["psi_F_deg_unwrap"] = np.degrees(course_rad_uw)

    # Omega (deg/s)
    if col_omega_deg_s is not None:
        out["omega_F_deg_s"] = pd.to_numeric(out[col_omega_deg_s], errors="coerce").to_numpy(dtype=float)
    else:
        t = out["t"].to_numpy(dtype=float)
        psi_unwrap = out["psi_F_deg_unwrap"].to_numpy(dtype=float)
        omega = _safe_gradient(psi_unwrap, t)
        out["omega_F_deg_s"] = omega

    return out


# ---------------------------
# Path discovery
# ---------------------------

def _resolve_from_root(root: Path, episode: int) -> Dict[str, Path]:
    """Resolve controller CSV paths from a root dir."""
    ep_name = f"ep{int(episode):03d}.csv"
    out: Dict[str, Path] = {}

    # 1) canonical
    for ctrl in CTRL_ORDER:
        p = root / ctrl / "traces" / ep_name
        if p.exists():
            out[ctrl] = p

    # 2) glob fallback for missing
    for ctrl in CTRL_ORDER:
        if ctrl in out:
            continue
        hits = list(root.glob(f"**/{ctrl}/**/{ep_name}"))
        hits = [h for h in hits if h.is_file()]
        if hits:
            hits.sort(key=lambda x: len(str(x)))
            out[ctrl] = hits[0]

    missing = [c for c in CTRL_ORDER if c not in out]
    if missing:
        raise FileNotFoundError(
            "Could not locate all controller traces under root. Missing: "
            + ", ".join(missing)
            + f"\nroot={root}\nepisode={episode} (expected file {ep_name})"
        )
    return out

def _episodes_in_ctrl(root: Path, ctrl: str) -> Set[int]:
    eps: Set[int] = set()
    canonical = root / ctrl / "traces"
    if canonical.exists():
        for p in canonical.glob("ep*.csv"):
            m = _EP_RE.match(p.name)
            if m:
                eps.add(int(m.group(1)))
    for p in root.glob(f"**/{ctrl}/**/ep*.csv"):
        if not p.is_file():
            continue
        m = _EP_RE.match(p.name)
        if m:
            eps.add(int(m.group(1)))
    return eps

def _available_matched_episodes(root: Path) -> List[int]:
    sets = [_episodes_in_ctrl(root, c) for c in CTRL_ORDER]
    if not sets:
        return []
    common = set.intersection(*sets)
    return sorted(common)

def _parse_episode_spec(spec: Optional[str]) -> Optional[List[int]]:
    """Parse episode selection: 'all', '1-300', '1,2,5-10', ..."""
    if spec is None:
        return None
    s = str(spec).strip().lower()
    if s in ("", "all", "*"):
        return None
    s = s.replace(" ", "")
    eps: Set[int] = set()
    for part in s.split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
        elif ":" in part:
            a, b = part.split(":", 1)
        else:
            eps.add(int(part))
            continue
        i0, i1 = int(a), int(b)
        if i1 < i0:
            i0, i1 = i1, i0
        eps.update(range(i0, i1 + 1))
    return sorted(eps)

def _get_term_reason(df: pd.DataFrame) -> str:
    if "term_reason" not in df.columns or df.empty:
        return "?"
    try:
        return str(df["term_reason"].iloc[-1])
    except Exception:
        return "?"


# ---------------------------
# Plot helpers (labels consistent with paper notation)
# ---------------------------

LBL_T = r"$t\ [\mathrm{s}]$"
LBL_X = r"$x\ [\mathrm{m}]$"
LBL_Y = r"$y\ [\mathrm{m}]$"

LBL_E_FORM_TRUE = r"$e_{\mathrm{form,true}}\ [\mathrm{m}]$"
LBL_SIGMA_MAX = r"$\sigma_{\max}\ [\mathrm{m}]$"

LBL_LAMMIN_JW = r"$\lambda_{\min}(\mathbf{J}_w)$"
LBL_VBAR_PERP = r"$\overline{v}_{\perp}\ [\mathrm{m/s}]$"

LBL_VF = r"$v_F\ [\mathrm{m/s}]$"
LBL_PSI_F = r"$\psi_F\ [^\circ]$"
LBL_OMEGA_F_DEG = r"$\omega_F\ [^\circ/\mathrm{s}]$"
LBL_OMEGA_F_RAD = r"$\omega_F\ [\mathrm{rad/s}]$"


def _apply_common_timeseries_style(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(LBL_T)

def _controller_legend(ax: plt.Axes, *, ncol: int = 2, loc: str = "best", fontsize: int = 9, title: Optional[str] = None) -> None:
    handles = [Line2D([0], [0], color=CTRL_COLOR[c], lw=2) for c in CTRL_ORDER]
    labels = [CTRL_LABEL[c] for c in CTRL_ORDER]
    ax.legend(handles, labels, loc=loc, ncol=ncol, fontsize=fontsize, title=title)


# ---------------------------
# Panel builders (each returns a Figure)
# ---------------------------

def _panel_trajectories(dfs: Dict[str, pd.DataFrame], *, title: Optional[str] = None) -> plt.Figure:
    df_ref = dfs["rl"]
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 5.2))

    ax.plot(df_ref["pL_x"], df_ref["pL_y"], color="k", lw=1.6, label=r"$\mathbf{p}_L$")
    ax.plot(df_ref["pFdes_x"], df_ref["pFdes_y"], color="0.5", lw=1.2, ls="--", label=r"$\mathbf{p}_{F,\mathrm{des}}$")

    for ctrl in CTRL_ORDER:
        df = dfs[ctrl]
        ax.plot(df["pF_x"], df["pF_y"], color=CTRL_COLOR[ctrl], lw=1.9, label=rf"$\mathbf{{p}}_F$ ({CTRL_LABEL[ctrl]})")
        # start/end
        ax.scatter([df["pF_x"].iloc[0]], [df["pF_y"].iloc[0]], color=CTRL_COLOR[ctrl], s=18, marker="o", alpha=0.9)
        ax.scatter([df["pF_x"].iloc[-1]], [df["pF_y"].iloc[-1]], color=CTRL_COLOR[ctrl], s=22, marker="x", alpha=0.9)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(LBL_X)
    ax.set_ylabel(LBL_Y)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig

def _panel_true_error(dfs: Dict[str, pd.DataFrame], *, title: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.6))
    for ctrl in CTRL_ORDER:
        df = dfs[ctrl]
        ax.plot(df["t"], df["err_true_form"], color=CTRL_COLOR[ctrl], lw=1.8, label=CTRL_LABEL[ctrl])
    _apply_common_timeseries_style(ax)
    ax.set_ylabel(LBL_E_FORM_TRUE)
    if title:
        ax.set_title(title)
    _controller_legend(ax, ncol=2, loc="best", fontsize=9)
    fig.tight_layout()
    return fig

def _panel_uncertainty(dfs: Dict[str, pd.DataFrame], *, title: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.6))
    for ctrl in CTRL_ORDER:
        df = dfs[ctrl]
        ax.plot(df["t"], df["std_max"], color=CTRL_COLOR[ctrl], lw=1.8, label=CTRL_LABEL[ctrl])
    _apply_common_timeseries_style(ax)
    ax.set_ylabel(LBL_SIGMA_MAX)
    if title:
        ax.set_title(title)
    _controller_legend(ax, ncol=2, loc="best", fontsize=9)
    fig.tight_layout()
    return fig

def _panel_info_excitation(
    dfs: Dict[str, pd.DataFrame],
    *,
    title: Optional[str] = None,
    fim_floor: float = 1e-16,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.6))
    ax2 = ax.twinx()

    # left: lambda_min(J_w) (solid, log)
    for ctrl in CTRL_ORDER:
        df = dfs[ctrl]
        y = pd.to_numeric(df["fim_win_eig_min"], errors="coerce").to_numpy(dtype=float)
        y = np.where(np.isfinite(y), np.maximum(float(fim_floor), y), np.nan)
        ax.plot(df["t"], y, color=CTRL_COLOR[ctrl], lw=1.7, ls="-")

    ax.set_yscale("log")
    _apply_common_timeseries_style(ax)
    ax.set_ylabel(LBL_LAMMIN_JW)

    # right: vbar_perp ~ sens_avg (dashed)
    for ctrl in CTRL_ORDER:
        df = dfs[ctrl]
        y2 = pd.to_numeric(df["sens_avg"], errors="coerce").to_numpy(dtype=float)
        ax2.plot(df["t"], y2, color=CTRL_COLOR[ctrl], lw=1.5, ls="--", alpha=0.75)

    ax2.set_ylabel(LBL_VBAR_PERP)

    if title:
        ax.set_title(title)

    # --- Legends: controller colors + line styles (requested) ---
    handles_ctrl = [Line2D([0], [0], color=CTRL_COLOR[c], lw=2) for c in CTRL_ORDER]
    labels_ctrl = [CTRL_LABEL[c] for c in CTRL_ORDER]
    leg1 = ax.legend(handles_ctrl, labels_ctrl, title="Controller", loc="upper left", fontsize=8)
    ax.add_artist(leg1)

    handles_style = [
        Line2D([0], [0], color="k", lw=2, ls="-"),
        Line2D([0], [0], color="k", lw=2, ls="--"),
    ]
    labels_style = [
        r"$\lambda_{\min}(\mathbf{J}_w)$ (left)",
        r"$\overline{v}_{\perp}$ (right)",
    ]
    ax.legend(handles_style, labels_style, title="Line style", loc="upper right", fontsize=8)

    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig

def _panel_speed(dfs: Dict[str, pd.DataFrame], *, title: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.6))
    for ctrl in CTRL_ORDER:
        df = dfs[ctrl]
        ax.plot(df["t"], df["speed_F"], color=CTRL_COLOR[ctrl], lw=1.8)
    _apply_common_timeseries_style(ax)
    ax.set_ylabel(LBL_VF)
    if title:
        ax.set_title(title)
    _controller_legend(ax, ncol=2, loc="best", fontsize=9)
    fig.tight_layout()
    return fig

def _panel_course(dfs: Dict[str, pd.DataFrame], *, title: Optional[str] = None, course_mode: str = "unwrap") -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.6))
    key = "psi_F_deg_unwrap" if str(course_mode).lower().startswith("un") else "psi_F_deg_wrap"
    for ctrl in CTRL_ORDER:
        df = dfs[ctrl]
        ax.plot(df["t"], df[key], color=CTRL_COLOR[ctrl], lw=1.8)
    _apply_common_timeseries_style(ax)
    ax.set_ylabel(LBL_PSI_F)
    if title:
        ax.set_title(title)
    _controller_legend(ax, ncol=2, loc="best", fontsize=9)
    fig.tight_layout()
    return fig

def _panel_omega(dfs: Dict[str, pd.DataFrame], *, title: Optional[str] = None, omega_unit: str = "deg_s") -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.6))
    ylab = LBL_OMEGA_F_DEG
    for ctrl in CTRL_ORDER:
        df = dfs[ctrl]
        omega = pd.to_numeric(df["omega_F_deg_s"], errors="coerce").to_numpy(dtype=float)
        if str(omega_unit).lower().startswith("rad"):
            omega = np.radians(omega)
            ylab = LBL_OMEGA_F_RAD
        ax.plot(df["t"], omega, color=CTRL_COLOR[ctrl], lw=1.8)
    _apply_common_timeseries_style(ax)
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)
    _controller_legend(ax, ncol=2, loc="best", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------
# Combined episode figure (paper / extended)
# ---------------------------

def build_episode_figure(
    *,
    paths: Dict[str, Path],
    episode: Optional[int],
    layout: str = "paper",
    title_mode: str = "paper",
    course_mode: str = "unwrap",
    omega_unit: str = "deg_s",
) -> plt.Figure:
    """Create the combined multi-panel figure for one episode."""
    dfs: Dict[str, pd.DataFrame] = {c: _enrich(_load_trace(paths[c])) for c in CTRL_ORDER}

    layout = str(layout).lower()
    if layout not in ("paper", "extended"):
        raise ValueError("layout must be 'paper' or 'extended'")

    if layout == "paper":
        fig, axs = plt.subplots(2, 2, figsize=(11.2, 7.2))
    else:
        fig, axs = plt.subplots(4, 2, figsize=(11.2, 12.2))

    # ---- (a) Trajectories
    ax = axs[0, 0]
    df_ref = dfs["rl"]
    ax.plot(df_ref["pL_x"], df_ref["pL_y"], color="k", lw=1.6, label=r"$\mathbf{p}_L$")
    ax.plot(df_ref["pFdes_x"], df_ref["pFdes_y"], color="0.5", lw=1.2, ls="--", label=r"$\mathbf{p}_{F,\mathrm{des}}$")
    for ctrl in CTRL_ORDER:
        df = dfs[ctrl]
        ax.plot(df["pF_x"], df["pF_y"], color=CTRL_COLOR[ctrl], lw=1.9, label=rf"$\mathbf{{p}}_F$ ({CTRL_LABEL[ctrl]})")
        ax.scatter([df["pF_x"].iloc[0]], [df["pF_y"].iloc[0]], color=CTRL_COLOR[ctrl], s=18, marker="o", alpha=0.9)
        ax.scatter([df["pF_x"].iloc[-1]], [df["pF_y"].iloc[-1]], color=CTRL_COLOR[ctrl], s=22, marker="x", alpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(LBL_X)
    ax.set_ylabel(LBL_Y)
    ax.set_title("Trajectories")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    # ---- (b) True formation error
    ax = axs[0, 1]
    for ctrl in CTRL_ORDER:
        df = dfs[ctrl]
        ax.plot(df["t"], df["err_true_form"], color=CTRL_COLOR[ctrl], lw=1.8, label=CTRL_LABEL[ctrl])
    _apply_common_timeseries_style(ax)
    ax.set_ylabel(LBL_E_FORM_TRUE)
    ax.set_title("True formation error")
    ax.legend(loc="best", fontsize=9, ncol=2)

    # ---- (c) Uncertainty sigma_max
    ax = axs[1, 0]
    for ctrl in CTRL_ORDER:
        df = dfs[ctrl]
        ax.plot(df["t"], df["std_max"], color=CTRL_COLOR[ctrl], lw=1.8, label=CTRL_LABEL[ctrl])
    _apply_common_timeseries_style(ax)
    ax.set_ylabel(LBL_SIGMA_MAX)
    ax.set_title(r"Estimator uncertainty $\sigma_{\max}$")
    ax.legend(loc="best", fontsize=9, ncol=2)

    # ---- (d) Information and excitation
    ax = axs[1, 1]
    ax2 = ax.twinx()

    # solid: lambda_min(J_w)
    for ctrl in CTRL_ORDER:
        df = dfs[ctrl]
        y = pd.to_numeric(df["fim_win_eig_min"], errors="coerce").to_numpy(dtype=float)
        y = np.where(np.isfinite(y), np.maximum(1e-16, y), np.nan)
        ax.plot(df["t"], y, color=CTRL_COLOR[ctrl], lw=1.7, ls="-")
    ax.set_yscale("log")
    ax.set_xlabel(LBL_T)
    ax.set_ylabel(LBL_LAMMIN_JW)
    ax.grid(True, alpha=0.25)

    # dashed: vbar_perp ~ sens_avg
    for ctrl in CTRL_ORDER:
        df = dfs[ctrl]
        y2 = pd.to_numeric(df["sens_avg"], errors="coerce").to_numpy(dtype=float)
        ax2.plot(df["t"], y2, color=CTRL_COLOR[ctrl], lw=1.5, ls="--", alpha=0.75)
    ax2.set_ylabel(LBL_VBAR_PERP)

    ax.set_title("Information and excitation")

    # Legends (controller + line style)
    handles_ctrl = [Line2D([0], [0], color=CTRL_COLOR[c], lw=2) for c in CTRL_ORDER]
    labels_ctrl = [CTRL_LABEL[c] for c in CTRL_ORDER]
    leg1 = ax.legend(handles_ctrl, labels_ctrl, title="Controller", loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    handles_style = [Line2D([0], [0], color="k", lw=2, ls="-"), Line2D([0], [0], color="k", lw=2, ls="--")]
    labels_style = [r"$\lambda_{\min}(\mathbf{J}_w)$ (left)", r"$\overline{v}_{\perp}$ (right)"]
    ax.legend(handles_style, labels_style, title="Line style", loc="upper right", fontsize=8)

    if layout == "extended":
        # ---- (e) speed
        ax = axs[2, 0]
        for ctrl in CTRL_ORDER:
            df = dfs[ctrl]
            ax.plot(df["t"], df["speed_F"], color=CTRL_COLOR[ctrl], lw=1.8)
        _apply_common_timeseries_style(ax)
        ax.set_ylabel(LBL_VF)
        ax.set_title("Follower speed")

        # ---- (f) course
        ax = axs[2, 1]
        key = "psi_F_deg_unwrap" if str(course_mode).lower().startswith("un") else "psi_F_deg_wrap"
        for ctrl in CTRL_ORDER:
            df = dfs[ctrl]
            ax.plot(df["t"], df[key], color=CTRL_COLOR[ctrl], lw=1.8)
        _apply_common_timeseries_style(ax)
        ax.set_ylabel(LBL_PSI_F)
        ax.set_title("Follower course")

        # ---- (g) omega
        ax = axs[3, 0]
        ylab = LBL_OMEGA_F_DEG
        for ctrl in CTRL_ORDER:
            df = dfs[ctrl]
            omega = pd.to_numeric(df["omega_F_deg_s"], errors="coerce").to_numpy(dtype=float)
            if str(omega_unit).lower().startswith("rad"):
                omega = np.radians(omega)
                ylab = LBL_OMEGA_F_RAD
            ax.plot(df["t"], omega, color=CTRL_COLOR[ctrl], lw=1.8)
        _apply_common_timeseries_style(ax)
        ax.set_ylabel(ylab)
        ax.set_title("Follower angular rate")

        # ---- (h) empty / reserved
        axs[3, 1].axis("off")

        # Add a legend for motion panels (use controller legend once)
        axs[2, 0].legend(
            [Line2D([0], [0], color=CTRL_COLOR[c], lw=2) for c in CTRL_ORDER],
            [CTRL_LABEL[c] for c in CTRL_ORDER],
            loc="best",
            fontsize=9,
            ncol=2,
        )

    # Suptitle for review (term reasons)
    if str(title_mode).lower() == "review":
        bits = []
        for ctrl in CTRL_ORDER:
            bits.append(f"{CTRL_LABEL[ctrl]}:{_get_term_reason(dfs[ctrl])}")
        ep_txt = f"ep{int(episode):03d} | " if episode is not None else ""
        fig.suptitle(ep_txt + ", ".join(bits), fontsize=10, y=0.995)

    fig.tight_layout(rect=(0, 0, 1, 0.97) if str(title_mode).lower() == "review" else None)
    return fig


# ---------------------------
# Export panels (NEW)
# ---------------------------

def export_episode_panels(
    *,
    paths: Dict[str, Path],
    episode: Optional[int],
    export_root: Path,
    dpi: int = 220,
    title_mode: str = "paper",
    course_mode: str = "unwrap",
    omega_unit: str = "deg_s",
) -> Path:
    """Export each panel as separate PNG into export_root/epXXX/ (or export_root/episode_unknown/)."""
    dfs: Dict[str, pd.DataFrame] = {c: _enrich(_load_trace(paths[c])) for c in CTRL_ORDER}

    ep_dirname = f"ep{int(episode):03d}" if episode is not None else "episode_unknown"
    out_dir = Path(export_root) / ep_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    # titles: keep short in review, omit in paper
    def t(txt: str) -> Optional[str]:
        if str(title_mode).lower() == "paper":
            return None
        return txt

    figs = [
        ("a_trajectories.png", _panel_trajectories(dfs, title=t("Trajectories"))),
        ("b_true_formation_error.png", _panel_true_error(dfs, title=t("True formation error"))),
        ("c_estimator_uncertainty.png", _panel_uncertainty(dfs, title=t(r"Estimator uncertainty $\sigma_{\max}$"))),
        ("d_information_excitation.png", _panel_info_excitation(dfs, title=t("Information and excitation"))),
        ("e_follower_speed.png", _panel_speed(dfs, title=t("Follower speed"))),
        ("f_follower_course.png", _panel_course(dfs, title=t("Follower course"), course_mode=course_mode)),
        ("g_follower_omega.png", _panel_omega(dfs, title=t("Follower angular rate"), omega_unit=omega_unit)),
    ]

    for fname, fig in figs:
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=int(dpi))
        plt.close(fig)

    # optional small text summary (useful when browsing folders)
    if str(title_mode).lower() == "review":
        bits = [f"{CTRL_LABEL[c]}:{_get_term_reason(dfs[c])}" for c in CTRL_ORDER]
        (out_dir / "summary.txt").write_text(", ".join(bits) + "\n", encoding="utf-8")

    return out_dir


# ---------------------------
# PDF-all (review)
# ---------------------------

def make_pdf_all(
    *,
    root: Path,
    out_pdf: Path,
    episodes: Optional[List[int]] = None,
    layout: str = "paper",
    dpi: int = 220,
    title_mode: str = "review",
    course_mode: str = "unwrap",
    omega_unit: str = "deg_s",
) -> None:
    root = Path(root)
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    common_eps = _available_matched_episodes(root)
    if not common_eps:
        raise FileNotFoundError(f"No common episodes found under root={root}")

    eps_to_do = common_eps if episodes is None else [e for e in episodes if e in set(common_eps)]
    if not eps_to_do:
        raise FileNotFoundError("After filtering, no episodes remain to plot.")

    with PdfPages(out_pdf) as pdf:
        info = pdf.infodict()
        info["Title"] = "Figure 11 example trajectories (all episodes)"
        info["Subject"] = "RL vs baselines (matched seeds)"

        for i, ep in enumerate(eps_to_do, 1):
            paths = _resolve_from_root(root, int(ep))
            fig = build_episode_figure(
                paths=paths,
                episode=int(ep),
                layout=layout,
                title_mode=title_mode,
                course_mode=course_mode,
                omega_unit=omega_unit,
            )
            pdf.savefig(fig, dpi=int(dpi))
            plt.close(fig)

            if (i == 1) or (i == len(eps_to_do)) or (i % 25 == 0):
                print(f"[PDF] pages: {i}/{len(eps_to_do)}")

    print("[PDF] Done:", out_pdf)


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    # mode A: root+episode
    ap.add_argument("--root", type=str, default=None, help="Root directory containing controller subfolders.")
    ap.add_argument("--episode", type=int, default=None, help="Episode index, e.g. 176 for ep176.")

    # mode B: explicit files
    ap.add_argument("--rl", type=str, default=None, help="Path to rl/traces/epXXX.csv")
    ap.add_argument("--pid", type=str, default=None, help="Path to baseline_pid/traces/epXXX.csv")
    ap.add_argument("--pid-exc", dest="pid_exc", type=str, default=None, help="Path to baseline_pid_exc/traces/epXXX.csv")
    ap.add_argument("--random", type=str, default=None, help="Path to baseline_random/traces/epXXX.csv")

    # output modes
    ap.add_argument("--out", type=str, default=None, help="Output PNG path for combined figure (single-episode mode).")
    ap.add_argument("--save-pdf", action="store_true", help="Also save single-episode PDF next to --out.")
    ap.add_argument("--export-panels", action="store_true", help="Export each panel as separate PNG into --export-root/epXXX/.")
    ap.add_argument("--export-root", type=str, default=".", help="Root folder for --export-panels output (default: current dir).")

    # review PDF mode
    ap.add_argument("--pdf-all", action="store_true", help="Generate a multi-page PDF for many episodes (requires --root).")
    ap.add_argument("--pdf-out", type=str, default=None, help="Output PDF path for --pdf-all (default: fig11_all_episodes.pdf)")
    ap.add_argument("--episodes", type=str, default="all", help="Episode selection for --pdf-all: 'all', '1-300', '1,2,5-10', ...")

    # styling/options
    ap.add_argument("--layout", choices=["paper", "extended"], default="paper")
    ap.add_argument("--title-mode", choices=["paper", "review"], default=None)
    ap.add_argument("--course-mode", choices=["wrap", "unwrap"], default="unwrap")
    ap.add_argument("--omega-unit", choices=["deg_s", "rad_s"], default="deg_s")
    ap.add_argument("--dpi", type=int, default=220)

    args = ap.parse_args()

    # Decide title mode default: review for pdf-all, paper otherwise
    title_mode = args.title_mode
    if title_mode is None:
        title_mode = "review" if bool(args.pdf_all) else "paper"

    # -----------------------------
    # MULTI-PAGE PDF MODE
    # -----------------------------
    if args.pdf_all:
        if not args.root:
            raise SystemExit("--pdf-all requires --root")

        out_pdf = Path(args.pdf_out) if args.pdf_out else Path("fig11_all_episodes.pdf")
        eps_spec = _parse_episode_spec(args.episodes)

        make_pdf_all(
            root=Path(args.root),
            out_pdf=out_pdf,
            episodes=eps_spec,
            layout=str(args.layout),
            dpi=int(args.dpi),
            title_mode=str(title_mode),
            course_mode=str(args.course_mode),
            omega_unit=str(args.omega_unit),
        )
        return

    # -----------------------------
    # SINGLE EPISODE MODE
    # -----------------------------
    paths: Dict[str, Path]
    ep_for_naming: Optional[int] = None

    if args.root and (args.episode is not None):
        ep_for_naming = int(args.episode)
        paths = _resolve_from_root(Path(args.root), ep_for_naming)
    else:
        if not (args.rl and args.pid and args.pid_exc and args.random):
            raise SystemExit(
                "Provide either (--root AND --episode) OR all explicit paths: "
                "--rl --pid --pid-exc --random"
            )
        paths = {
            "rl": Path(args.rl),
            "baseline_pid": Path(args.pid),
            "baseline_pid_exc": Path(args.pid_exc),
            "baseline_random": Path(args.random),
        }

    did_any = False

    # (A) export panels
    if bool(args.export_panels):
        out_dir = export_episode_panels(
            paths=paths,
            episode=ep_for_naming,
            export_root=Path(args.export_root),
            dpi=int(args.dpi),
            title_mode=str(title_mode),
            course_mode=str(args.course_mode),
            omega_unit=str(args.omega_unit),
        )
        print("[OK] Panels exported to:", out_dir)
        did_any = True

    # (B) combined figure
    if args.out is not None:
        out_png = Path(args.out)
        out_png.parent.mkdir(parents=True, exist_ok=True)

        fig = build_episode_figure(
            paths=paths,
            episode=ep_for_naming,
            layout=str(args.layout),
            title_mode=str(title_mode),
            course_mode=str(args.course_mode),
            omega_unit=str(args.omega_unit),
        )
        fig.savefig(out_png, dpi=int(args.dpi))
        if bool(args.save_pdf):
            fig.savefig(out_png.with_suffix(".pdf"))
        plt.close(fig)

        print("[OK] Saved combined figure:", out_png)
        if bool(args.save_pdf):
            print("[OK] Saved combined PDF:", out_png.with_suffix(".pdf"))
        did_any = True

    if not did_any:
        raise SystemExit("Nothing to do: set --out and/or --export-panels (or use --pdf-all).")


if __name__ == "__main__":
    main()

