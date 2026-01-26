# -*- coding: utf-8 -*-
"""
mapy.py
================================

Szybki start (copy-paste):
  pip install numpy matplotlib
  python uuv_make_fim_maps_nav_triptych_v2.py \
    --out-dir figs \
    --maneuvers straight turn sine \
    --horizon 30 \
    --meas-period 1.0 \
    --sigma-s 0.015 \
    --vrel-mag 1.0 \
    --turn-rate-deg-s 8.0 \
    --sine-amp-deg 35 \
    --sine-freq-hz 0.05


Mapy informacyjne (WARUNEK B) dla Doppler-only (range-rate):
  - FIM (okienkowa) -> log10 lambda_min
  - CRLB (okienkowa) -> log10 trace(CRLB)
  - used samples (gating)

NOWE (zgodnie z prośbą):
  1) Polar w "stylu nawigacyjnym":
       - 0° u góry (North)
       - 90° po prawej (East)
       - kąty rosną zgodnie z ruchem wskazówek (clockwise)
     (czyli jak bearing/azymut na mapie).
  2) Triptych: dla każdej metryki generujemy JEDNĄ figurę z 3 panelami
     (straight / turn / sine) na wspólnej skali kolorów.

Spójność z Twoim układem odniesienia:
  - x w prawo (E), y w górę (N)
  - r0 = [rho*cos(phi), rho*sin(phi)] gdzie phi jest kątem matematycznym od osi +x (E), CCW.
  - Do rysowania w stylu nawigacyjnym przemapowujemy ten kąt na bearing:
        bearing_deg = (90 - phi_deg) mod 360
    aby:
      phi=0° (E) -> bearing=90° (prawo)
      phi=90° (N) -> bearing=0° (góra)
      phi=180° (W) -> bearing=270° (lewo)

Wymagania:
  pip install numpy matplotlib

Przykład (Twój wariant):
  python mapy.py \
    --out-dir figs \
    --maneuvers straight turn sine \
    --horizon 30 \
    --meas-period 1.0 \
    --sigma-s 0.015 \
    --vrel-mag 1.0 \
    --turn-rate-deg-s 8.0 \
    --sine-amp-deg 35 \
    --sine-freq-hz 0.05

Wyjście:
  figs/obsmap_triptych_fim_lammin_T30_navpolar.png (+ pdf)
  figs/obsmap_triptych_crlb_trace_T30_navpolar.png (+ pdf)
  figs/obsmap_triptych_used_samples_T30_navpolar.png (+ pdf)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, Callable, List

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Math helpers
# -------------------------

def deg2rad(a_deg: float) -> float:
    return float(a_deg) * math.pi / 180.0


def rot2(theta_rad: float) -> np.ndarray:
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def bearing_from_phi_deg(phi_deg_math: np.ndarray) -> np.ndarray:
    """
    Przekształcenie kąta matematycznego (0°=E, CCW) na bearing nawigacyjny
    (0°=N, CW).
    """
    phi_deg_math = np.asarray(phi_deg_math, dtype=float)
    return (90.0 - phi_deg_math) % 360.0


# -------------------------
# Doppler Jacobian
# -------------------------

def doppler_jacobian_H(r: np.ndarray, v_rel: np.ndarray, rho_min: float = 1e-9) -> Tuple[np.ndarray, float, float]:
    """
    Model Dopplera/range-rate:
      s = - rhat · v_rel + noise

    Jacobian:
      H = ∂s/∂r = -(v_rel - (rhat^T v_rel) rhat)^T / rho

    Zwraca:
      H : (2,)
      rho : ||r||
      v_perp : składowa poprzeczna v_rel względem rhat (proxy pobudzenia)
    """
    r = np.asarray(r, dtype=float).reshape(2,)
    v_rel = np.asarray(v_rel, dtype=float).reshape(2,)

    rho = float(np.linalg.norm(r))
    if rho <= rho_min:
        return np.zeros(2, dtype=float), rho, 0.0

    rhat = r / rho
    proj = float(np.dot(rhat, v_rel))
    vnorm = float(np.linalg.norm(v_rel))
    v_perp = math.sqrt(max(0.0, vnorm * vnorm - proj * proj))

    H = -(v_rel - proj * rhat) / rho
    return H.astype(float), rho, float(v_perp)


# -------------------------
# v_rel(t) templates
# -------------------------

def vrel_template_factory(
    maneuver: str,
    vrel_mag: float,
    turn_rate_deg_s: float,
    sine_amp_deg: float,
    sine_freq_hz: float,
) -> Callable[[float], np.ndarray]:
    """
    Zwraca funkcję v_rel(t) (2D) dla wzorca manewru.

    Konwencja: v_rel(0) = [vrel_mag, 0] (czyli +x / East).
    """
    v0 = np.array([float(vrel_mag), 0.0], dtype=float)
    m = maneuver.lower().strip()

    if m == "straight":
        def vrel(t: float) -> np.ndarray:
            return v0
        return vrel

    if m == "turn":
        omega = deg2rad(float(turn_rate_deg_s))
        def vrel(t: float) -> np.ndarray:
            return rot2(omega * float(t)) @ v0
        return vrel

    if m == "sine":
        amp = deg2rad(float(sine_amp_deg))
        freq = float(sine_freq_hz)
        def vrel(t: float) -> np.ndarray:
            theta = amp * math.sin(2.0 * math.pi * freq * float(t))
            return rot2(theta) @ v0
        return vrel

    raise ValueError(f"Unknown maneuver: {maneuver}. Use: straight, turn, sine.")


# -------------------------
# Windowed FIM/CRLB
# -------------------------

def fim_window_for_initial_condition(
    r0: np.ndarray,
    *,
    horizon_s: float,
    meas_period_s: float,
    sigma_s: float,
    vrel_of_t: Callable[[float], np.ndarray],
    rho_min_gate: float,
    v_min_gate: float,
    v_perp_min_gate: float,
    eps_reg: float,
) -> Dict[str, float]:
    """
    Okienkowa FIM (T=horizon) i CRLB dla jednego startu r0.

    Dyskretyzacja:
      - próbki co Ts (meas_period_s),
      - propagacja r <- r + v_rel*Ts
    """
    r = np.asarray(r0, dtype=float).reshape(2,).copy()
    T = float(horizon_s)
    Ts = float(meas_period_s)
    if Ts <= 0:
        raise ValueError("meas_period_s must be > 0")

    K = int(max(1, round(T / Ts)))
    Rinv = 1.0 / (float(max(1e-12, sigma_s)) ** 2)

    Iw = np.zeros((2, 2), dtype=float)
    used = 0

    for k in range(K):
        t = float(k) * Ts
        v_rel = np.asarray(vrel_of_t(t), dtype=float).reshape(2,)
        H, rho, v_perp = doppler_jacobian_H(r, v_rel)
        vnorm = float(np.linalg.norm(v_rel))

        # gating: spójne z praktycznymi filtrami
        if (rho > float(rho_min_gate)) and (vnorm > float(v_min_gate)) and (v_perp > float(v_perp_min_gate)):
            Iw += Rinv * np.outer(H, H)
            used += 1

        r = r + v_rel * Ts

    eigs = np.linalg.eigvalsh(Iw)
    lam_min = float(max(0.0, eigs[0]))
    lam_max = float(max(0.0, eigs[1]))

    eps = float(max(0.0, eps_reg))
    Cr = np.linalg.inv(Iw + eps * np.eye(2))
    crlb_trace = float(np.trace(Cr))

    return {
        "used_meas": float(used),
        "lam_min": lam_min,
        "lam_max": lam_max,
        "crlb_trace": crlb_trace,
    }


def make_maps(
    *,
    rho_min: float,
    rho_max: float,
    n_rho: int,
    phi_min_deg: float,
    phi_max_deg: float,
    n_phi: int,
    horizon_s: float,
    meas_period_s: float,
    sigma_s: float,
    vrel_mag: float,
    maneuver: str,
    turn_rate_deg_s: float,
    sine_amp_deg: float,
    sine_freq_hz: float,
    rho_min_gate: float,
    v_min_gate: float,
    v_perp_min_gate: float,
    eps_reg: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Zwraca:
      rho_grid (Nrho,)
      phi_grid_deg (Nphi,)  [kąt matematyczny: 0=E, CCW]
      maps: lam_min, crlb_trace, used_meas  (Nrho, Nphi)
    """
    rho_grid = np.linspace(float(rho_min), float(rho_max), int(n_rho))
    phi_grid = np.linspace(float(phi_min_deg), float(phi_max_deg), int(n_phi))

    vrel_of_t = vrel_template_factory(
        maneuver=maneuver,
        vrel_mag=float(vrel_mag),
        turn_rate_deg_s=float(turn_rate_deg_s),
        sine_amp_deg=float(sine_amp_deg),
        sine_freq_hz=float(sine_freq_hz),
    )

    lam_min_map = np.zeros((rho_grid.size, phi_grid.size), dtype=float)
    crlb_map = np.zeros_like(lam_min_map, dtype=float)
    used_map = np.zeros_like(lam_min_map, dtype=float)

    for i, rho in enumerate(rho_grid):
        for j, phi_deg in enumerate(phi_grid):
            phi = deg2rad(phi_deg)
            r0 = np.array([rho * math.cos(phi), rho * math.sin(phi)], dtype=float)

            m = fim_window_for_initial_condition(
                r0,
                horizon_s=float(horizon_s),
                meas_period_s=float(meas_period_s),
                sigma_s=float(sigma_s),
                vrel_of_t=vrel_of_t,
                rho_min_gate=float(rho_min_gate),
                v_min_gate=float(v_min_gate),
                v_perp_min_gate=float(v_perp_min_gate),
                eps_reg=float(eps_reg),
            )
            lam_min_map[i, j] = float(m["lam_min"])
            crlb_map[i, j] = float(m["crlb_trace"])
            used_map[i, j] = float(m["used_meas"])

    return rho_grid, phi_grid, {"lam_min": lam_min_map, "crlb_trace": crlb_map, "used_meas": used_map}


# -------------------------
# Plot helpers (triptych)
# -------------------------

def _safe_log10(x: np.ndarray, floor: float) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(x, dtype=float), float(floor)))


def _centers_to_edges(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 2:
        return np.array([x[0] - 0.5, x[0] + 0.5], dtype=float)
    dx = np.diff(x)
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = x[:-1] + 0.5 * dx
    edges[0] = x[0] - 0.5 * dx[0]
    edges[-1] = x[-1] + 0.5 * dx[-1]
    return edges


def _prepare_polar_bins_nav(phi_grid_deg_math: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Przygotowanie danych do pcolormesh w układzie polarnym (NAWIGACYJNYM).

    Wejście:
      phi_grid_deg_math: kąty matematyczne (0=E, CCW), np. [-180..180]
      Z: (Nrho, Nphi)

    Wyjście:
      theta_edges_rad: krawędzie binów theta w radianach, w skali bearingów [0..2π)
      Z_sorted: Z posortowane po rosnącym bearingu

    Obsługa duplikatu:
      jeśli zakres pokrywa 360° i ostatnia kolumna to duplikat (np. -180 i +180),
      obcinamy ostatnią kolumnę.
    """
    phi = np.asarray(phi_grid_deg_math, dtype=float).reshape(-1)
    Zm = np.asarray(Z, dtype=float)
    if phi.size != Zm.shape[1]:
        raise ValueError("phi_grid_deg length must match Z.shape[1]")

    # full-circle with duplicated endpoint?
    if phi.size > 2 and abs((phi[-1] - phi[0]) - 360.0) < 1e-6:
        phi = phi[:-1]
        Zm = Zm[:, :-1]

    bearing = bearing_from_phi_deg(phi)
    order = np.argsort(bearing)
    bearing_sorted = bearing[order]
    Z_sorted = Zm[:, order]

    # If it looks like a full 0..360 coverage -> use clean edges 0..360
    if bearing_sorted.size > 4:
        step = float(np.median(np.diff(bearing_sorted)))
        step = max(step, 1e-9)
        full = abs((bearing_sorted[-1] - bearing_sorted[0] + step) - 360.0) < 1e-3
    else:
        full = False

    if full:
        theta_edges = np.deg2rad(np.linspace(0.0, 360.0, bearing_sorted.size + 1))
    else:
        theta_edges = np.deg2rad(_centers_to_edges(bearing_sorted))

    return theta_edges, Z_sorted


def plot_triptych_polar_nav(
    results: Dict[str, Dict[str, np.ndarray]],
    maneuvers: List[str],
    *,
    Z_key: str,
    vmin: float,
    vmax: float,
    cbar_label: str,
    suptitle: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    """
    Rysuje 1xN paneli (polar nawigacyjny) dla danej metryki.
    """
    n = len(maneuvers)
    fig, axs = plt.subplots(
        1, n,
        subplot_kw={"projection": "polar"},
        figsize=(4.6 * n, 4.8),
        constrained_layout=True,
    )

    if n == 1:
        axs = [axs]

    mappable = None
    for ax, man in zip(axs, maneuvers):
        rho = results[man]["rho"]
        phi = results[man]["phi"]
        Z = results[man][Z_key]

        theta_edges, Zs = _prepare_polar_bins_nav(phi, Z)
        rho_edges = _centers_to_edges(rho)

        Theta, Rho = np.meshgrid(theta_edges, rho_edges)

        pc = ax.pcolormesh(
            Theta, Rho, Zs,
            shading="auto",
            vmin=float(vmin),
            vmax=float(vmax),
        )
        mappable = pc

        # NAVIGATION STYLE
        ax.set_theta_zero_location("N")  # 0° at North (up)
        ax.set_theta_direction(-1)       # clockwise increasing -> 90° at East (right)

        ax.set_rmin(float(rho[0]))
        ax.set_rmax(float(rho[-1]))
        ax.set_title(str(man), pad=14)

        # readable grid
        ax.grid(True)

        # nice cardinal ticks
        ax.set_thetagrids([0, 90, 180, 270], labels=["0°", "90°", "180°", "270°"])

    fig.suptitle(suptitle, y=1.03)

    if mappable is not None:
        cb = fig.colorbar(mappable, ax=axs, shrink=0.85, pad=0.03)
        cb.set_label(cbar_label)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf, dpi=220)
    plt.close(fig)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="figs")

    ap.add_argument("--maneuvers", nargs="+", default=["straight", "turn", "sine"],
                    help="Lista: straight turn sine (kolejność zostaje zachowana)")

    ap.add_argument("--horizon", type=float, default=30.0)
    ap.add_argument("--meas-period", type=float, default=1.0)
    ap.add_argument("--sigma-s", type=float, default=0.015)

    ap.add_argument("--vrel-mag", type=float, default=1.0)

    ap.add_argument("--turn-rate-deg-s", type=float, default=8.0)
    ap.add_argument("--sine-amp-deg", type=float, default=35.0)
    ap.add_argument("--sine-freq-hz", type=float, default=0.05)

    ap.add_argument("--rho-min", type=float, default=25.0)
    ap.add_argument("--rho-max", type=float, default=220.0)
    ap.add_argument("--n-rho", type=int, default=100)

    ap.add_argument("--phi-min-deg", type=float, default=-180.0)
    ap.add_argument("--phi-max-deg", type=float, default=180.0)
    ap.add_argument("--n-phi", type=int, default=181)

    ap.add_argument("--rho-min-gate", type=float, default=20.0)
    ap.add_argument("--v-min-gate", type=float, default=0.05)
    ap.add_argument("--v-perp-min-gate", type=float, default=0.25)

    ap.add_argument("--eps-reg", type=float, default=1e-12)

    # Optional: robust scaling by quantiles (for nicer visuals if extremes dominate)
    ap.add_argument("--robust-scale", action="store_true",
                    help="Zakres kolorów wyznacz z percentyli (1..99) zamiast min..max.")
    ap.add_argument("--q-lo", type=float, default=1.0)
    ap.add_argument("--q-hi", type=float, default=99.0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maneuvers = [str(m).strip() for m in args.maneuvers]
    if not maneuvers:
        raise ValueError("Provide at least one maneuver.")

    # 1) compute maps for each maneuver
    results: Dict[str, Dict[str, np.ndarray]] = {}

    Kmax = int(max(1, round(float(args.horizon) / float(args.meas_period))))

    for man in maneuvers:
        rho, phi, maps = make_maps(
            rho_min=float(args.rho_min),
            rho_max=float(args.rho_max),
            n_rho=int(args.n_rho),
            phi_min_deg=float(args.phi_min_deg),
            phi_max_deg=float(args.phi_max_deg),
            n_phi=int(args.n_phi),
            horizon_s=float(args.horizon),
            meas_period_s=float(args.meas_period),
            sigma_s=float(args.sigma_s),
            vrel_mag=float(args.vrel_mag),
            maneuver=str(man),
            turn_rate_deg_s=float(args.turn_rate_deg_s),
            sine_amp_deg=float(args.sine_amp_deg),
            sine_freq_hz=float(args.sine_freq_hz),
            rho_min_gate=float(args.rho_min_gate),
            v_min_gate=float(args.v_min_gate),
            v_perp_min_gate=float(args.v_perp_min_gate),
            eps_reg=float(args.eps_reg),
        )

        Z_fim = _safe_log10(maps["lam_min"], floor=1e-16)
        Z_crlb = _safe_log10(maps["crlb_trace"], floor=1e-12)
        Z_used = maps["used_meas"]

        results[man] = {
            "rho": rho,
            "phi": phi,
            "Z_fim": Z_fim,
            "Z_crlb": Z_crlb,
            "Z_used": Z_used,
        }

    # 2) shared color scales across maneuvers (per metric)
    def _global_minmax(key: str) -> Tuple[float, float]:
        vals = np.concatenate([results[m][key].ravel() for m in maneuvers], axis=0)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0.0, 1.0
        if args.robust_scale:
            lo = float(np.percentile(vals, float(args.q_lo)))
            hi = float(np.percentile(vals, float(args.q_hi)))
            if hi <= lo:
                return float(np.min(vals)), float(np.max(vals))
            return lo, hi
        return float(np.min(vals)), float(np.max(vals))

    fim_vmin, fim_vmax = _global_minmax("Z_fim")
    crlb_vmin, crlb_vmax = _global_minmax("Z_crlb")
    used_vmin, used_vmax = 0.0, float(Kmax)

    print("[COMMON SCALE]")
    print(f"  FIM  log10(lambda_min): vmin={fim_vmin:.3f}, vmax={fim_vmax:.3f}")
    print(f"  CRLB log10(trace):      vmin={crlb_vmin:.3f}, vmax={crlb_vmax:.3f}")
    print(f"  used samples:           vmin={used_vmin:.1f}, vmax={used_vmax:.1f}")

    # 3) triptychs (one figure per metric)
    T = int(round(float(args.horizon)))

    plot_triptych_polar_nav(
        results, maneuvers,
        Z_key="Z_fim",
        vmin=fim_vmin, vmax=fim_vmax,
        cbar_label=r"$\log_{10}\lambda_{\min}(I_w)$",
        suptitle=f"Okienkowa FIM (T={T}s), Doppler-only — polar (NAV)",
        out_png=out_dir / f"obsmap_triptych_fim_lammin_T{T}_navpolar.png",
        out_pdf=out_dir / f"obsmap_triptych_fim_lammin_T{T}_navpolar.pdf",
    )

    plot_triptych_polar_nav(
        results, maneuvers,
        Z_key="Z_crlb",
        vmin=crlb_vmin, vmax=crlb_vmax,
        cbar_label=r"$\log_{10}\mathrm{tr}(\mathrm{CRLB}_w)$",
        suptitle=f"Okienkowa CRLB (T={T}s), Doppler-only — polar (NAV)",
        out_png=out_dir / f"obsmap_triptych_crlb_trace_T{T}_navpolar.png",
        out_pdf=out_dir / f"obsmap_triptych_crlb_trace_T{T}_navpolar.pdf",
    )

    plot_triptych_polar_nav(
        results, maneuvers,
        Z_key="Z_used",
        vmin=used_vmin, vmax=used_vmax,
        cbar_label="used Doppler samples",
        suptitle=f"Liczba użytych próbek (gating), T={T}s — polar (NAV)",
        out_png=out_dir / f"obsmap_triptych_used_samples_T{T}_navpolar.png",
        out_pdf=out_dir / f"obsmap_triptych_used_samples_T{T}_navpolar.pdf",
    )

    print("[DONE] Saved triptychs to:", out_dir.resolve())


if __name__ == "__main__":
    main()
