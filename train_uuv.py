# train_uuv.py
# -*- coding: utf-8 -*-
"""
Trening PPO dla UUVRelPosEnv z Stable-Baselines3 (Gymnasium backend).

Uwaga:
  • Wymaga:
      pip install gymnasium stable-baselines3[extra] numpy
"""

import argparse
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from uuv_env import UUVRelPosEnv


def make_env(use_range_anchors: bool, render_mode: Optional[str]):
    def _thunk():
        return UUVRelPosEnv(use_range_anchors=use_range_anchors, render_mode=render_mode)
    return _thunk


def train(use_range_anchors: bool, total_timesteps: int, n_envs: int, render_mode: Optional[str]) -> None:
    tag = "with_anchors" if use_range_anchors else "without_anchors"
    print(f"[INFO] Start training PPO ({tag}), total_timesteps={total_timesteps}, n_envs={n_envs}")

    vec_env = make_vec_env(
        make_env(use_range_anchors, render_mode),
        n_envs=n_envs,
    )
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    model.learn(total_timesteps=total_timesteps)

    model_path = f"ppo_uuv_{tag}"
    vecnorm_path = f"vecnorm_uuv_{tag}.pkl"

    model.save(model_path)
    vec_env.save(vecnorm_path)

    print(f"[INFO] Zapisano model do: {model_path}")
    print(f"[INFO] Zapisano VecNormalize do: {vecnorm_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-anchors", action="store_true",
                        help="Użyj wariantu bez kotwic (use_range_anchors=False).")
    parser.add_argument("--timesteps", type=int, default=3_000_000,
                        help="Liczba kroków treningu (domyślnie 3e6).")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Liczba równoległych środowisk (domyślnie 8).")
    parser.add_argument("--render-mode", choices=["none", "human", "rgb_array", "pygame"], default="none",
                        help="Tryb renderowania środowiska (domyślnie none - bez renderu).")
    args = parser.parse_args()

    use_range_anchors = not args.no_anchors
    render_mode = None if args.render_mode == "none" else args.render_mode
    train(use_range_anchors, args.timesteps, args.n_envs, render_mode)


if __name__ == "__main__":
    main()
