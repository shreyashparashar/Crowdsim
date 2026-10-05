"""Fit model parameters so a past event's simulation matches what really happened.

observed.csv: an `hour` column plus any metric columns produced by `run`, e.g.
  posts_total            (mention volume from social listening / Reddit / GDELT counts)
  net_sentiment_posts    (-1..1 average sentiment of mentions)
  aware_<story>          (survey awareness, 0..1)
  mean_opinion, adoption ...
Volume-like columns (posts_*, new_adopters) are compared by SHAPE (best scale factor is
fitted), because a 100k-agent world doesn't have your real audience size.
"""
import copy

import numpy as np
import yaml

from .engine import DEFAULT_MODEL
from .report import read_csv

BOUNDS = {
    "feed_visibility": (0.05, 0.6), "algo_boost": (0.5, 10), "post_rate": (0.03, 0.5),
    "accept_bias": (-2, 2), "share_bias": (-5, -1), "w_emotion": (0.5, 5), "w_skepticism": (0.5, 5),
    "w_confirmation": (0, 3), "w_social_proof": (0, 4), "forget_rate": (0.005, 0.1),
    "stifle_rate": (0, 0.4), "novelty_halflife": (6, 120), "opinion_impact": (0.05, 1.0),
    "negativity_bias": (0, 1.5), "social_influence": (0.005, 0.2), "confidence_bound": (0.1, 1.5),
    "correction_efficacy": (0.1, 1), "adopt_base_rate": (0.0002, 0.01), "a_bias": (-4, 1),
}


def _loss(obs, hist):
    h = obs["hour"].astype(int)
    h = h[h < len(hist["hour"])]
    total = 0.0
    for col, y in obs.items():
        if col == "hour" or col not in hist:
            continue
        y = y[: len(h)]
        s = hist[col][h]
        if col.startswith("posts") or col == "new_adopters":
            scale = (s @ y) / max(s @ s, 1e-9)
            s = s * scale
            norm = max(np.abs(y).max(), 1e-9)
        else:
            norm = 1.0
        total += np.sqrt(np.mean(((s - y) / norm) ** 2))
    return total


def calibrate(a):
    from .__main__ import build_world, load, simulate

    obs = read_csv(a.observed)
    cfg = load(a.scenario, a.set, a.size, int(obs["hour"].max()) + 1)
    pop, net = build_world(cfg)
    names = [p.strip() for p in a.params.split(",") if p.strip()]
    for n in names:
        if n not in BOUNDS:
            raise SystemExit(f"unknown param {n}; choose from {', '.join(BOUNDS)}")
    rng = np.random.default_rng(0)
    base = {**DEFAULT_MODEL, **(cfg.get("model") or {})}
    best = ({n: base[n] for n in names}, None)

    def score(params):
        c = copy.deepcopy(cfg)
        c["model"] = {**(c.get("model") or {}), **params}
        return _loss(obs, simulate(c, pop, net, log=None).history_arrays())

    best = (best[0], score(best[0]))
    print(f"start loss {best[1]:.4f}")
    for i in range(a.trials):
        # first half: explore the whole box; second half: shrink around the best point
        width = 1.0 if i < a.trials // 2 else 0.25 * (1 - (i - a.trials // 2) / max(a.trials // 2, 1)) + 0.05
        cand = {}
        for n in names:
            lo, hi = BOUNDS[n]
            centre = best[0][n]
            cand[n] = float(np.clip(centre + rng.uniform(-1, 1) * width * (hi - lo), lo, hi)) if i >= a.trials // 2 \
                else float(rng.uniform(lo, hi))
        s = score(cand)
        mark = ""
        if s < best[1]:
            best, mark = (cand, s), "  <- best"
        print(f"trial {i + 1:3d}/{a.trials} loss {s:.4f}{mark}")
    cfg["model"] = {**(cfg.get("model") or {}), **{k: round(v, 5) for k, v in best[0].items()}}
    with open(a.out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"best loss {best[1]:.4f}; params {best[0]}\nwrote {a.out} (population size is the calibration size; "
          f"set it back with --size when running)")
