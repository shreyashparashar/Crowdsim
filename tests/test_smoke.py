"""Run: python tests/test_smoke.py  (small world, checks the core mechanics hold)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from crowdsim.__main__ import build_world, load, simulate  # noqa: E402

HERE = os.path.dirname(__file__)


def test_launch():
    cfg = load(os.path.join(HERE, "..", "scenarios", "product_launch.yaml"), size=20_000, hours=120)
    pop, net = build_world(cfg, log=lambda *_: None)
    sim = simulate(cfg, pop, net, log=None)
    h = sim.history_arrays()
    assert len(h["hour"]) == 120
    assert h["aware_launch"][-1] > 0.2, "launch should reach a good share of people"
    assert np.all(np.diff(h["aware_launch"]) >= 0), "awareness never decreases"
    assert np.all(np.abs(sim.opinion) < 1), "opinions stay in (-1, 1)"
    assert h["adoption"][-1] > h["adoption"][0], "someone buys"
    assert h["aware_battery_rumor"][29] == 0 and h["aware_battery_rumor"][-1] > 0, "rumor starts on time"


def test_overrides():
    cfg = load(os.path.join(HERE, "..", "scenarios", "product_launch.yaml"),
               sets=["narratives.company_denial.start=12", "model.feed_visibility=0.1"])
    assert next(n for n in cfg["narratives"] if n["id"] == "company_denial")["start"] == 12
    assert cfg["model"]["feed_visibility"] == 0.1


if __name__ == "__main__":
    test_launch()
    test_overrides()
    print("ok")
