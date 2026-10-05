"""Usage:
  python -m crowdsim run scenarios/product_launch.yaml [--size N] [--runs R] [--set key=value ...]
  python -m crowdsim compare outputs/a outputs/b --out outputs/compare
  python -m crowdsim calibrate scenarios/x.yaml --observed data/observed.csv [--trials 40]

--set uses dotted paths; list items are addressed by id:
  --set narratives.company_denial.start=36   --set model.feed_visibility=0.3
"""
import argparse
import copy
import csv
import os
import sys
import time

import numpy as np
import yaml

from .engine import Simulation
from .network import build_network
from .population import Population
from . import report


def load(path, sets=(), size=None, hours=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for s in sets:
        key, val = s.split("=", 1)
        cur = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if isinstance(cur, list):
                cur = next((x for x in cur if x.get("id") == part or x.get("name") == part), None)
                if cur is None:
                    sys.exit(f"--set {key}: no list item with id/name '{part}'")
            else:
                cur = cur.setdefault(part, {})
        cur[parts[-1]] = yaml.safe_load(val)
    cfg.setdefault("simulation", {})
    if size:
        cfg["population"]["size"] = size
    if hours:
        cfg["simulation"]["hours"] = hours
    return cfg


def build_world(cfg, log=print):
    seed = int(cfg["simulation"].get("seed", 42))
    rng = np.random.default_rng(seed)
    t0 = time.time()
    pop = Population(cfg, rng)
    net = build_network(pop, cfg, rng)
    log(f"world: {pop.n:,} agents, {net.n_edges:,} ties, top influencer has "
        f"{net.followers.max():,} followers ({time.time() - t0:.1f}s)")
    return pop, net


def simulate(cfg, pop, net, run_id=0, log=print):
    hours = int(cfg["simulation"].get("hours", 168))
    sim = Simulation(cfg, pop, net, int(cfg["simulation"].get("seed", 42)) + 1000 + run_id)
    t0 = time.time()
    for t in range(hours):
        sim.step()
        if log and (t + 1) % 24 == 0:
            h = sim.history[-1]
            log(f"  run {run_id + 1} day {(t + 1) // 24:3d} | opinion {h['mean_opinion']:+.3f} "
                f"| negative {h['negative_share'] * 100:5.1f}% | posts/h {h['posts_total']:7d} "
                f"| adoption {h['adoption'] * 100:5.2f}% | {time.time() - t0:5.0f}s")
    return sim


def cmd_run(a):
    cfg = load(a.scenario, a.set, a.size, a.hours)
    name = cfg.get("name", os.path.splitext(os.path.basename(a.scenario))[0])
    out = a.out or os.path.join("outputs", name)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    pop, net = build_world(cfg)
    runs, first, pickups = [], None, {}
    for r in range(a.runs):
        sim = simulate(cfg, pop, net, r)
        runs.append(sim.history_arrays())
        for n in sim.narr:
            if n.pickup:
                pickups.setdefault(n.id, []).append(n.picked_up_at)
        report.write_csv(f"{out}/metrics_run{r + 1}.csv", runs[-1])
        if first is None:
            first = sim
    q = report.quantiles(runs)
    report.write_csv(f"{out}/metrics_summary.csv", q)
    _agent_sample(first, pop, net, f"{out}/agents_sample.csv", a.sample)
    with open(f"{out}/scenario_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    meta = dict(scenario=name, agents=pop.n, edges=int(net.n_edges), hours=len(q["hour"]),
                seconds=time.time() - t0)
    buyer_segs = [s for i, s in enumerate(pop.segment_names) if pop.can_adopt[pop.segment == i].any()]
    insights = report.make_report(
        out, cfg, q, [n.id for n in first.narr], pop.segment_names, buyer_segs,
        [(int(t), e) for t, e in first.events], {"opinion": first.opinion, "segment": pop.segment},
        a.runs, meta, extra=[
            f"Mainstream media picked up '{k}' in {sum(x is not None for x in v)} of {len(v)} runs"
            + (f" (median day {np.median([x for x in v if x is not None]) / 24:.1f})." if any(x is not None for x in v) else ".")
            for k, v in pickups.items()])
    print("\n".join(["", *insights, "", f"report: {out}/report.html"]))


def _agent_sample(sim, pop, net, path, k):
    """A few thousand individual agents (traits + final state) to inspect real 'people'."""
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(pop.n, min(k, pop.n), replace=False))
    names = ["unaware", "spreading", "believes", "rejected"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        traits = ["openness", "conformity", "reactivity", "skepticism", "expressiveness",
                  "novelty", "trust_brand", "trust_media", "activity"]
        w.writerow(["agent", "segment", "followers", "influencer", *traits, "opinion_start",
                    "opinion_end", "adopted", *[f"story_{n.id}" for n in sim.narr]])
        infl = set(net.influencers.tolist())
        for i in idx:
            w.writerow([i, pop.segment_names[pop.segment[i]], net.followers[i], int(i in infl),
                        *[f"{getattr(pop, t)[i]:.3f}" for t in traits],
                        f"{pop.opinion0[i]:.3f}", f"{sim.opinion[i]:.3f}", int(sim.adopted[i]),
                        *[names[sim.state[n.idx, i]] for n in sim.narr]])


def main():
    ap = argparse.ArgumentParser(prog="crowdsim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("scenario")
    r.add_argument("--size", type=int)
    r.add_argument("--hours", type=int)
    r.add_argument("--runs", type=int, default=1)
    r.add_argument("--set", action="append", default=[])
    r.add_argument("--out")
    r.add_argument("--sample", type=int, default=3000)
    c = sub.add_parser("compare")
    c.add_argument("dirs", nargs="+")
    c.add_argument("--out", default="outputs/compare")
    k = sub.add_parser("calibrate")
    k.add_argument("scenario")
    k.add_argument("--observed", required=True)
    k.add_argument("--trials", type=int, default=40)
    k.add_argument("--size", type=int, default=100_000)
    k.add_argument("--params", default="feed_visibility,share_bias,accept_bias,opinion_impact,social_influence")
    k.add_argument("--set", action="append", default=[])
    k.add_argument("--out", default="calibrated.yaml")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "compare":
        report.compare(a.dirs, a.out)
    else:
        from .calibrate import calibrate
        calibrate(a)


if __name__ == "__main__":
    main()
