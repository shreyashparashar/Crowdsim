"""Synthetic population. Each agent = one index into a set of numpy arrays (its "brain").

Traits (all 0..1 unless noted):
  openness        how far an opinion can be from yours and still move you
  conformity      how strongly you drift toward your friends' opinions
  reactivity      emotional reactivity: how hard news hits you, how aroused you get
  skepticism      resistance to low-credibility claims
  expressiveness  tendency to post/share (heavily skewed: most people lurk)
  novelty         innovativeness (early adopter vs laggard)
  trust_brand     trust in the company's own statements
  trust_media     trust in news outlets
  media_reach     hourly chance of catching a mass-media item at intensity 1
  activity        chance of being online in a peak hour (lognormal, heavy tail)
State that changes during the run (opinion, arousal, beliefs, purchases) lives in engine.py.
"""
import numpy as np

# default Beta(a, b) for each trait; a segment can shift the mean via `traits: {name: mean}`
TRAITS = {
    "openness": (2, 2), "conformity": (2, 2), "reactivity": (2, 2), "skepticism": (2, 3),
    "expressiveness": (0.6, 6), "novelty": (2, 4), "trust_brand": (2, 2),
    "trust_media": (2, 2), "media_reach": (1.5, 12),
}


class Population:
    def __init__(self, cfg, rng):
        pc = cfg["population"]
        n = int(pc["size"])
        segs = pc["segments"]
        shares = np.array([s["share"] for s in segs], float)
        counts = np.floor(shares / shares.sum() * n).astype(int)
        counts[np.argmax(counts)] += n - counts.sum()

        self.n = n
        self.segment_names = [s["name"] for s in segs]
        # agents are laid out sorted by segment; network.py relies on this for homophily
        self.segment = np.repeat(np.arange(len(segs), dtype=np.int16), counts)
        f32 = np.float32
        for name in TRAITS:
            setattr(self, name, np.empty(n, f32))
        self.activity = np.empty(n, f32)
        self.opinion0 = np.empty(n, f32)
        self.need = np.empty(n, f32)
        self.price_sensitivity = np.empty(n, f32)
        self.can_adopt = np.zeros(n, bool)
        self.adopted0 = np.zeros(n, bool)

        def beta_mean(mean, conc, size):
            mean = min(max(mean, 0.01), 0.99)
            return rng.beta(mean * conc, (1 - mean) * conc, size)

        lo = 0
        for s, c in zip(segs, counts):
            sl = slice(lo, lo + c)
            lo += c
            over = s.get("traits") or {}
            for name, (a, b) in TRAITS.items():
                if name in over:
                    getattr(self, name)[sl] = beta_mean(over[name], a + b, c)
                else:
                    getattr(self, name)[sl] = rng.beta(a, b, c)
            act = rng.lognormal(np.log(0.25 * s.get("activity", 1.0)), 0.7, c)
            self.activity[sl] = np.clip(act, 0.01, 0.95)
            op = rng.normal(s.get("baseline_opinion", 0.0), s.get("opinion_sd", 0.25), c)
            self.opinion0[sl] = np.clip(op, -0.95, 0.95)
            self.need[sl] = beta_mean(s.get("need", 0.5), 6, c)
            self.price_sensitivity[sl] = beta_mean(s.get("price_sensitivity", 0.5), 6, c)
            self.can_adopt[sl] = bool(s.get("can_adopt", False))
            self.adopted0[sl] = rng.random(c) < s.get("already_adopted", 0.0)
