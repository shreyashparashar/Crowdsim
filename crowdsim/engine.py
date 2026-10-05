"""One tick = one hour. Every hour, for every agent:

  1. Am I online? (personal activity x time-of-day curve)
  2. For every story in circulation (official news, rumors, denials, reviews...):
       - people spreading it may post (more if emotional, aroused, expressive, story is fresh)
       - I can see it via friends' posts, the platform's trending feed, or mass media
       - first time I see it I accept or reject it, based on credibility, my skepticism,
         whether it fits what I already think (confirmation bias), how many friends
         believe it (social proof), trust in the source, and whether I've seen it debunked
       - if I accept it, it shifts my opinion (negative news hits harder) and my arousal,
         and I may start spreading it
       - spreaders get bored or find everyone already knows (stifling) and stop
       - a rejected claim seen over and over can still stick (illusory-truth effect)
       - accepting a debunk can un-convince believers, but only partly (continued influence),
         and the denial itself spreads word of the rumor (Streisand effect)
  3. My opinion drifts toward what my friends post, if it's not too far from mine
     (bounded confidence), and slowly relaxes back to my baseline.
  4. If I'm a potential customer and aware of the product, I may buy (Bass-style:
     own attitude + innovativeness + friends who already bought). Buyers later
     experience the product and may post reviews, which spread like any other story.

All of this is vectorised: the whole social step is ONE sparse-matrix x dense-matrix
product per hour, so a million agents run on a laptop.
"""
import numpy as np

UNAWARE, SPREADER, BELIEVER, REJECTER = 0, 1, 2, 3

DEFAULT_MODEL = dict(
    # exposure
    feed_visibility=0.4,      # chance you notice one post in your feed (per unit tie weight)
    feed_decay=0.85,          # unread posts sink in the feed by this factor per hour
    algo_boost=3.0,           # how strongly trending stories get pushed to everyone
    trend_cap=0.03,           # max hourly chance of seeing a story via trending
    post_rate=0.3,            # base posting rate for spreaders (per online hour)
    # acceptance (logit weights)
    accept_bias=0.0, w_credibility=2.0, w_skepticism=3.0, w_confirmation=1.5,
    w_social_proof=2.0, w_trust=1.0, w_prior_debunk=2.5,
    w_emotion_belief=1.5,     # emotional content short-circuits scrutiny in reactive people
    # sharing
    share_bias=-3.0, w_emotion=3.0, w_arousal=1.5, w_expressive=4.0,
    # story lifecycle
    forget_rate=0.02, stifle_rate=0.1, novelty_halflife=36.0,
    repetition_effect=0.01, correction_efficacy=0.6, continued_influence=0.5,
    streisand=0.15,           # chance a denial tells someone about the rumor it denies
    # opinions & emotion
    opinion_impact=0.4, negativity_bias=0.5, social_influence=0.05,
    confidence_bound=0.5, opinion_relax=0.002, arousal_gain=0.6, arousal_decay=0.92,
    # adoption
    adopt_base_rate=0.002, a_opinion=3.0, a_need=1.5, a_price=1.0, a_peer=3.0,
    a_innov=1.0, a_bias=-2.0, experience_delay=48, experience_impact=0.4, review_rate=0.15,
)

# share of people online by hour of day (0..23), peak = 1
DIURNAL = np.array([.25, .15, .10, .08, .08, .12, .30, .55, .70, .75, .75, .80,
                    .85, .80, .75, .75, .80, .85, .95, 1.0, 1.0, .95, .75, .45], np.float32)

SOURCE_BY_KIND = {"official": "brand", "debunk": "brand", "ad": "brand", "news": "media"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Narrative:
    """A story in circulation. `media` windows and `start` are in absolute hours."""

    def __init__(self, d, idx, p):
        self.idx = idx
        self.id = d["id"]
        self.kind = d.get("kind", "news")
        self.valence = float(d.get("valence", 0.0))          # -1 bad for the brand .. +1 good
        self.credibility = float(d.get("credibility", 0.5))
        self.emotionality = float(d.get("emotionality", 0.5))
        self.start = int(d.get("start", 0))
        self.about_product = bool(d.get("about_product", True))
        self.truth = d.get("truth")
        self.source = d.get("source", SOURCE_BY_KIND.get(self.kind))
        self.seed = d.get("seed") or {}
        self.halflife = float(d.get("novelty_halflife", p["novelty_halflife"]))
        self.media = [(int(m["start"]), int(m["end"]), float(m["intensity"])) for m in d.get("media", [])]
        self.pickup = d.get("media_pickup")
        self.picked_up_at = None
        self.debunks = d.get("debunks")
        self.debunk_target = None
        self.corrected = 0

    def media_intensity(self, t):
        return sum(i for s, e, i in self.media if s <= t < e)

    def novelty(self, t):
        return 0.5 ** ((t - self.start) / self.halflife)


class Simulation:
    def __init__(self, cfg, pop, net, seed):
        self.pop, self.net = pop, net
        self.p = p = {**DEFAULT_MODEL, **(cfg.get("model") or {})}
        self.rng = np.random.default_rng(seed)
        sim = cfg.get("simulation") or {}
        self.start_hour = int(sim.get("start_hour_of_day", 9))
        prod = cfg.get("product") or {}
        self.quality = prod.get("quality")

        defs = list(cfg["narratives"])
        if self.quality is not None and not any(d["id"] == "user_reviews" for d in defs):
            q = float(self.quality)  # reviews by real buyers: a story seeded by adoption itself
            defs.append(dict(id="user_reviews", kind="review", valence=q, credibility=0.7,
                             emotionality=0.3 + 0.4 * abs(q), start=0, novelty_halflife=1e9))
        self.narr = [Narrative(d, i, p) for i, d in enumerate(defs)]
        ids = {n.id: n for n in self.narr}
        if len(ids) != len(self.narr):
            raise ValueError("narrative ids must be unique")
        for n in self.narr:
            if n.debunks:
                if n.debunks not in ids:
                    raise ValueError(f"{n.id} debunks unknown narrative {n.debunks!r}")
                n.debunk_target = ids[n.debunks].idx
        self.debunkers = {n.idx: [d for d in self.narr if d.debunk_target == n.idx] for n in self.narr}
        self.review = ids.get("user_reviews")

        K, N = len(self.narr), pop.n
        self.K = K
        self.state = np.zeros((K, N), np.int8)
        self.opinion = pop.opinion0.copy()
        self.arousal = np.zeros(N, np.float32)
        self.adopted = pop.adopted0.copy()
        self.adopt_time = np.full(N, -10**6, np.int32)
        self.product_aware = self.adopted.copy()
        self.X = np.zeros((N, 2 * K + 3), np.float32)
        self._zeros = np.zeros(N, np.float32)
        self.feed = np.zeros((K, N), np.float32)   # unread posts per story, consumed when online
        self.t = 0
        self.history = []
        self.events = []
        S = len(pop.segment_names)
        self.seg_n = np.maximum(np.bincount(pop.segment, minlength=S), 1)
        self.seg_buyers = np.bincount(pop.segment, weights=pop.can_adopt, minlength=S)

    # ---------------------------------------------------------------- helpers
    def _impact(self, n, idx, sign=1.0):
        """Accepting story n moves opinion; tanh/atanh keeps it inside (-1, 1) with saturation."""
        if not n.about_product or idx.size == 0:
            return
        p = self.p
        neg = 1 + p["negativity_bias"] * (n.valence < 0)
        delta = sign * p["opinion_impact"] * n.valence * (0.5 + 0.5 * n.credibility) * neg \
            * (0.5 + self.pop.reactivity[idx])
        o = np.clip(self.opinion[idx], -0.995, 0.995)
        self.opinion[idx] = np.tanh(np.arctanh(o) + delta)

    def _seed(self, n):
        s, rng, pop = n.seed, self.rng, self.pop
        count = int(s["share"] * pop.n) if "share" in s else int(s.get("count", 0))
        if count <= 0:
            return
        target = s.get("target", "random")
        if target == "influencers":
            idx = self.net.influencers[:count]
        elif target.startswith("segment:"):
            name = target.split(":", 1)[1]
            pool = np.flatnonzero(pop.segment == pop.segment_names.index(name))
            idx = rng.choice(pool, min(count, pool.size), replace=False)
        elif target == "active":
            pool = np.argsort(-pop.expressiveness)[: max(count * 20, 1000)]
            idx = rng.choice(pool, min(count, pool.size), replace=False)
        else:
            idx = rng.choice(pop.n, min(count, pop.n), replace=False)
        idx = idx[self.state[n.idx, idx] == UNAWARE]
        self.state[n.idx, idx] = SPREADER
        if n.about_product:
            self.product_aware[idx] = True
        self._impact(n, idx)
        self.events.append((self.t, f"'{n.id}' released ({target}, {idx.size} seeds)"))

    def _encounter(self, n, idx, sp, t):
        """First contact with story n for agents idx: accept/reject, maybe start spreading."""
        p, pop, rng, k = self.p, self.pop, self.rng, n.idx
        logit = (p["accept_bias"] + p["w_credibility"] * (2 * n.credibility - 1)
                 - p["w_skepticism"] * pop.skepticism[idx] * (1 - n.credibility)
                 + p["w_confirmation"] * self.opinion[idx] * n.valence
                 + p["w_social_proof"] * sp[idx]
                 + p["w_emotion_belief"] * n.emotionality * (pop.reactivity[idx] - 0.3))
        if n.source == "brand":
            logit = logit + p["w_trust"] * (2 * pop.trust_brand[idx] - 1)
        elif n.source == "media":
            logit = logit + p["w_trust"] * (2 * pop.trust_media[idx] - 1)
        for d in self.debunkers[k]:
            if t >= d.start:
                st = self.state[d.idx, idx]
                logit = logit - p["w_prior_debunk"] * d.credibility * ((st == SPREADER) | (st == BELIEVER))
        acc = rng.random(idx.size) < sigmoid(logit)
        a_idx, r_idx = idx[acc], idx[~acc]
        share = sigmoid(p["share_bias"] + p["w_emotion"] * n.emotionality
                        + p["w_arousal"] * self.arousal[a_idx]
                        + p["w_expressive"] * pop.expressiveness[a_idx]) * (0.3 + 0.7 * n.novelty(t))
        spread = rng.random(a_idx.size) < share
        self.state[k, a_idx[spread]] = SPREADER
        self.state[k, a_idx[~spread]] = BELIEVER
        self.state[k, r_idx] = REJECTER
        if n.about_product:
            self.product_aware[idx] = True
        self._impact(n, a_idx)
        self.arousal[a_idx] = np.minimum(
            1.0, self.arousal[a_idx] + p["arousal_gain"] * n.emotionality * pop.reactivity[a_idx])
        if n.debunk_target is not None:
            self._correct(n, a_idx)
            # Streisand effect: the denial itself informs people who never heard the claim
            target = self.narr[n.debunk_target]
            new = idx[self.state[target.idx, idx] == UNAWARE]
            new = new[rng.random(new.size) < p["streisand"]]
            if new.size:
                self._encounter(target, new, self._zeros, t)

    def _correct(self, d, idx):
        """Agents who just accepted debunk d may drop their belief in its target."""
        target = self.narr[d.debunk_target]
        st = self.state[target.idx, idx]
        cand = idx[(st == SPREADER) | (st == BELIEVER)]
        if cand.size == 0:
            return
        prob = self.p["correction_efficacy"] * d.credibility * (0.5 + 0.5 * self.pop.skepticism[cand])
        flip = cand[self.rng.random(cand.size) < prob]
        self.state[target.idx, flip] = REJECTER
        self._impact(target, flip, sign=-(1 - self.p["continued_influence"]))
        target.corrected += flip.size

    # ---------------------------------------------------------------- main loop
    def step(self):
        t, p, pop, rng, net, K = self.t, self.p, self.pop, self.rng, self.net, self.K
        N = pop.n
        for n in self.narr:
            if n.start == t:
                self._seed(n)
        online = rng.random(N, dtype=np.float32) < pop.activity * DIURNAL[(self.start_hour + t) % 24]
        active = [n for n in self.narr if t >= n.start]

        # who posts what this hour
        X = self.X
        X.fill(0)
        any_post = np.zeros(N, bool)
        posts, believers = {}, {}
        for n in active:
            st = self.state[n.idx]
            bel = (st == SPREADER) | (st == BELIEVER)
            X[:, K + n.idx] = bel
            believers[n.idx] = int(bel.sum())
            si = np.flatnonzero((st == SPREADER) & online)
            pp = (p["post_rate"] * (0.5 + 3 * pop.expressiveness[si]) * (1 + self.arousal[si])
                  * (0.3 + 0.7 * n.novelty(t)) * (0.5 + n.emotionality))
            posted = si[rng.random(si.size) < pp]
            X[posted, n.idx] = 1
            any_post[posted] = True
            posts[n.idx] = posted.size
        X[:, 2 * K] = self.opinion * any_post
        X[:, 2 * K + 1] = any_post
        X[:, 2 * K + 2] = self.adopted

        Y = net.W @ X   # the entire social step: everyone hears everyone they follow

        # media agenda-setting: newsrooms pick up stories once enough people believe them
        for n in active:
            pu = n.pickup
            if pu and n.picked_up_at is None and believers[n.idx] / N >= float(pu.get("threshold", 0.02)):
                s = t + int(pu.get("delay", 2))
                n.media.append((s, s + int(pu.get("duration", 24)), float(pu.get("intensity", 0.03))))
                n.picked_up_at = t
                self.events.append((t, f"media picked up '{n.id}'"))

        # exposure -> accept / reject
        for n in active:
            k = n.idx
            feed = self.feed[k]
            feed *= p["feed_decay"]
            feed += Y[:, k]
            sp = Y[:, K + k] / net.instr_safe
            un = np.flatnonzero(self.state[k] == UNAWARE)
            if un.size == 0:
                continue
            on = online[un]
            p_net = 1 - np.exp(-p["feed_visibility"] * feed[un])
            p_trend = min(p["trend_cap"], p["algo_boost"] * posts[k] / N)
            p_media = n.media_intensity(t) * pop.media_reach[un]
            p_seen = 1 - (1 - p_net * on) * (1 - p_trend * on) * (1 - p_media)
            seen = un[rng.random(un.size) < p_seen]
            if seen.size:
                self._encounter(n, seen, sp, t)

        # story lifecycle: spreaders stop, repeated claims creep in
        for n in active:
            k, st = n.idx, self.state[n.idx]
            si = np.flatnonzero(st == SPREADER)
            if si.size:
                sp = Y[si, K + k] / net.instr_safe[si]
                stop = p["forget_rate"] + p["stifle_rate"] * sp + 0.05 * (1 - n.novelty(t))
                st[si[rng.random(si.size) < stop]] = BELIEVER
            if p["repetition_effect"] > 0:
                ri = np.flatnonzero((st == REJECTER) & online & (self.feed[k] > 0))
                if ri.size:
                    pr = p["repetition_effect"] * (1 - np.exp(-p["feed_visibility"] * self.feed[k, ri])) \
                        * (1 - pop.skepticism[ri])
                    flip = ri[rng.random(ri.size) < pr]
                    st[flip] = BELIEVER
                    self._impact(n, flip)
            self.feed[k, online] = 0   # whoever logged in has scrolled their feed

        # opinion dynamics: bounded-confidence drift toward what friends post
        hear = np.flatnonzero(online & (Y[:, 2 * K + 1] > 0))
        if hear.size:
            m = Y[hear, 2 * K] / Y[hear, 2 * K + 1]
            o = self.opinion[hear]
            diff = m - o
            eps = p["confidence_bound"] * (0.5 + pop.openness[hear])
            self.opinion[hear] = o + p["social_influence"] * (0.5 + pop.conformity[hear]) \
                * diff * np.exp(-(diff / eps) ** 2)
        self.opinion += p["opinion_relax"] * (pop.opinion0 - self.opinion)
        self.arousal *= p["arousal_decay"]

        # adoption (Bass-style hazard), then delayed product experience + reviews
        new_adopt = 0
        ci = np.flatnonzero(pop.can_adopt & ~self.adopted & self.product_aware)
        if ci.size:
            att = sigmoid(p["a_opinion"] * self.opinion[ci] + p["a_need"] * (2 * pop.need[ci] - 1)
                          - p["a_price"] * (2 * pop.price_sensitivity[ci] - 1)
                          + p["a_peer"] * Y[ci, 2 * K + 2] / net.instr_safe[ci]
                          + p["a_innov"] * (2 * pop.novelty[ci] - 1) + p["a_bias"])
            haz = p["adopt_base_rate"] * att * (0.3 + 0.7 * online[ci])
            new = ci[rng.random(ci.size) < haz]
            self.adopted[new] = True
            self.adopt_time[new] = t
            new_adopt = new.size
        if self.quality is not None:
            ex = np.flatnonzero(self.adopt_time == t - int(p["experience_delay"]))
            if ex.size:
                o = np.clip(self.opinion[ex], -0.995, 0.995)
                self.opinion[ex] = np.tanh(np.arctanh(o) + p["experience_impact"] * float(self.quality))
                r = ex[rng.random(ex.size) < p["review_rate"] + pop.expressiveness[ex]]
                self.state[self.review.idx, r] = SPREADER

        self._record(t, online, posts, new_adopt)
        self.t += 1

    # ---------------------------------------------------------------- metrics
    def _record(self, t, online, posts, new_adopt):
        pop, N, o = self.pop, self.pop.n, self.opinion
        rec = {"hour": t, "online_share": float(online.mean())}
        total = sum(posts.values())
        rec["posts_total"] = total
        val = sum(posts.get(n.idx, 0) * n.valence for n in self.narr if n.about_product)
        rel = sum(posts.get(n.idx, 0) for n in self.narr if n.about_product)
        rec["net_sentiment_posts"] = val / rel if rel else 0.0
        for n in self.narr:
            c = np.bincount(self.state[n.idx], minlength=4)
            rec[f"aware_{n.id}"] = (N - c[0]) / N
            rec[f"believe_{n.id}"] = (c[1] + c[2]) / N
            rec[f"spreading_{n.id}"] = c[1] / N
            rec[f"posts_{n.id}"] = posts.get(n.idx, 0)
            rec[f"media_{n.id}"] = n.media_intensity(t)
            if n.debunks is None and self.debunkers[n.idx]:
                rec[f"corrected_{n.id}"] = n.corrected / N
        rec["mean_opinion"] = float(o.mean())
        rec["positive_share"] = float((o > 0.2).mean())
        rec["negative_share"] = float((o < -0.2).mean())
        rec["polarization"] = float(o.std())
        rec["buzz"] = float(self.arousal.mean())
        rec["product_awareness"] = float(self.product_aware.mean())
        buyers = pop.can_adopt.sum()
        rec["adoption"] = float((self.adopted & pop.can_adopt).sum() / max(buyers, 1))
        rec["new_adopters"] = new_adopt
        seg_o = np.bincount(pop.segment, weights=o, minlength=len(self.seg_n)) / self.seg_n
        seg_a = np.bincount(pop.segment, weights=self.adopted & pop.can_adopt, minlength=len(self.seg_n))
        for i, name in enumerate(pop.segment_names):
            rec[f"opinion_{name}"] = float(seg_o[i])
            if self.seg_buyers[i]:
                rec[f"adoption_{name}"] = float(seg_a[i] / self.seg_buyers[i])
        self.history.append(rec)

    def history_arrays(self):
        keys = self.history[0].keys()
        return {k: np.array([h[k] for h in self.history], float) for k in keys}
