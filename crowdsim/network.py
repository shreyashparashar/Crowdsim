"""Social graph: dense local communities (friends, family, coworkers) + a heavy-tailed
"follow" layer (popular accounts get most followers, the top ones become influencers).

Stored as one sparse matrix W where W[i, j] = how strongly i sees j's posts.
So `W @ x` gives every agent the weighted sum of x over the people it listens to.
"""
import numpy as np
from scipy import sparse


class Network:
    pass


def build_network(pop, cfg, rng):
    nc = cfg.get("network") or {}
    n = pop.n
    mean_deg = float(nc.get("mean_degree", 12))
    local_frac = float(nc.get("local_fraction", 0.65))
    comm = float(nc.get("community_size", 150))
    homophily = float(nc.get("homophily", 0.8))
    n_infl = max(1, int(float(nc.get("influencer_share", 0.001)) * n))
    w_local = float(nc.get("local_tie_weight", 1.0))
    w_follow = float(nc.get("follow_tie_weight", 0.5))

    # 1. communities: agents are sorted by segment, so chopping the line into chunks gives
    #    segment-homogeneous groups; shuffling (1-homophily) of them mixes groups.
    order = np.arange(n, dtype=np.int32)
    m = int((1 - homophily) * n)
    if m > 1:
        idx = rng.choice(n, m, replace=False)
        order[idx] = order[rng.permutation(idx)]
    sizes = (rng.lognormal(np.log(comm), 0.6, int(2 * n / comm) + 10) + 5).astype(np.int64)
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    bounds = bounds[bounds < n]
    bounds = np.append(bounds, n)
    starts, lens = bounds[:-1], np.diff(bounds)
    comm_of_pos = (np.searchsorted(bounds, np.arange(n), side="right") - 1).astype(np.int32)

    # 2. local mutual ties inside each community
    k_loc = rng.poisson(mean_deg * local_frac / 2, n)
    src_pos = np.repeat(np.arange(n, dtype=np.int32), k_loc)
    c = comm_of_pos[src_pos]
    partner_pos = starts[c] + (rng.random(src_pos.size) * lens[c]).astype(np.int64)
    a, b = order[src_pos], order[partner_pos]
    del src_pos, c, partner_pos

    # 3. follow ties: target chosen proportional to a Pareto "popularity" -> influencers
    k_fol = rng.poisson(mean_deg * (1 - local_frac), n)
    follower = np.repeat(np.arange(n, dtype=np.int32), k_fol)
    cum = np.cumsum(rng.pareto(1.2, n) + 1.0)
    followed = np.searchsorted(cum, rng.random(follower.size) * cum[-1]).astype(np.int32)
    followed = np.minimum(followed, n - 1)

    dst = np.concatenate([a, b, follower])
    src = np.concatenate([b, a, followed])
    w = np.concatenate([np.full(2 * a.size, w_local, np.float32), np.full(follower.size, w_follow, np.float32)])
    keep = dst != src
    W = sparse.csr_matrix((w[keep], (dst[keep], src[keep])), shape=(n, n), dtype=np.float32)
    del a, b, dst, src, w, keep

    net = Network()
    net.W = W
    net.instrength = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
    net.instr_safe = np.maximum(net.instrength, 1e-6)
    net.followers = np.bincount(followed, minlength=n).astype(np.int32)
    net.influencers = np.argsort(-net.followers)[:n_infl].astype(np.int32)
    net.community = np.empty(n, np.int32)
    net.community[order] = comm_of_pos
    net.n_edges = W.nnz

    # influencers post a lot and are online a lot
    pop.expressiveness[net.influencers] = rng.uniform(0.6, 0.95, n_infl)
    pop.activity[net.influencers] = rng.uniform(0.6, 0.95, n_infl)
    return net
