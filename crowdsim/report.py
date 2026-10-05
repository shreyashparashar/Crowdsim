"""Turns run histories into metrics CSVs, charts and a single self-contained HTML report."""
import base64
import csv
import html
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def quantiles(runs):
    """runs: list of {col: array}. Returns {col_p10/_p50/_p90: array}."""
    out = {"hour": runs[0]["hour"]}
    for k in runs[0]:
        if k == "hour":
            continue
        a = np.vstack([r[k] for r in runs])
        for q in (10, 50, 90):
            out[f"{k}_p{q}"] = np.percentile(a, q, axis=0)
    return out


def write_csv(path, cols):
    keys = list(cols)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for row in zip(*(cols[k] for k in keys)):
            w.writerow([f"{v:.6g}" for v in row])


def read_csv(path):
    with open(path) as f:
        r = csv.reader(f)
        keys = next(r)
        rows = np.array([[float(v) for v in row] for row in r])
    return {k: rows[:, i] for i, k in enumerate(keys)}


def _fig_to_b64(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _band(ax, q, key, label, scale=1.0):
    h = q["hour"] / 24
    ax.plot(h, q[f"{key}_p50"] * scale, label=label)
    if f"{key}_p10" in q:
        ax.fill_between(h, q[f"{key}_p10"] * scale, q[f"{key}_p90"] * scale, alpha=0.2)


def make_report(out, cfg, q, narr, segs, buyer_segs, events, final, n_runs, meta, extra=()):
    figs = []

    def new(title, ylabel):
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.set_title(title)
        ax.set_xlabel("day")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        return fig, ax

    fig, ax = new("Who has heard each story", "% of population")
    for n in narr:
        _band(ax, q, f"aware_{n}", n, 100)
    ax.legend(fontsize=8)
    figs.append(("reach", _fig_to_b64(fig, f"{out}/reach.png")))

    fig, ax = new("Conversation volume (posts per hour)", "posts")
    for n in narr:
        _band(ax, q, f"posts_{n}", n)
    ax.legend(fontsize=8)
    figs.append(("volume", _fig_to_b64(fig, f"{out}/volume.png")))

    fig, ax = new("Attitude toward the brand/product by segment", "mean opinion (-1..1)")
    for s in segs:
        _band(ax, q, f"opinion_{s}", s)
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=8)
    figs.append(("opinion", _fig_to_b64(fig, f"{out}/opinion.png")))

    fig, ax = new("Sentiment", "share")
    _band(ax, q, "positive_share", "positive (>0.2)")
    _band(ax, q, "negative_share", "negative (<-0.2)")
    _band(ax, q, "net_sentiment_posts", "net sentiment of posts")
    _band(ax, q, "buzz", "emotional arousal")
    ax.legend(fontsize=8)
    figs.append(("sentiment", _fig_to_b64(fig, f"{out}/sentiment.png")))

    if buyer_segs:
        fig, ax = new("Adoption (cumulative, % of segment)", "%")
        for s in buyer_segs:
            _band(ax, q, f"adoption_{s}", s, 100)
        ax.legend(fontsize=8)
        figs.append(("adoption", _fig_to_b64(fig, f"{out}/adoption.png")))

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, s in enumerate(segs):
        ax.hist(final["opinion"][final["segment"] == i], bins=60, range=(-1, 1), alpha=0.5, label=s, density=True)
    ax.set_title("Final opinion distribution (run 1)")
    ax.legend(fontsize=8)
    figs.append(("distribution", _fig_to_b64(fig, f"{out}/distribution.png")))

    insights = _insights(q, narr, segs, buyer_segs, events) + list(extra)
    with open(f"{out}/summary.json", "w") as f:
        json.dump({"meta": meta, "insights": insights, "events": events}, f, indent=2)

    imgs = "\n".join(f'<img src="data:image/png;base64,{b}" alt="{k}">' for k, b in figs)
    lis = "\n".join(f"<li>{html.escape(s)}</li>" for s in insights)
    evs = "\n".join(f"<li>day {t / 24:.1f}: {html.escape(e)}</li>" for t, e in events)
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(meta['scenario'])} — crowdsim</title>
<style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.5}}
img{{max-width:100%;border:1px solid #ddd;margin:.5rem 0}} code{{background:#f3f3f3;padding:0 .3em}}
.meta{{color:#666;font-size:.9rem}}</style></head><body>
<h1>{html.escape(meta['scenario'])}</h1>
<p class="meta">{meta['agents']:,} agents · {meta['edges']:,} social ties · {meta['hours']} hours ·
{n_runs} Monte-Carlo run(s), bands = 10th–90th percentile · built in {meta['seconds']:.0f}s</p>
<h2>Key findings</h2><ul>{lis}</ul>
<h2>Timeline</h2><ul>{evs}</ul>
<h2>Charts</h2>{imgs}
<p class="meta">Uncalibrated simulations are for comparing scenarios ("what if we deny the rumor
6 hours earlier?"), not for point forecasts. Calibrate against a past event first.</p>
</body></html>"""
    with open(f"{out}/report.html", "w") as f:
        f.write(page)
    return insights


def _insights(q, narr, segs, buyer_segs, events):
    h = q["hour"]
    out = []
    for n in narr:
        reach = q[f"aware_{n}_p50"]
        if reach[-1] <= 0:
            continue
        peak_post = int(h[np.argmax(q[f"posts_{n}_p50"])])
        line = (f"'{n}' reached {reach[-1] * 100:.1f}% of everyone "
                f"(believed by {q[f'believe_{n}_p50'][-1] * 100:.1f}%); conversation peaked on day {peak_post / 24:.1f}.")
        if f"corrected_{n}_p50" in q:
            line += f" Debunking un-convinced {q[f'corrected_{n}_p50'][-1] * 100:.2f}% of the population."
        out.append(line)
    for s in segs:
        a, b = q[f"opinion_{s}_p50"][0], q[f"opinion_{s}_p50"][-1]
        out.append(f"{s}: attitude {a:+.2f} → {b:+.2f} ({'up' if b > a else 'down'} {abs(b - a):.2f}).")
    neg = q["negative_share_p50"]
    out.append(f"Negative share of population peaked at {neg.max() * 100:.1f}% on day {h[np.argmax(neg)] / 24:.1f} "
               f"and ended at {neg[-1] * 100:.1f}%.")
    out.append(f"Emotional buzz peaked on day {h[np.argmax(q['buzz_p50'])] / 24:.1f}.")
    for s in buyer_segs:
        out.append(f"Adoption among {s}: {q[f'adoption_{s}_p50'][-1] * 100:.2f}% "
                   f"(range {q[f'adoption_{s}_p10'][-1] * 100:.2f}–{q[f'adoption_{s}_p90'][-1] * 100:.2f}%).")
    return out


def compare(dirs, out):
    """Overlay key curves from several finished runs (e.g. 'deny early' vs 'deny late')."""
    os.makedirs(out, exist_ok=True)
    data = {os.path.basename(os.path.normpath(d)): read_csv(f"{d}/metrics_summary.csv") for d in dirs}
    keys = ["mean_opinion", "negative_share", "buzz", "adoption", "net_sentiment_posts", "posts_total"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    for ax, k in zip(axes.flat, keys):
        for name, q in data.items():
            _band(ax, q, k, name)
        ax.set_title(k)
        ax.set_xlabel("day")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/compare.png", dpi=110)
    plt.close(fig)
    print(f"{'scenario':30s} " + " ".join(f"{k:>20s}" for k in keys))
    for name, q in data.items():
        print(f"{name:30s} " + " ".join(f"{q[f'{k}_p50'][-1]:20.4f}" for k in keys))
    print(f"wrote {out}/compare.png")
