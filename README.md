# crowdsim

Simulate how **a million people** react to a product launch or a news event — customers
and non-customers alike — with rumors, news coverage, denials, influencers, word of mouth
and purchases all happening at once, hour by hour, on a realistic social network.

Free: pure Python (numpy/scipy), runs on a laptop (1M agents ≈ 3 min and < 1 GB RAM per
simulated fortnight on one CPU core) or on free GitHub Actions runners.

```bash
pip install -r requirements.txt
python -m crowdsim run scenarios/product_launch.yaml            # 1,000,000 agents, 14 days
python -m crowdsim run scenarios/product_launch.yaml --size 100000 --runs 5   # quick, with uncertainty bands
open outputs/phone_launch_with_rumor/report.html
```

## What each person is

Every agent has its own set of traits and its own evolving state (stored as numpy arrays
so a million of them fit in memory):

| Traits (fixed) | What they do |
|---|---|
| skepticism | resists low-credibility claims |
| reactivity | how hard news hits; how emotional you get; emotional content bypasses scrutiny |
| openness / conformity | bounded-confidence opinion drift toward friends |
| expressiveness | posting tendency, heavily skewed (most people lurk, a few post a lot) |
| activity + time-of-day | when you're online (people sleep; posts wait in your feed) |
| trust in brand / media | weight given to official statements vs news outlets |
| media reach | exposure to TV/news broadcasts |
| need, price sensitivity, innovativeness | purchase behaviour |

| State (changes every hour) | |
|---|---|
| opinion of the brand/product, −1…+1 | moved by stories you accept and friends you hear |
| emotional arousal | rises with emotional stories, decays over hours |
| per story: unaware / spreading / believes / rejected | the rumor lifecycle |
| unread feed per story | posts from people you follow, waiting until you log in |
| adopted (bought) | plus delayed product experience and reviews |

## What happens each hour

1. People log in according to their activity and the time of day.
2. Spreaders post (more if the story is emotional and fresh, if they're aroused, if they're expressive).
3. Posts flow along the social graph into followers' feeds — one sparse-matrix product moves
   every story, every opinion signal and every purchase signal for all agents at once.
4. People see stories through their feed, the platform's trending boost, or mass media.
5. First contact → accept or reject, driven by credibility, skepticism, confirmation bias,
   social proof (how many friends believe it), source trust, emotional pull, and prior debunks.
6. Acceptance shifts opinion (negative news hits harder) and may turn you into a spreader.
7. Spreaders lose interest or find everyone already knows (rumor stifling).
8. Rejected claims seen repeatedly can still stick (illusory-truth effect).
9. Debunks un-convince some believers, only partly (continued-influence effect), and the
   denial itself tells new people about the rumor (Streisand effect).
10. Newsrooms pick up stories once enough people believe them (agenda setting) — this is
    a tipping point, which is why `--runs` matters: you get "rumor went mainstream in 2 of 5 runs".
11. Opinions drift toward friends' posts if not too far away (bounded confidence), and relax to baseline.
12. Potential customers who know about the product may buy (Bass-style: attitude, need, price,
    innovativeness, friends who bought). Two days later they've used it, their opinion moves with
    real product `quality`, and some post reviews that spread like any other story.

The social graph: dense local communities (friends/family/work, segment-homophilous) plus a
heavy-tailed follow layer that produces natural influencers (the top account in 1M agents
gets tens of thousands of followers).

## Writing a scenario

See `scenarios/product_launch.yaml` (launch + false rumor + denial + reviews) and
`scenarios/data_breach.yaml` (a news event with outrage, boycott call and apology).

```yaml
population:
  size: 1000000
  segments:            # any number; customers and non-customers
    - name: prospects
      share: 0.22
      baseline_opinion: 0.05   # -1..1 toward the brand/product
      opinion_sd: 0.25
      can_adopt: true          # false = public that talks but doesn't buy
      need: 0.65
      price_sensitivity: 0.55
      already_adopted: 0.0
      activity: 1.0            # multiplier on time online
      traits: {skepticism: 0.5, trust_brand: 0.4}   # shift any trait mean
narratives:
  - id: battery_rumor
    kind: rumor                # official | news | rumor | debunk | ad | review (sets source trust)
    valence: -0.8              # bad (-1) .. good (+1) for the brand
    credibility: 0.35
    emotionality: 0.9
    start: 30                  # hour
    seed: {target: active, share: 0.0006}   # influencers | random | active | segment:<name>; count or share
    media: [{start: 40, end: 60, intensity: 0.04}]
    media_pickup: {threshold: 0.01, delay: 3, duration: 36, intensity: 0.04}
  - id: company_denial
    kind: debunk
    debunks: battery_rumor
product:
  quality: 0.3                 # optional; enables buyer experience + review word of mouth
model: {}                      # override any behavioural constant from engine.DEFAULT_MODEL
```

## What-if analysis (the main use)

Change one thing from the command line, then compare:

```bash
python -m crowdsim run scenarios/product_launch.yaml --size 200000 --runs 5 \
  --set narratives.company_denial.start=36 --set "narratives.company_denial.media=[{start: 36, end: 60, intensity: 0.05}]" \
  --out outputs/deny_early
python -m crowdsim run scenarios/product_launch.yaml --size 200000 --runs 5 --out outputs/deny_late
python -m crowdsim compare outputs/deny_early outputs/deny_late
```

Outputs per run: `report.html` (charts + auto-written findings), `metrics_summary.csv`
(10th/50th/90th percentile of every metric, hourly), `metrics_runN.csv`,
`agents_sample.csv` (a few thousand individual people: traits, start/end opinion, what they
believed, whether they bought), `summary.json`.

## Making it predictive: calibration

Out of the box the behavioural constants are plausible, literature-inspired defaults — good
for comparing strategies, **not** for forecasting real numbers. To forecast, fit the model to a
past event similar to the one you care about, then run the new scenario with those parameters.

```bash
python -m crowdsim calibrate scenarios/my_past_launch.yaml --observed data/observed.csv --trials 60
python -m crowdsim run calibrated.yaml --size 1000000 --runs 5
```

`observed.csv` has an `hour` column and any metric columns the model outputs (see
`data/example_observed.csv`). Mention volume (`posts_total`, `posts_<story>`) is matched by
shape; shares and sentiment are matched directly. Free data sources:

- **Mention volume over time**: Reddit (official API), Bluesky (open firehose), Mastodon, YouTube comments API
- **Search interest**: Google Trends (`pytrends`)
- **News coverage**: GDELT (free, global, 15-minute resolution) → use for `media` intensity windows
- **Sentiment**: run a free model (e.g. a HuggingFace sentiment model) over the posts you collected
- **Awareness / opinion**: any survey you or a public poll have

Calibrate on several past events, keep the parameters that work across all of them, and
backtest before trusting a forecast.

## Running free on GitHub

1. Push this folder to a public GitHub repo.
2. Actions tab → **simulate** → *Run workflow* (choose scenario, size, runs, extra `--set` args).
3. Download the `simulation-report` artifact, or enable Pages (Settings → Pages → Source:
   GitHub Actions) to get the report at `https://<you>.github.io/<repo>/`.

`test` runs a small smoke test on every push (`python tests/test_smoke.py` locally).

## Limits worth knowing

- Agents are rule-based cognitive models, not language models: a million LLM "brains" would
  cost serious money. The behaviour comes from well-studied mechanisms (bounded-confidence
  opinion dynamics, Daley–Kendall rumor spreading, Bass diffusion, agenda setting,
  continued-influence and illusory-truth effects), whose strengths are knobs in `DEFAULT_MODEL`.
- One opinion dimension (toward the brand/product). Multiple topics would need more state columns.
- One blended platform. Separate platforms = separate graphs and feeds.
- Tipping points (media pickup) make outcomes genuinely uncertain; always use `--runs ≥ 5`
  for decisions and read the bands, not just the median line.

## Ideas for next steps

- **LLM focus group, still free**: take 20–50 agents from `agents_sample.csv`, turn their
  traits + what they saw into a persona prompt, and have a local model via Ollama voice their
  reactions. Gives qualitative "why" on top of the quantitative "how many".
- Geography / languages / time zones as segments with their own diurnal curves.
- Multi-brand competition (opinion vector instead of scalar).
- Store-level or channel-level sales instead of a single adoption hazard.
