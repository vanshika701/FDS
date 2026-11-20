# Decision & bug log

Why the project is built the way it is, and what went wrong along the way.
Each entry: **what** was decided → **why** → **trade-off**.

## Design decisions

### D1. One common table format for every dataset
**What:** every loader outputs `(entity_id, timestamp, demand_mbps)`.
**Why:** after that point, features, models, simulator and metrics are identical for all
four datasets, so differences in results come from the data, not from code paths.
**Trade-off:** dataset-specific detail (e.g. Milan's SMS/call columns) is thrown away.

### D2. Synthetic stand-ins first, real data second
**What:** each dataset has a synthetic generator copying its traffic *shape* (daily cycle,
burstiness, weekend effect, time zone), plus injected missing values and spikes.
**Why:** raw data is 20+ GB (Milan alone is 62 × ~330 MB), CAIDA needs a formal access
request, and the laptop had little free disk. The pipeline can be built and tested now;
switching to real data is a one-line config change (`source: real`).
**Trade-off:** synthetic results show the *method* works; they are not evidence about real
networks. Every output records its data source, and the report must say so.

### D3. Raw data streamed one file at a time (Colab or local)
**What:** `scripts/prepare_real_data.py` downloads a raw file, keeps only the selected users,
deletes the raw file, and moves on. `notebooks/colab_data_prep.ipynb` runs the same script.
**Why:** peak disk use is one raw file (~370 MB), not 20 GB.

### D4. Rows indexed by the *target* window
**What:** row τ holds y = demand(τ) and features computed from windows ≤ τ−1.
**Why:** this makes leakage impossible by construction, and `tests/test_features.py` checks it:
changing all data at τ and later leaves every feature of row τ unchanged.

### D5. `entity_avg` is an expanding mean, peak hours come from fold 1 only
**What:** stricter than the proposal ("average in the training period").
**Why:** a single "training-period average" over the whole 85% would leak validation-fold
information into earlier walk-forward folds. An expanding mean never sees the future.

### D6. For hourly data, `mean_1h`/`std_1h` use 3 windows
**Why:** with 1-hour windows, a 1-hour mean is just `lag_1`, and a 1-sample std is undefined.

### D7. One pooled model per dataset (not one per user)
**Why:** the proposal's choice: fast with hundreds of users, and users share daily patterns.

### D8. Folds split on unique timestamps, not rows
**Why:** all users of a window must land on the same side of a split; otherwise the model
could train on user A at 10:00 and be validated on user B at 10:00.

### D9. Cheaper tuning, full refit
**What:** the RF grid (3 depths × 2 leaf sizes) is tuned with 60 trees on ≤30k rows; the chosen
config is refit with 200 trees on ≤60k rows.
**Why:** the full grid took ~12 min for Milan alone (25 s per fit × 30 fits).
Relative ranking of configs is stable at 60 trees.
**Trade-off:** slightly noisier selection; reported here.

### D10. HistGradientBoosting instead of GradientBoostingRegressor
**Why:** same algorithm family, histogram-based, ~20× faster on 10⁵ rows. Settings kept from
Section 3.7 (300 iterations, learning rate 0.05, depth 3).

### D11. Capacity from the development period
**What:** C = 0.7 × peak total demand of the *non-holdout* data.
**Why:** setting C from holdout peaks would use future information.

### D12. Headroom ablation (not in the proposal)
**What:** Eq. 2 (the predictive allocator) always hands out all of C, while Eq. 1
(Proportional Share) caps each user at the demand it reported. That is two differences at
once: *forecast vs last value* **and** *use all of C vs not*. We add "Reactive Prop. +
headroom" (Eq. 2 fed with last-window demand) to separate them.
**Why:** without it, the gain of the predictive method would be wrongly credited entirely to
the forecast. `results/gain_decomposition.csv` reports both parts.

### D13. Oracle = Proportional on true demand (Max-Min on true demand kept as a 2nd reference)
**What:** the upper-bound benchmark hands out `a_i = r_i · min(1, C / Σr)` using the true
next-window demand.
**Why:** our fairness metric is Jain's index on *satisfaction* `s_i = a_i / r_i`. Proportional
on the truth gives every user the same satisfaction, so Jain = 1 exactly, and it uses as much
of the link as the demand allows. Max-Min on the truth protects small users but leaves big
users less satisfied, so it is *not* the optimum for this metric (see bug B5).

### D16. "Stranded capacity" as the real waste metric
**What:** `stranded = min(over-allocation, unmet demand) / C` per window.
**Why:** plain "over-allocation" counts capacity given to someone who did not need it, even
when nobody else needed it either. Nothing was lost then. Stranded capacity counts only
capacity that sat idle *while another user went short* (see bug B6).

### D14. Cross-dataset errors in scale-free units
**What:** each user's demand is divided by its own development-period mean before
cross-dataset training/testing; error is reported in those units.
**Why:** Milan values are activity counts, MAWI/CAIDA are backbone Mbps, FCC is home Mbps.
Raw MAE across them is meaningless; relative error is comparable.

### D17. Real MAWI via agurim aggregate flows, not raw pcaps
**What:** `scripts/fetch_mawi_agurim.py` downloads 42 days of MAWI's public *agurim* service:
the significant aggregate flows on the WIDE backbone (traffic grouped by address prefix),
at 10-minute resolution, one request per day.
**Why:** one raw MAWI capture is ~24 GB for only 15 minutes. That is too big to process here, and
15 minutes is too short to learn daily patterns. agurim covers the same backbone continuously.
**Trade-off:** an entity is an aggregate flow, not a single IP. agurim only lists a flow on days
when it is ≥1% of traffic, and its prefixes nest/regroup from day to day. With the standard
"drop users missing >10%" rule, **9 stable flows** remain (277 seen in total). We kept the rule
instead of loosening it, because filling day-long gaps would mean inventing data.

### D18. Extension forecasters scored on identical rows, and through the allocator
**What:** SARIMA (per user, daily season, log1p, parameters fitted on development data then
applied with fixed parameters) and a pooled 1-layer LSTM (24-window input, early stopping on the
last 15% of development data) predict exactly the holdout windows the main models predict.
Each forecaster is then also fed into the Predictive allocator.
**Why:** the proposal's question is not "which model has the lowest error" but "does a better
forecast give a fairer link?". Scoring both on the same rows answers that directly.
SARIMA is skipped for 10-minute data (a 144-step season is impractically slow), as the proposal
anticipated.

### D15. Chart colors follow the method, not its rank
**What:** one fixed, colorblind-validated palette in `src/bwalloc/style.py`, shared by all
figures and the dashboard; marker shape encodes the family (traditional / proposed /
benchmark / ablation).

## Bugs found and fixed

| # | Bug | How it was found | Fix |
|---|-----|------------------|-----|
| B1 | Synthetic seed used `hash(name)`, which Python randomizes per process, so runs were not reproducible | Code review before first run | Fixed per-profile offset `index(name) * 1000` |
| B2 | First CV run did not finish within 10 min | Timed out | Profiled one fit (25 s), led to D9 |
| B3 | First color palette put blue next to violet: ΔE 10.5, fails even for normal vision | Ran the palette validator instead of eyeballing | Switched to the validated fixed slot order |
| B4 | `Series.dt.dayofweek` deprecated in pandas 3 | Test warnings | `day_of_week` |
| B5 | "Oracle" (Max-Min on true demand) scored *lower* than the Predictive method on Milan (Jain 0.997 vs 1.000), so it was not an upper bound | Sanity-checked results: the benchmark must never lose | Oracle redefined as Proportional-on-truth (D13); new unit test proves Jain = 1 for it and < 1 for Max-Min-on-truth |
| B6 | Predictive looked very "wasteful" (62% of C over-allocated on FCC) although nobody was hurt by it | Number did not match the lower unmet demand | Added stranded capacity (D16) + unit test |
| B7 | `import bwalloc` suddenly failed after a successful run | Traceback in `make_tables.py` | macOS set the `hidden` flag on the editable-install `.pth` (Desktop is iCloud-synced), and Python 3.13+ skips hidden `.pth` files. Scripts now add `src/` themselves (`scripts/_bootstrap.py`), pytest uses `pythonpath = ["src"]` |
| B9 | Milan download got HTTP 403 on a plain metadata request | Traceback line pointed at `milan_file_list()`; curl worked | Harvard's firewall blocks Python's default `Python-urllib` User-Agent. Requests now send a generic client name |
| B10 | Milan files need a guestbook response before download | API returned "required Guestbook response" | Script POSTs the guestbook (email from an env var the user sets) and downloads from the signed URL |
| B11 | Latent: if `mean_1h` were dropped (short trace), `pct_change_1h` would be all-NaN and delete every row | Code review while adding real packet data | Drop them together; test with a 15-minute 1-min-window trace |
| B12 | MAWI fetcher crashed at start: bare `%` in an argparse help string | Traceback | Escaped as `%%` |
| B8 | Figure: the green diamond was drawn over the blue one, so the "Predictive (Prop.)" label pointed at the wrong color | Looked at the rendered PNG | Most important methods drawn last (on top) |
