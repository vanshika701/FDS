# CSD355 – Prediction-Based Bandwidth Allocation: Project Checklist

Source of truth for scope: `CSD355 - Foundation of Data Science Project.docx`.
Each item is worth 1 point. Progress = checked / total.

**Order of work:** all ML and experiment work (Phases 0–6, including real data) is finished
first. The dashboard (Phase 7) is built only after that, on the final results. The write-up comes last.

**Storage note:** raw datasets (Milan, MAWI, CAIDA, FCC) are multi-GB, so they are downloaded
and preprocessed **in Google Colab** (`notebooks/colab_data_prep.ipynb`), or streamed one day at a
time locally. Only the small processed `(entity_id, timestamp, demand)` files are kept. Until then,
the pipeline runs on **synthetic stand-ins** that copy each dataset's traffic shape. These are
clearly labeled everywhere.

---

## ⏰ Reminders – things only you can do
- [x] ~~Run the real Milan download~~ → done: 19 days (2013-11-04..22) of 150 cells. Stopped early because Harvard's server slowed to ~200 KB/s (~5 h for the rest); resumable with the same command if you ever want 28 days.
- [x] ~~Download FCC Measuring Broadband America data~~ → fcc.gov links unreachable; **replaced by CESNET-TimeSeries24** (real per-IP ISP traffic, downloaded automatically from Zenodo). SNAP was checked too: only graphs, no traffic volumes.
- [ ] *(Optional)* Request CAIDA anonymized-trace access (needs approval; caida.org → Data → request). Until then CAIDA stays synthetic.

## Phase 0 – Setup
- [x] 1. Python virtual env + `requirements.txt`
- [x] 2. Project structure (`src/bwalloc`, `scripts`, `dashboard`, `notebooks`, `tests`, `results`)
- [x] 3. Central `config.yaml` (Section 3.7 settings: C=70%, n users, seed 42, RF/GB params, 5 splits, 15% holdout)

## Phase 1 – Data (Section 3.2)
- [x] 4. Common schema `(entity_id, timestamp, demand_mbps)` + validation
- [x] 5. Synthetic generators for 4 dataset profiles (Milan mobile, MAWI backbone, CAIDA backbone, FCC home)
- [x] 6. Real-data loaders (Milan txt, MAWI/CAIDA pcap-aggregated CSV, FCC CSV)
- [x] 7. Preprocessing: UTC + sort, resample to window, gap fill ≤2 windows, drop users >10% missing
- [x] 8. Spike clipping at 99.9th pct (fit on train only) + fixed user selection (50–200)
- [x] 9. Colab notebook to download + aggregate real datasets into small processed files

## Phase 2 – Features (Section 3.3)
- [x] 10. Lag features (lag_1/2/3, lag_24h, lag_7d)
- [x] 11. Rolling features (mean_1h, std_1h, mean_6h, max_6h, mean_24h)
- [x] 12. Trend, time (cyclic hour), context (total_load, share_of_total, entity_avg from train only)
- [x] 13. Leakage-safety unit test (no feature uses t+1 or later)

## Phase 3 – Prediction models (Sections 3.4–3.5)
- [x] 14. Baselines: Last Value, 1-hour Moving Average
- [x] 15. Linear Regression (StandardScaler fit on train), Random Forest, Gradient Boosting
- [x] 16. Walk-forward CV (TimeSeriesSplit, 5 folds) on first 85%, MAE/RMSE/% improvement vs last value
- [x] 17. Model selection + final fit + one-time evaluation on 15% holdout
- [x] 18. Cross-dataset 4×4 error matrix

## Phase 4 – Allocation simulator (Section 3.6)
- [x] 19. Round Robin, Max-Min Fairness (water-filling), Proportional Share
- [x] 20. Predictive Proportional + Predictive Max-Min + Oracle
- [x] 21. Metrics: Jain's index on satisfaction, utilization, unmet demand, starvation, time per window
- [x] 22. Allocator unit tests (capacity never exceeded, correctness on toy cases)
- [x] 23. Simulation over holdout windows for all datasets × methods

## Phase 5 – Experiments & analysis
- [x] 24. Capacity sweep (C = 50%…90% of peak)
- [x] 25. Forecast-noise sensitivity (Jain vs noise level)
- [x] 26. One-command pipeline `scripts/run_pipeline.py` saving all results to `results/`
- [x] 27. Static figures (fairness vs utilization, noise curve, capacity sweep, error matrix, feature importance)
- [x] 28. Comparison table of all parameters, traditional vs proposed (Required add-on 3), incl. headroom ablation

## Phase 6 – Real data & ML extensions
- [x] 29. Run real Milan data (Colab or streamed locally); plug into pipeline — script ready & tested, done, 19 days real (see Reminders) (needs your guestbook email)
- [x] 30. Real MAWI (42 days of agurim backbone flows, 9 stable flows) + real CESNET (replaces FCC; 100 research-ISP IPs, 6 weeks). CAIDA → see Reminders
- [x] 31. Re-run full pipeline + figures on real data (done for real MAWI; done for real MAWI + CESNET; final re-run once Milan finishes downloading)
- [x] 32. Optional extension: LSTM/SARIMA comparison (Section 3.4 "optional")

## Phase 7 – Dashboard (Required add-on 2), built on final results
- [ ] 33. Streamlit dashboard: parameters, data overview, model results
- [ ] 34. Dashboard: allocation comparison, trade-off plots, sensitivity, cross-dataset matrix
- [ ] 35. Dashboard: "Traditional vs Proposed" comparison + interactive what-if simulator

## Phase 8 – Write-up & final polish
- [ ] 36. Section: where traditional methods fall short + how the predictive approach addresses it (backed by our numbers)
- [ ] 37. Honest limitations section (synthetic vs real, where prediction does NOT help)
- [ ] 38. README with setup, Colab instructions, how to run, results summary
- [ ] 39. Final report document / slides for the professor
- [ ] 40. Final review: tests pass, code cleanup, reproducibility check from clean clone

---

**Progress: 32 / 40 (80%)** 