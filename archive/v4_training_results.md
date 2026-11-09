# V4 training results

This is the experiment behind V4. The current model is V5, run from `predict_matchup.py` and documented in the top-level `README.md`.

The script that produced these numbers is `archive/training/train_oliver_hybrid_v4.py`. Its outputs are in `archive/results/`: `oliver_hybrid_summary.json`, `oliver_model_forward_backtest.csv`, and `oliver_model_tournament_aggregate.csv`.

That test compared three ideas on the same 2022–2025 tournament games (268):

| Model | Winner accuracy | Margin miss |
|---|---:|---:|
| Oliver original, 40/25/20/15, with home court in that test | 63.43% | 11.51 |
| Eight learned Four Factor weights | 60.45% | 11.47 |
| Learned weights plus a V3 residual | 60.45% | 11.46 |

Oliver's 40/25/20/15 figures are approximate importance shares from *Basketball on Paper*. They were a benchmark in that test. They are not the formula V5 runs, and they are not a point-spread equation. The learned eight-weight model is the core that V5 kept, then refit with home court and schedule.

On the full 2022–2026 slate that older learned model was about 70% with a margin miss near 9.2 points. V5 added strength of schedule and moved the same forward test to 70.4% and 9.09, and the tournament rate from about 61% to 65.7%.
