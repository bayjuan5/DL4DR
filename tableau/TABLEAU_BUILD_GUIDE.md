# Tableau Public Build Guide — DL4DR Benchmark Dashboard

Estimated time: 4–6 hours for a first Tableau build. Uses Tableau Public (free desktop app).
Each step names the Tableau skill it practises, which maps to the Analyst JD.

## 0. Setup
1. Install **Tableau Public** (desktop) and create a free account.
2. *Connect → Text file* → open `00_kpis.csv`. Then *Data → New Data Source* for each other CSV
   (keep them as separate sources; no joins needed).
3. On each source, check data types: `p_value` and `sign_test_p` as Number (decimal); `fold` as String
   (discrete); booleans as Boolean.

## 1. Calculated fields (create once, reuse)
| Source | Name | Formula | Skill |
|---|---|---|---|
| 01_model_metrics | `Is DL4DR` | `CONTAINS([Model], "DL4DR")` | Boolean calc |
| 01_model_metrics | `Headline` | `[Statistic] = "value" OR [Statistic] = "median"` | Filter calc |
| 02_paired_vs_dl4dr | `-log10 p` | `-LOG([P Value])` | Math calc |
| 02_paired_vs_dl4dr | `CI width` | `[Ci Hi] - [Ci Lo]` | Error bars |
| 04_cv_folds | `Gate gain (median R2)` | `[Median R2 Full] - [Median R2 Gate Off]` | Ablation delta |
| 05_zero_shot_by_tissue | `Lower` / `Upper` | `[Mean R2] - [Sd R2]` / `[Mean R2] + [Sd R2]` | Error bars |
| 06_gradcam_driver_genes | `Rank` | `RANK(SUM([N Cell Lines Top Driver]))` | Table calc |

Add a **parameter** `Metric` (string list: R2, RMSE, Pearson r, Spearman rho) and a calc
`Selected metric = [Metric] = [Metric (parameter)]` — drop it on Filters (True) in sheets 2–3 so one
dropdown switches every chart.

## 2. Worksheets

**Sheet 1 — KPI tiles** (`00_kpis`): *Text* mark; `Kpi` on Columns, `SUM(Value)` on Text. Format large font.

**Sheet 2 — The ranking flip (slope chart, the centrepiece)** (`01_model_metrics`)
- Filters: `Headline` = True, `Selected metric` = True.
- Columns: `Aggregation`; Rows: `Rank` (reverse the axis so rank 1 is on top). Mark: *Line*; `Model` on Detail.
- Dual axis with a *Circle* mark; `Model` on Label; `Is DL4DR` on Color (highlight DL4DR, grey the rest).
- Title: "Same models, same records — the winner depends on aggregation".

**Sheet 3 — Metric bars** (`01_model_metrics`): `Value` vs `Model`, columns split by `Aggregation`,
sorted descending; for per-cell-line, add `std` as a reference band or tooltip.

**Sheet 4 — Paired comparison forest plot** (`02_paired_vs_dl4dr`)
- Rows: `Comparator`; Columns: `Median Delta`. Mark *Circle*.
- Error bars: dual axis with *Gantt Bar* — start at `Ci Lo`, size = `CI width`.
- Reference line at 0. Color by `Favours`. Tooltip: wins/losses, p, effect size.

**Sheet 5 — Beats the baseline?** (`03_vs_additive_baseline`): stacked bar of `Wins` / `Losses` / `Ties`
per model (use *Measure Names*), or `Win Rate` bar with a 50% reference line.

**Sheet 6 — Fold stability** (`04_cv_folds`): `Fold` on Columns; *Measure Values* with
`Median R2 Full`, `Median R2 Gate Off`, `Pooled R2 Full`, `Pooled R2 Additive`; Line + circle marks.
Shows the gate-off ablation and fold-to-fold variation (time-series-style trend practice).

**Sheet 7 — Zero-shot by tissue** (`05_zero_shot_by_tissue`): horizontal bars `Mean R2` by `Tissue`,
sorted; error bars from `Lower`/`Upper`; size or label by `N Cell Lines`; color `Is Training Tissue`.

**Sheet 8 — GradCAM driver genes** (`06_gradcam_driver_genes`): bars of `N Cell Lines Top Driver`
by `Gene`; color by `Annotation` (red = known driver). Filter to top 20 with the `Rank` calc.

**Sheet 9 (optional) — Breast residual gain** (`07_breast_top10...`): dumbbell chart,
`R2 Memorization Only` → `R2 Residual Fusion` per `Cell Line`.

## 3. Dashboards (fixed size 1200 × 800)
- **Page 1 "Overview":** KPI tiles (top), slope chart (left, large), metric bars (right), `Metric` parameter control.
- **Page 2 "Head-to-head":** forest plot + baseline win-rate bars + fold stability.
- **Page 3 "Biology":** tissue zero-shot bars + driver-gene bars + breast dumbbell.
- Add a *Dashboard Action → Highlight* on `Model` so clicking a model highlights it across all sheets.
- Combine pages into a **Story** if you want a guided narrative.

## 4. Publish
1. Replace the approximate tissue values (see README) with exact ones.
2. *File → Save to Tableau Public*. Title: "DL4DR: Drug-Response Prediction on Unseen Cancer Cell Lines".
3. Paste the public link into `README.md` and the GitHub repo description; add a screenshot `dashboard.png`.
4. CV line (once it is live): *Tableau (Tableau Public portfolio: <link>)*.

## Talking points for an interview
- Why pooled and per-group metrics disagree (between-group vs. within-group variance) — directly analogous to
  team-level vs. game-level metrics in sports analytics, or store-level vs. company-level in business.
- Paired non-parametric testing (Wilcoxon, sign test), effect sizes, and multiple-comparison correction.
- Ablation as a controlled experiment; fold-to-fold stability as a robustness check.
