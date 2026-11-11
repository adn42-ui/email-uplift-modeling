# Who should get the email? Uplift modelling on the Hillstrom experiment

An end-to-end causal inference project built around a targeting case study.

## The question

An online retailer randomly sent 64,000 customers a men's merchandise email,
a women's merchandise email, or no email.
**How should the retailer use its two emails, and does targeting beat
emailing everyone?**

## The headline answer

For the women's email, no targeting policy beats emailing everyone: an
apparent win from one test set (+$38 per 1,000 customers) did not replicate
under cross-fitting (−$9, 95% CI [−$28, +$7]). The uplift models mostly
rediscovered one rule, that customers who bought only men's merchandise
barely respond to the women's email. The larger finding is that the men's
email works about equally for every segment, and sending it to everyone has
the highest estimated profit ($181 vs. $77 per 1,000 customers at an assumed
30% margin and $0.05 per email); that gap is suggestive on spend (p = 0.031),
strong on visits and conversions, and may reflect that particular creative
and not the department.

## Reports

| Report | What it covers |
|---|---|
| **[Final report](reports/final_report.md)** | The whole story in interview order: question, metric, method, result, caveats |
| [Interview prep](reports/interview_prep.md) | 60-second summary and 10 follow-up questions |
| [Phase 1: EDA](reports/phase1_eda.md) | Randomization checks, average treatment effects |
| [Phase 2: Uplift models](reports/phase2_uplift.md) | S-, T-, X-learner and causal forest; Qini, uplift curves, AUUC |
| [Phase 3: Policy](reports/phase3_policy.md) | Profit by targeting policy on one test set; persuadables |
| [Phase 4: Robustness](reports/phase4_robustness.md) | Cross-fitted re-test, calibration, stability, why it is causal, limitations |
| [Phase 4b: Three-arm decision](reports/phase4b_three_arm.md) | Men's vs. women's email by segment, no ML |

Phases 2–3 used a single 30% test set. Phase 4 re-tested them on all
customers; where they disagree, Phase 4 is the result to quote.
[PROGRESS.md](PROGRESS.md) has a scorecard of what replicated.

## How to reproduce

Python 3.13. One command per phase, run from the project root, in this order
(Phase 3 reads scores saved by Phase 2; the Phase 4 analysis reads the cache
written by the cross-fitting step).

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m src.phase1_eda          # Phase 1
python -m src.phase2_uplift       # Phase 2  (about 1 minute)
python -m src.phase3_policy       # Phase 3
python -m src.phase4_crossfit     # Phase 4, step 1: out-of-fold scores (about 6 minutes)
python -m src.phase4_robustness   # Phase 4, step 2: re-test, calibration, stability (about 4 minutes)
python -m src.phase4_simulation   # Phase 4, step 3: confounding simulation
python -m src.phase4b_three_arm   # Phase 4b
```

- The dataset downloads automatically on first run into `data/`.
- Results are seeded (`RANDOM_SEED` in `src/config.py`).
- Margin, cost per email and targeting cut-offs are assumptions set in
  `src/config.py`.
- On Windows, if the project sits in a deeply nested folder, create the
  virtual environment at a short path (pip can hit the 260-character limit).

## Data

[Hillstrom MineThatData E-Mail Analytics challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
(2008). 64,000 customers who purchased in the previous 12 months, randomized
one third each to the three arms. Outcomes cover the following two weeks.

| Column | Meaning |
|---|---|
| `recency` | Months since last purchase (1–12) |
| `history` | Dollars spent in the past year |
| `history_segment` | Binned `history` (redundant with `history`) |
| `mens`, `womens` | 1 if the customer bought men's / women's merchandise in the past year |
| `zip_code` | Urban, Suburban, Rural |
| `newbie` | 1 if a new customer in the past 12 months |
| `channel` | Channel(s) purchased from in the past year: Phone, Web, Multichannel |
| `segment` | **Treatment arm**: Mens E-Mail, Womens E-Mail, No E-Mail |
| `visit`, `conversion`, `spend` | **Outcomes**: visited, purchased, dollars spent |

## Project structure

```
src/
  config.py             paths, column names, seed, business assumptions, plot colours
  data.py               load the dataset, build feature matrices
  stats.py              difference in means, bootstrap CI, regression adjustment, SMD
  learners.py           S-, T-, X-learner (hand-written) and causal forest, each explained in its docstring
  evaluation.py         uplift / Qini curves, AUUC, paired bootstrap
  policy.py             value a targeting policy from randomized outcomes (IPW)
  phase1_eda.py         Phase 1: balance checks and average treatment effects
  phase2_uplift.py      Phase 2: fit the models, evaluate on a held-out test set
  phase3_policy.py      Phase 3: profit by policy, sensitivity grid, persuadables
  phase4_crossfit.py    Phase 4: repeated 5-fold out-of-fold scores (slow; cached)
  phase4_robustness.py  Phase 4: policy re-test, calibration, stability
  phase4_simulation.py  Phase 4: confounding demonstration on simulated observational data
  phase4b_three_arm.py  Phase 4b: men's vs. women's email by purchase segment
reports/
  final_report.md, interview_prep.md, phase*.md
  figures/              generated figures
  tables/               generated tables (CSV)
data/                   dataset cache (git-ignored)
cache/                  out-of-fold scores (git-ignored)
requirements.txt        pinned versions
PROGRESS.md             status, replication scorecard, agreed modelling rules
```

## Method notes

- Features are only the eight pre-treatment covariates; outcomes are never
  used as features.
- The propensity is known by design, so the X-learner, the causal forest and
  all IPW estimates use the actual treated share, not a fitted model.
- Policies are valued from actual randomized outcomes, never from a model's
  own predictions.
- Model-vs-model and policy-vs-policy comparisons use paired bootstrap
  intervals on the same resamples.
