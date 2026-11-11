# Progress and carry-forward notes

## Status

| Phase | Status |
|---|---|
| 1. Setup, randomization check, baseline ATEs | Done ([report](reports/phase1_eda.md)) |
| 2. Uplift modelling | Done ([report](reports/phase2_uplift.md)) |
| 3. Business decision | Done ([report](reports/phase3_policy.md)); 3-arm extension skipped for now |
| 4. Robustness | Done ([report](reports/phase4_robustness.md)) |
| 4b. Three-arm decision (no ML) | Done ([report](reports/phase4b_three_arm.md)) |
| 5. Write-up | Done ([final report](reports/final_report.md), [interview prep](reports/interview_prep.md)) |

All phases are complete. The one deliberate gap: the "Recommendation"
section of the final report is a TODO for the author.

## Scorecard: which findings survived Phase 4

Phase 4 re-tested everything on all 42,693 women's-email + control customers
with repeated 5-fold cross-fitting (5 × 5). **Where Phases 2–3 and Phase 4
disagree, quote Phase 4.**

| # | Earlier finding | Phase 4 result | Verdict |
|---|---|---|---|
| P1 | Randomization holds; both emails raise visit, conversion, spend on average | Full-sample, not split-dependent | Unchanged |
| P2-a | All models rank visit uplift better than random | AUUC vs. random: forest +0.0068 [0.0051, 0.0085], X-learner +0.0073 [0.0056, 0.0090]; about 30% more area than random | **Replicated** |
| P2-b | Forest and X-learner are statistically tied | AUUC difference −0.0005 [−0.0011, 0.0002], p = 0.16 | **Replicated** |
| P2-c | Visit-trained models also rank incremental revenue better than random (p ≈ 0.04 for two of four) | Forest +0.034 [−0.040, 0.106], p = 0.34; X-learner p = 0.23 | **Did not replicate** |
| P3-1 | Top 90% beats email everyone by $38 per 1,000 | −$9 [−$28, $7] (forest); −$12 [−$36, $10] (X-learner) | **Did not replicate** |
| P3-2 | Top 30/50/70% indistinguishable from email everyone | All intervals include zero; point estimates −$38 to $0 | **Replicated** |
| P3-3 | Bottom 10% (men's-only) has a negative spend lift | Bottom decile +$0.48 (SE $0.41); men's-only group +$0.28 [−$0.12, $0.68] | **Did not replicate** |
| P3-4 | Model beats random targeting at every share in point estimate | +$8 to +$31 at 30/50/70%, about zero at 90%; no interval excludes zero | **Weakened** |
| P3-5 | Persuadables are defined by product affinity | Deciles 1–4 are ~100% men's-only, deciles 6–10 are 0% | **Replicated** |
| P3-6 | Top 30% has a significant spend lift; top-minus-bottom nearly significant | No decile ordering in spend; women's-or-both minus men's-only spend lift +$0.27, p = 0.32 | **Weakened** |
| P3-7 | Email everyone beats no one at the base case ($77 per 1,000) | $77; bootstrap CI [−$4, $156], p = 0.06 (analytic CI [$1, $154]) | Unchanged: **borderline** |

New in Phase 4:

- **The models mostly rediscover one rule.** At the rule's email volume
  (54.9%), 99.4% of customers the forest excludes are men's-only buyers
  (X-learner 97.7%). Model minus rule: forest +$5 [−$1, $12]; X-learner +$9
  [$2, $18] per 1,000.
- **Neither the rule nor any model policy beats email everyone.** Rule:
  −$14 [−$70, $42].
- **Visit uplift has essentially two levels** (women's-merchandise buyers
  about +7.3 pp, men's-only about +1.1 pp). Within each level the model's
  ranking shows no signal. Calibration slope 0.83 (forest), 0.95 (X-learner).
- **Rankings are stable overall** (Spearman 0.91 forest, 0.86 X-learner
  across repeats) **but top-30% membership is not** (80% / 73% agreement
  between repeats).
- **Simulation:** with assignment confounded by history, naive visit ATE is
  39% too high; IPW with history is within 2%; IPW without history is still
  13% too high. On spend the simulation is too noisy to show the
  unmeasured-confounder failure.

## Phase 4b findings (all 64,000 customers, three arms, pre-specified segments)

- **The men's email works about equally in every segment** (visit lift +6.9
  pp men's-only, +7.1 pp women's-only, +13.4 pp both). Its effect does not
  differ between men's-only and women's-only buyers (p = 0.84 visit, 0.81
  spend).
- **The women's email is the one that depends on purchase history** (+7.4 pp
  women's-only vs. +1.1 pp men's-only).
- **Base-case profit per 1,000:** women's to everyone $77 [$1, $153]; men's
  to everyone $181 [$98, $264]; matched rule $131 [$55, $211]; best per
  segment (optimistic) $184.
- **Men's-to-everyone minus women's-to-everyone: +$104 [$8, $198],
  p = 0.031.** Suggestive on spend (0.093 after correcting for three policy
  comparisons); strong on visits and conversions.
- **Matched rule is not established as better or worse** than either
  single-email policy (+$54 vs. women's, p = 0.097; −$50 vs. men's,
  p = 0.17). The "match the email to purchase history" hypothesis from
  Phase 4 is not supported.
- **Winning pre-specified policy: men's email to everyone.** Caveat: the two
  emails are different creatives, so this may be about this particular
  email, not the department.

## Phase 5 decisions (as written)

- Next experiment: confirm men's-to-everyone in a fresh randomized test,
  with a longer outcome window, unsubscribes tracked, and if possible a
  design that separates creative from department. Rough sizing: about
  36,000 per arm for the observed men's-vs-women's spend gap.
- Final report question: "How should the retailer use its two emails, and
  does targeting beat emailing everyone?" Story told in the order it
  happened.

- Structure: clarify the question → metric → method → result →
  recommendation → caveats.
- Leave the "Recommendation" section as a TODO with bullet-point findings
  (the user writes it).
- "Interview prep" section: 10 likely follow-up questions, no answers.
- Headline numbers should be the Phase 4 ones. The Phase 3 top-90% result
  should be told as a cautionary tale, not a result.

## Modelling rules agreed for all phases

- Features are only the 8 pre-treatment covariates; never visit, conversion
  or spend.
- Known propensity (treated share) for the X-learner and causal forest.
- Paired bootstrap for all policy-vs-policy and model-vs-model comparisons;
  random baseline computed inside each resample.
- Policies are valued from actual randomized outcomes (IPW), never from
  model predictions.
- Spend is revenue, not profit: profit = margin × incremental revenue −
  cost per email × emails. Base case margin 30%, cost $0.05, both
  assumptions set in `src/config.py`.

## Environment notes

- The virtual environment lives outside the project at
  `C:\Users\akach\.venvs\hillstrom-uplift` (Windows path-length limit).
- `cache/phase4_oof_scores.csv` holds the out-of-fold scores (about 6
  minutes to regenerate with `python -m src.phase4_crossfit`).
