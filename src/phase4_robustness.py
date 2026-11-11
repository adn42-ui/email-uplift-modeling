"""Phase 4, steps 1-3: re-test the Phase 2-3 findings on out-of-fold scores.

    python -m src.phase4_crossfit      (once; slow; writes the cache)
    python -m src.phase4_robustness    (this file; a few minutes of bootstrap)

Everything here uses all 42,693 women's-email + control customers, each
scored by a model that did not train on them (see phase4_crossfit.py).

Conventions
    * "Top X%" is ranked WITHIN each fold. Each fold's scores come from a
      different fitted model, so we let each model pick its own top X%
      rather than compare raw scores across models.
    * Policies are valued with IPW on actual spend, at the base-case margin
      and cost in config.py, exactly as in Phase 3.
    * Each of the 5 repeats gives one estimate. We report their mean, their
      range (how much the answer moves when only the split changes), and a
      bootstrap CI for the mean.
    * Bootstrap: customers are resampled (each arm separately) with their
      out-of-fold targeting decisions held fixed, and the 5 repeats are
      averaged inside each resample. It captures sampling noise in outcomes,
      NOT retraining noise; the range across repeats shows the latter.
"""
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from .data import load_hillstrom, two_arm_subset
from .evaluation import paired_bootstrap, summarize_draws, uplift_metrics
from .phase4_crossfit import OOF_PATH
from .policy import ipw_contributions, profit_per_1000
from .stats import diff_in_means

TABLES_DIR = cfg.REPORTS_DIR / "tables"
MODELS = {"Causal forest": "forest", "X-learner": "xlearner"}   # label -> column prefix
REPEATS = range(cfg.CV_REPEATS)
EVERYONE, RULE = "Email everyone", "Rule: skip men's-only buyers"


# ====================================================================== helpers
def rank_within_fold(score: pd.Series, fold: pd.Series) -> pd.Series:
    """Rank 1 = highest score in its fold. Ties broken by row order."""
    return score.groupby(fold).rank(ascending=False, method="first")


def top_share_within_fold(score: pd.Series, fold: pd.Series, share: float) -> np.ndarray:
    """True for the top `share` of each fold by score."""
    fold_size = fold.map(fold.value_counts())
    return (rank_within_fold(score, fold) <= np.round(share * fold_size)).to_numpy()


def lift(frame: pd.DataFrame, outcome: str) -> dict:
    """Emailed-minus-control difference in means inside `frame`."""
    return diff_in_means(frame.loc[frame.treated == 1, outcome].to_numpy(float),
                         frame.loc[frame.treated == 0, outcome].to_numpy(float))


def bootstrap_rows(t: np.ndarray, n_boot: int, seed: int):
    """Yield row indices for each resample, resampling each arm separately."""
    rng = np.random.default_rng(seed)
    treated, control = np.nonzero(t == 1)[0], np.nonzero(t == 0)[0]
    for _ in range(n_boot):
        yield np.r_[rng.choice(treated, len(treated)), rng.choice(control, len(control))]


# ====================================================================== 1. policies
def build_policies(df, oof, mens_only):
    """{policy name: list of 5 boolean masks (one per repeat)}."""
    n = len(df)
    rule_mask = ~mens_only
    policies = {EVERYONE: [np.ones(n, bool)] * cfg.CV_REPEATS, RULE: [rule_mask] * cfg.CV_REPEATS}
    for label, col in MODELS.items():
        for share in cfg.TARGET_SHARES:
            policies[f"{label}: top {share:.0%}"] = [
                top_share_within_fold(oof[f"{col}_{r}"], oof[f"fold_{r}"], share) for r in REPEATS]
        # The model given exactly as many emails as the rule sends.
        policies[f"{label}: top {rule_mask.mean():.1%} (rule's share)"] = [
            top_share_within_fold(oof[f"{col}_{r}"], oof[f"fold_{r}"], rule_mask.mean()) for r in REPEATS]
    return policies


def policy_table(df, policies) -> pd.DataFrame:
    t = df["treated"].to_numpy()
    g = ipw_contributions(df["spend"].to_numpy(float), t, t.mean())
    names = list(policies)

    # --- per-repeat point estimates
    per_repeat = {name: np.array([profit_per_1000((g * m).mean(), m.mean()) for m in masks])
                  for name, masks in policies.items()}
    shares = {name: np.mean([m.mean() for m in masks]) for name, masks in policies.items()}

    # --- bootstrap of the repeat-averaged value. Averaging the 5 repeats'
    # values equals valuing the "average mask" (how often each customer is
    # targeted across repeats), so one matrix product per resample suffices.
    avg_mask = np.column_stack([np.mean(policies[name], axis=0) for name in names])   # N x K
    weighted = g[:, None] * avg_mask
    draws = np.empty((cfg.N_BOOTSTRAP, len(names)))
    for b, rows in enumerate(bootstrap_rows(t, cfg.N_BOOTSTRAP, cfg.RANDOM_SEED)):
        draws[b] = profit_per_1000(weighted[rows].mean(axis=0), avg_mask[rows].mean(axis=0))
    draws = dict(zip(names, draws.T))

    def compare(name, other_point, other_draws, prefix):
        diff = per_repeat[name] - other_point
        out = {f"{prefix}": diff.mean(), f"{prefix}_repeat_min": diff.min(), f"{prefix}_repeat_max": diff.max()}
        out.update({f"{prefix}_{k}": v for k, v in summarize_draws(draws[name] - other_draws).items()})
        return out

    rows = [{"policy": "Email no one", "share_emailed": 0.0, "profit_per_1000": 0.0,
             **{f"vs_everyone{s}": v for s, v in [("", -per_repeat[EVERYONE].mean()),
                                                   *[(f"_{k}", x) for k, x in summarize_draws(-draws[EVERYONE]).items()]]}}]
    for name in names:
        row = {"policy": name, "share_emailed": shares[name], "profit_per_1000": per_repeat[name].mean(),
               "profit_repeat_min": per_repeat[name].min(), "profit_repeat_max": per_repeat[name].max(),
               **{f"profit_{k}": v for k, v in summarize_draws(draws[name]).items() if k != "boot_p"}}
        if name != EVERYONE:
            row.update(compare(name, per_repeat[EVERYONE], draws[EVERYONE], "vs_everyone"))
            # Random targeting at the same share = share x email-everyone, from the same resample.
            row.update(compare(name, shares[name] * per_repeat[EVERYONE], shares[name] * draws[EVERYONE], "vs_random"))
        if name not in (EVERYONE, RULE):
            row.update(compare(name, per_repeat[RULE], draws[RULE], "vs_rule"))
        rows.append(row)
    return pd.DataFrame(rows)


def rule_overlap(policies, mens_only) -> pd.DataFrame:
    """Is the model rediscovering the rule? Among the customers a model
    leaves out, what share bought only men's merchandise?"""
    rows = []
    for name, masks in policies.items():
        if name in (EVERYONE, RULE):
            continue
        excluded_mens_only = [mens_only[~m].mean() for m in masks]
        mens_only_excluded = [(~m)[mens_only].mean() for m in masks]
        rows.append({"policy": name, "share_excluded": 1 - np.mean([m.mean() for m in masks]),
                     "pct_of_excluded_who_are_mens_only": 100 * np.mean(excluded_mens_only),
                     "repeat_min": 100 * np.min(excluded_mens_only), "repeat_max": 100 * np.max(excluded_mens_only),
                     "pct_of_mens_only_who_are_excluded": 100 * np.mean(mens_only_excluded)})
    return pd.DataFrame(rows)


def subgroup_table(df) -> pd.DataFrame:
    """Actual lift by last year's purchase category. No model involved, and
    the grouping was fixed before looking at Phase 4 results."""
    group = np.select([(df.mens == 1) & (df.womens == 0), (df.mens == 0) & (df.womens == 1)],
                      ["Men's only", "Women's only"], "Both")
    rows, res = [], {}
    for name in ["Men's only", "Women's only", "Both", "Women's only or both"]:
        g = df[group != "Men's only"] if name == "Women's only or both" else df[group == name]
        for outcome in cfg.OUTCOMES:
            r = res[(name, outcome)] = lift(g, outcome)
            rows.append({"group": name, "n": len(g), "buyers_emailed": int(g.loc[g.treated == 1, "conversion"].sum()),
                         "buyers_control": int(g.loc[g.treated == 0, "conversion"].sum()), "outcome": outcome,
                         "control_mean": r["control_mean"], "treated_mean": r["treated_mean"], "lift": r["ate"],
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_value": r["p_value"]})
    from scipy import stats
    for outcome in cfg.OUTCOMES:   # is the lift different between the rule's two groups?
        a, b = res[("Women's only or both", outcome)], res[("Men's only", outcome)]
        diff, se = a["ate"] - b["ate"], np.hypot(a["se"], b["se"])
        rows.append({"group": "(Women's only or both) minus (Men's only)", "outcome": outcome, "lift": diff,
                     "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se,
                     "p_value": 2 * stats.norm.sf(abs(diff / se))})
    return pd.DataFrame(rows)


# ====================================================================== ranking quality (Phase 2 re-test)
def auuc_table(df, oof) -> pd.DataFrame:
    """AUUC vs. random on out-of-fold scores, for visit and for spend."""
    t = df["treated"].to_numpy()
    # Score = within-fold rank, flipped so higher = better, scaled to (0, 1].
    scores = {(label, r): (1 - (rank_within_fold(oof[f"{col}_{r}"], oof[f"fold_{r}"]) - 1)
                           / oof[f"fold_{r}"].map(oof[f"fold_{r}"].value_counts())).to_numpy()
              for label, col in MODELS.items() for r in REPEATS}
    rows = []
    for outcome in ("visit", "spend"):
        y = df[outcome].to_numpy(float)
        draws = paired_bootstrap(y, t, scores)
        point = {k: uplift_metrics(y, t, s) for k, s in scores.items()}
        mean_draws = {}
        for label in MODELS:
            per_repeat = np.array([point[(label, r)]["auuc_vs_random"] for r in REPEATS])
            mean_draws[label] = np.mean([draws[(label, r)]["auuc_vs_random"] for r in REPEATS], axis=0)
            rows.append({"outcome": outcome, "model": label, "auuc_random": point[(label, 0)]["auuc_random"],
                         "auuc_vs_random": per_repeat.mean(), "repeat_min": per_repeat.min(),
                         "repeat_max": per_repeat.max(), **summarize_draws(mean_draws[label])})
        a, b = list(MODELS)
        rows.append({"outcome": outcome, "model": f"{a} minus {b}",
                     "auuc_vs_random": rows[-2]["auuc_vs_random"] - rows[-1]["auuc_vs_random"],
                     **summarize_draws(mean_draws[a] - mean_draws[b])})
    return pd.DataFrame(rows)


# ====================================================================== 2. calibration
def calibration_table(df, oof) -> pd.DataFrame:
    """Predicted vs. actual uplift by decile of out-of-fold predicted uplift.

    Decile 10 = highest predicted uplift. Point estimates are averaged over
    the 5 repeats. The SE shown is the average single-repeat SE: the repeats
    reuse the same customers, so averaging them does not shrink the
    uncertainty much, and this choice errs on the wide side.
    """
    rows = []
    for label, col in MODELS.items():
        for r in REPEATS:
            fold, score = oof[f"fold_{r}"], oof[f"{col}_{r}"]
            pct = (rank_within_fold(score, fold) - 1) / fold.map(fold.value_counts())   # 0 = best
            decile = 10 - np.floor(pct * 10).astype(int)                                 # 10 = best
            for d in range(1, 11):
                g = df[decile == d]
                row = {"model": label, "repeat": r, "decile": d, "n": len(g),
                       "predicted_visit_uplift": score[decile == d].mean(),
                       "pct_mens_only": 100 * ((g.mens == 1) & (g.womens == 0)).mean(),
                       "buyers_emailed": g.loc[g.treated == 1, "conversion"].sum(),
                       "buyers_control": g.loc[g.treated == 0, "conversion"].sum()}
                for outcome in cfg.OUTCOMES:
                    res = lift(g, outcome)
                    row[f"actual_{outcome}_lift"], row[f"se_{outcome}"] = res["ate"], res["se"]
                rows.append(row)
    per_repeat = pd.DataFrame(rows)
    table = per_repeat.drop(columns="repeat").groupby(["model", "decile"], as_index=False).mean()
    for outcome in cfg.OUTCOMES:
        table[f"ci_low_{outcome}"] = table[f"actual_{outcome}_lift"] - 1.96 * table[f"se_{outcome}"]
        table[f"ci_high_{outcome}"] = table[f"actual_{outcome}_lift"] + 1.96 * table[f"se_{outcome}"]

    # Calibration slope: regress actual decile lift on predicted decile mean.
    # 1.0 = predictions are on the right scale; below 1 = the model spreads
    # customers further apart than they really are.
    slopes = (per_repeat.groupby(["model", "repeat"])
              .apply(lambda g: np.polyfit(g["predicted_visit_uplift"], g["actual_visit_lift"], 1)[0])
              .groupby("model").agg(["mean", "min", "max"]).reset_index())
    slopes.columns = ["model", "visit_calibration_slope", "repeat_min", "repeat_max"]
    return table, slopes


# ====================================================================== 3. stability
def stability_tables(oof):
    rows, freq_rows = [], []
    for label, col in MODELS.items():
        corr = [oof[f"{col}_{a}"].corr(oof[f"{col}_{b}"], method="spearman") for a, b in combinations(REPEATS, 2)]
        rows.append({"comparison": f"{label}: same customer, different repeats", "spearman_mean": np.mean(corr),
                     "spearman_min": np.min(corr), "spearman_max": np.max(corr)})
        in_top = np.column_stack([top_share_within_fold(oof[f"{col}_{r}"], oof[f"fold_{r}"], cfg.PROFILE_SHARE)
                                  for r in REPEATS])
        times = in_top.sum(axis=1)
        agree = [in_top[in_top[:, a], b].mean() for a in REPEATS for b in REPEATS if a != b]
        freq_rows.append({"model": label, **{f"pct_in_top30_{k}_of_5": 100 * (times == k).mean() for k in range(6)},
                          "pct_of_ever_top30_who_are_always_top30": 100 * (times == 5).sum() / (times > 0).sum(),
                          "pct_top30_in_one_repeat_also_top30_in_another": 100 * np.mean(agree)})
    cross = [oof[f"forest_{r}"].corr(oof[f"xlearner_{r}"], method="spearman") for r in REPEATS]
    rows.append({"comparison": "Causal forest vs. X-learner, same repeat", "spearman_mean": np.mean(cross),
                 "spearman_min": np.min(cross), "spearman_max": np.max(cross)})
    return pd.DataFrame(rows), pd.DataFrame(freq_rows)


# ====================================================================== figures
def plot_policies(table: pd.DataFrame, path):
    cfg.apply_plot_style()
    t = table[table.policy.ne(EVERYONE) & table.policy.ne("Email no one")].iloc[::-1].reset_index(drop=True)
    color = lambda p: (cfg.MODEL_COLORS["Causal forest"] if p.startswith("Causal") else
                       cfg.MODEL_COLORS["X-learner"] if p.startswith("X-") else cfg.INK)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axvline(0, color=cfg.INK_MUTED, lw=1)
    for i, r in t.iterrows():
        ax.plot([r["vs_everyone_ci_low"], r["vs_everyone_ci_high"]], [i, i], color=color(r["policy"]), lw=2)
        ax.scatter(r["vs_everyone"], i, s=55, color=color(r["policy"]), edgecolor=cfg.SURFACE, linewidth=1.5, zorder=3)
        ax.plot([r["vs_everyone_repeat_min"], r["vs_everyone_repeat_max"]], [i - 0.28, i - 0.28],
                color=cfg.INK_MUTED, lw=1, marker="|", ms=5)
    ax.set_yticks(range(len(t)), t["policy"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Profit per 1,000 customers minus email-everyone ($)")
    ax.set_title(f"Margin {cfg.MARGIN:.0%}, ${cfg.COST_PER_EMAIL:.2f} per email, all 42,693 customers, out-of-fold scores.\n"
                 "Dot + bar: mean of 5 repeats, 95% bootstrap CI. Grey bar: range across repeats.",
                 fontsize=9, fontweight="normal", color=cfg.INK_MUTED)
    fig.suptitle("Does any targeting policy beat emailing everyone?", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(cal: pd.DataFrame, ate: dict, path):
    cfg.apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, (outcome, scale, unit) in zip(axes, [("visit", 100, "percentage points"), ("spend", 1, "$ per customer")]):
        ax.axhline(0, color=cfg.INK_MUTED, lw=0.8)
        ax.axhline(ate[outcome] * scale, color=cfg.INK_MUTED, lw=1.2, ls=":", label="Overall ATE")
        for (label, dx) in zip(MODELS, (-0.15, 0.15)):
            c = cal[cal.model == label]
            col = cfg.MODEL_COLORS[label]
            ax.errorbar(c["decile"] + dx, c[f"actual_{outcome}_lift"] * scale, yerr=1.96 * c[f"se_{outcome}"] * scale,
                        fmt="o", color=col, ms=5, lw=1.3, capsize=2, label=f"{label}: actual lift (95% CI)")
            if outcome == "visit":
                ax.plot(c["decile"] + dx, c["predicted_visit_uplift"] * scale, color=col, lw=1.5, ls="--",
                        label=f"{label}: predicted")
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Decile of predicted visit uplift (10 = highest)")
        ax.set_ylabel(f"Emailed minus control ({unit})")
        ax.set_title(f"Actual lift in {outcome}")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Calibration by out-of-fold decile (mean of 5 repeats)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ====================================================================== main
def save(table: pd.DataFrame, name: str, cols=None):
    table.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    print(f"\n=== {name} ===")
    print((table[cols] if cols else table).round(4).to_string(index=False))


def main():
    df = two_arm_subset(load_hillstrom(), cfg.WOMENS)
    oof = pd.read_csv(OOF_PATH, index_col="row")
    assert oof.index.equals(df.index), "cached scores do not line up with the data"
    mens_only = ((df.mens == 1) & (df.womens == 0)).to_numpy()
    print(f"customers={len(df):,}  buyers={int(df.conversion.sum())}  men's-only={mens_only.mean():.4f}")

    policies = build_policies(df, oof, mens_only)
    table = policy_table(df, policies)
    save(table, "phase4_policies", ["policy", "share_emailed", "profit_per_1000", "profit_ci_low", "profit_ci_high",
                                    "vs_everyone", "vs_everyone_repeat_min", "vs_everyone_repeat_max",
                                    "vs_everyone_ci_low", "vs_everyone_ci_high", "vs_everyone_boot_p"])
    print(table[["policy", "vs_random", "vs_random_ci_low", "vs_random_ci_high", "vs_random_boot_p",
                 "vs_rule", "vs_rule_repeat_min", "vs_rule_repeat_max", "vs_rule_ci_low", "vs_rule_ci_high",
                 "vs_rule_boot_p"]].round(3).to_string(index=False))
    save(rule_overlap(policies, mens_only), "phase4_rule_overlap")
    save(subgroup_table(df), "phase4_subgroups")
    plot_policies(table, cfg.FIGURES_DIR / "phase4_policies.png")

    cal, slopes = calibration_table(df, oof)
    save(cal, "phase4_calibration", ["model", "decile", "n", "predicted_visit_uplift", "actual_visit_lift", "se_visit",
                                     "actual_conversion_lift", "actual_spend_lift", "se_spend", "pct_mens_only",
                                     "buyers_emailed", "buyers_control"])
    save(slopes, "phase4_calibration_slopes")
    ate = {o: lift(df, o)["ate"] for o in cfg.OUTCOMES}
    plot_calibration(cal, ate, cfg.FIGURES_DIR / "phase4_calibration.png")

    stability, freq = stability_tables(oof)
    save(stability, "phase4_stability")
    save(freq, "phase4_top30_frequency")

    save(auuc_table(df, oof), "phase4_auuc")   # slowest step, last


if __name__ == "__main__":
    pd.set_option("display.width", 300, "display.max_columns", 40)
    main()
