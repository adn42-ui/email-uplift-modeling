"""Phase 3: turn uplift scores into a targeting decision and value it.

Run from the project root (after phase 2, whose saved test-set scores it reuses):

    python -m src.phase3_policy

Business assumptions (margin, cost per email, cut-offs) live in src/config.py.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from .data import covariate_matrix, load_hillstrom, two_arm_subset
from .evaluation import summarize_draws
from .policy import (EVERYONE, bootstrap_policies, ipw_contributions, policy_incremental_revenue,
                     profit_per_1000, top_share_mask)
from .stats import diff_in_means, standardized_mean_difference

TABLES_DIR = cfg.REPORTS_DIR / "tables"


def load_test_set() -> pd.DataFrame:
    """Phase 2's held-out customers: saved scores + outcomes, joined back to covariates."""
    preds = pd.read_csv(TABLES_DIR / "phase2_test_predictions.csv", index_col="row")
    covariates = two_arm_subset(load_hillstrom(), cfg.WOMENS)
    test = preds.join(covariates[cfg.PRE_TREATMENT_COVARIATES])
    # Guard against a mis-aligned join: outcomes saved by Phase 2 must match the source rows.
    assert (test["spend"] == covariates.loc[test.index, "spend"]).all()
    return test


# ====================================================================== tables
def revenue_table(point: dict, draws: dict, y, t, scores: dict) -> pd.DataFrame:
    """Incremental revenue per 1,000 customers for every policy (no margin or cost yet)."""
    rows = [{"policy": "Email no one", "share_emailed": 0.0, "inc_revenue_per_1000": 0.0,
             "ci_low": 0.0, "ci_high": 0.0}]
    for key in point:
        share = 1.0 if key == EVERYONE else key[1]
        label = EVERYONE if key == EVERYONE else f"{key[0]}: top {share:.0%}"
        ci = summarize_draws(1000 * draws[key])
        row = {"policy": label, "share_emailed": share, "inc_revenue_per_1000": 1000 * point[key],
               "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]}
        # Cross-check with the other valid estimator: plain difference in mean
        # spend (emailed - control) INSIDE the targeted group, times its share.
        mask = np.ones(len(y), bool) if key == EVERYONE else top_share_mask(scores[key[0]], share)
        within = diff_in_means(y[mask & (t == 1)], y[mask & (t == 0)])
        row["check_diff_in_means_per_1000"] = 1000 * share * within["ate"]
        row["lift_per_targeted_customer"] = within["ate"]
        rows.append(row)
    return pd.DataFrame(rows)


def profit_table(point: dict, draws: dict, margin: float, cost: float) -> pd.DataFrame:
    """Profit per 1,000 customers for every policy under one (margin, cost) assumption,
    with paired-bootstrap intervals for the two comparisons that matter."""
    everyone = profit_per_1000(point[EVERYONE], 1.0, margin, cost)
    everyone_draws = profit_per_1000(draws[EVERYONE], 1.0, margin, cost)

    def row(policy, source, share, profit, profit_draws):
        r = {"margin": margin, "cost_per_email": cost, "policy": policy, "scores": source,
             "share_emailed": share, "profit_per_1000": profit}
        r.update({f"profit_{k}": v for k, v in summarize_draws(profit_draws).items() if k != "boot_p"})
        return r

    rows = [{"margin": margin, "cost_per_email": cost, "policy": "Email no one", "scores": "",
             "share_emailed": 0.0, "profit_per_1000": 0.0, "profit_ci_low": 0.0, "profit_ci_high": 0.0,
             "vs_everyone": -everyone,
             **{f"vs_everyone_{k}": v for k, v in summarize_draws(-everyone_draws).items()}},
            row(EVERYONE, "", 1.0, everyone, everyone_draws)]

    for key in point:
        if key == EVERYONE:
            continue
        name, share = key
        profit = profit_per_1000(point[key], share, margin, cost)
        profit_draws = profit_per_1000(draws[key], share, margin, cost)
        # Random targeting at the same share: emailing a random `share` of
        # customers earns `share` x the email-everyone profit. Computed from
        # the SAME resample's email-everyone value, so the difference is paired.
        random_profit, random_draws = share * everyone, share * everyone_draws

        r = row(f"Model: top {share:.0%}", name, share, profit, profit_draws)
        r["vs_everyone"] = profit - everyone
        r.update({f"vs_everyone_{k}": v for k, v in summarize_draws(profit_draws - everyone_draws).items()})
        r["random_same_share"] = random_profit
        r.update({f"random_{k}": v for k, v in summarize_draws(random_draws).items() if k != "boot_p"})
        r["vs_random"] = profit - random_profit
        r.update({f"vs_random_{k}": v for k, v in summarize_draws(profit_draws - random_draws).items()})
        rows.append(r)
    return pd.DataFrame(rows)


def persuadables_profile(test: pd.DataFrame, score_col: str, share: float = cfg.PROFILE_SHARE):
    """Describe the top vs. bottom `share` of customers by PREDICTED uplift.

    Returns (covariate profile, actual randomized lift in each group).
    The profile is descriptive: it says what the model's high-score customers
    look like, not that those traits cause a bigger response.
    """
    top = top_share_mask(test[score_col].to_numpy(), share)
    bottom = top_share_mask(-test[score_col].to_numpy(), share)
    X = covariate_matrix(test, include_redundant=False)

    profile = pd.DataFrame({
        "covariate": X.columns,
        "top_group_mean": [X.loc[top, c].mean() for c in X.columns],
        "bottom_group_mean": [X.loc[bottom, c].mean() for c in X.columns],
        "all_test_mean": [X[c].mean() for c in X.columns],
        "smd_top_vs_bottom": [standardized_mean_difference(X.loc[top, c].to_numpy(), X.loc[bottom, c].to_numpy())
                              for c in X.columns],
    })

    # What actually happened in each group (emailed minus control, from the experiment).
    middle = ~(top | bottom)
    rows = []
    for group, mask in [(f"Top {share:.0%} (predicted persuadables)", top), ("Middle", middle),
                        (f"Bottom {share:.0%}", bottom)]:
        g = test[mask]
        for outcome in cfg.OUTCOMES:
            res = diff_in_means(g.loc[g.treated == 1, outcome].to_numpy(float),
                                g.loc[g.treated == 0, outcome].to_numpy(float))
            rows.append({"group": group, "n": int(mask.sum()), "mean_predicted_visit_uplift": g[score_col].mean(),
                         "outcome": outcome, "control_mean": res["control_mean"], "treated_mean": res["treated_mean"],
                         "actual_lift": res["ate"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                         "p_value": res["p_value"]})
    actual = pd.DataFrame(rows)

    # Is the top group's actual lift really bigger than the bottom group's?
    # The groups are different customers, so the two estimates are independent
    # and the SE of their difference is sqrt(se_top^2 + se_bottom^2).
    from scipy import stats
    z95 = stats.norm.ppf(0.975)
    for outcome in cfg.OUTCOMES:
        a = actual[actual.outcome == outcome].iloc[0]    # top
        b = actual[actual.outcome == outcome].iloc[-1]   # bottom
        diff = a["actual_lift"] - b["actual_lift"]
        se = np.hypot((a["ci_high"] - a["ci_low"]) / (2 * z95), (b["ci_high"] - b["ci_low"]) / (2 * z95))
        rows.append({"group": "Top minus bottom", "outcome": outcome, "actual_lift": diff,
                     "ci_low": diff - z95 * se, "ci_high": diff + z95 * se,
                     "p_value": 2 * stats.norm.sf(abs(diff / se))})
    return profile, pd.DataFrame(rows)


def excluded_group_detail(test: pd.DataFrame, score_col: str, share_emailed: float) -> pd.DataFrame:
    """What happened among the customers a top-`share_emailed` policy leaves out."""
    left_out = ~top_share_mask(test[score_col].to_numpy(), share_emailed)
    g = test[left_out]
    rows = []
    for arm, name in [(1, "women's email"), (0, "control")]:
        a = g[g.treated == arm]
        rows.append({"scores": score_col, "left_out_share": 1 - share_emailed, "arm": name, "n": len(a),
                     "visitors": int(a.visit.sum()), "buyers": int(a.conversion.sum()),
                     "total_spend": a.spend.sum(), "mean_spend": a.spend.mean(),
                     "largest_purchase": a.spend.max(), "share_mens_only": ((a.mens == 1) & (a.womens == 0)).mean()})
    return pd.DataFrame(rows)


# ====================================================================== figure
def plot_profit_curve(test: pd.DataFrame, base: pd.DataFrame, path):
    """Base-case profit per 1,000 customers as a function of the share emailed."""
    cfg.apply_plot_style()
    y, t = test["spend"].to_numpy(float), test["treated"].to_numpy()
    g = ipw_contributions(y, t, t.mean())
    n = len(test)
    share = np.arange(n + 1) / n

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, col in cfg.POLICY_SCORES.items():
        order = np.argsort(-test[col].to_numpy(), kind="stable")
        inc_revenue = np.r_[0.0, np.cumsum(g[order])] / n
        ax.plot(share * 100, profit_per_1000(inc_revenue, share), color=cfg.POLICY_COLORS[name], lw=2, label=name)

    everyone = base.loc[base.policy == EVERYONE].iloc[0]
    ax.plot([0, 100], [0, everyone["profit_per_1000"]], color=cfg.INK_MUTED, lw=1.5, ls="--",
            label="Random targeting")
    ax.axhline(0, color=cfg.INK_MUTED, lw=0.8)

    # 95% bootstrap intervals for the primary model at each cut-off, and for email-everyone.
    primary = next(iter(cfg.POLICY_SCORES))
    pts = pd.concat([base[base.scores == primary], base[base.policy == EVERYONE]])
    ax.errorbar(pts["share_emailed"] * 100, pts["profit_per_1000"],
                yerr=[pts["profit_per_1000"] - pts["profit_ci_low"], pts["profit_ci_high"] - pts["profit_per_1000"]],
                fmt="o", color=cfg.INK, ms=5, lw=1.2, capsize=3, zorder=5,
                label=f"95% CI: {primary} cut-offs and email everyone")

    ax.set_xlabel("% of customers emailed (highest predicted uplift first)")
    ax.set_ylabel("Profit per 1,000 customers ($)")
    ax.set_title(f"Assumes margin {cfg.MARGIN:.0%}, cost ${cfg.COST_PER_EMAIL:.2f} per email. "
                 "0% = email no one, 100% = email everyone.", fontsize=9, fontweight="normal",
                 color=cfg.INK_MUTED)
    fig.suptitle("Profit by share of customers emailed (held-out test set)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ====================================================================== main
def save(table: pd.DataFrame, name: str, show: bool = True):
    table.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    if show:
        print(f"\n=== {name} ===")
        print(table.to_string(index=False))


def main():
    test = load_test_set()
    y, t = test["spend"].to_numpy(float), test["treated"].to_numpy()
    scores = {name: test[col].to_numpy() for name, col in cfg.POLICY_SCORES.items()}
    print(f"test customers={len(test):,}  known propensity (treated share)={t.mean():.6f}  "
          f"buyers={int(test.conversion.sum())}")

    point = policy_incremental_revenue(y, t, scores)
    draws = bootstrap_policies(y, t, scores)

    save(revenue_table(point, draws, y, t, scores), "phase3_incremental_revenue")

    base = profit_table(point, draws, cfg.MARGIN, cfg.COST_PER_EMAIL)
    save(base, "phase3_profit_base_case")

    grid = pd.concat([profit_table(point, draws, m, c) for m in cfg.MARGIN_GRID for c in cfg.COST_GRID])
    save(grid, "phase3_profit_grid", show=False)
    primary = next(iter(cfg.POLICY_SCORES))
    compact = grid[(grid.scores == primary) | (grid.policy == EVERYONE)]
    print("\n=== grid (primary model + email everyone) ===")
    print(compact[["margin", "cost_per_email", "policy", "profit_per_1000", "profit_ci_low", "profit_ci_high",
                   "vs_everyone", "vs_everyone_ci_low", "vs_everyone_ci_high", "vs_everyone_boot_p"]]
          .to_string(index=False))
    # How many of the (assumption x policy) cells have a targeted policy that
    # beats email-everyone with an interval excluding zero?
    targeted = grid[grid.policy.str.startswith("Model")]
    print("\ncells where targeted beats everyone (CI excludes 0):",
          int((targeted.vs_everyone_ci_low > 0).sum()), "of", len(targeted))
    print("cells where targeted is worse than everyone (CI excludes 0):",
          int((targeted.vs_everyone_ci_high < 0).sum()), "of", len(targeted))
    print("cells where targeted beats random (CI excludes 0):",
          int((targeted.vs_random_ci_low > 0).sum()), "of", len(targeted))

    profile, actual = persuadables_profile(test, cfg.POLICY_SCORES[primary])
    save(profile, "phase3_persuadables_profile")
    save(actual, "phase3_persuadables_actual_lift")
    # Same profile for the sensitivity model, to see whether the description is model-specific.
    profile_x, _ = persuadables_profile(test, cfg.POLICY_SCORES["X-learner (visit)"])
    save(profile_x, "phase3_persuadables_profile_xlearner")

    # Email everyone vs. no one needs no model, so it can use the FULL experiment
    # (Phase 1 ATE and its CI) rather than only the 30% test set.
    ate = pd.read_csv(TABLES_DIR / "phase1_ate.csv")
    ate = ate[(ate.arm == cfg.WOMENS) & (ate.outcome == "spend")].iloc[0]
    save(pd.DataFrame([{"margin": m, "cost_per_email": c,
                        "profit_per_1000": profit_per_1000(ate["ate"], 1.0, m, c),
                        "ci_low": profit_per_1000(ate["ci_low"], 1.0, m, c),
                        "ci_high": profit_per_1000(ate["ci_high"], 1.0, m, c),
                        "break_even_cost_per_email": m * ate["ate"],
                        "break_even_cost_at_ci_low": m * ate["ci_low"]}
                       for m in cfg.MARGIN_GRID for c in cfg.COST_GRID]),
         "phase3_email_everyone_full_sample")

    # The customers a top-90% policy leaves out, for each set of scores.
    save(pd.concat([excluded_group_detail(test, col, 0.90) for col in cfg.POLICY_SCORES.values()]),
         "phase3_bottom_10pct_detail")

    plot_profit_curve(test, base, cfg.FIGURES_DIR / "phase3_profit_curve.png")


if __name__ == "__main__":
    pd.set_option("display.width", 280, "display.max_columns", 40)
    main()
