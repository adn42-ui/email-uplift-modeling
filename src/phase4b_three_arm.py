"""Phase 4b: the three-arm decision. Which email, if any, for each customer?

    python -m src.phase4b_three_arm

No machine learning. Phase 4 found that the response to the women's email is
driven almost entirely by one thing: what the customer bought last year. So
the natural next question is whether men's-only buyers should get the MEN'S
email instead. The experiment already contains a men's-email arm, so this
can be answered from existing data.

Segments (fixed in advance, from last year's purchases):
    Men's only   (mens = 1, womens = 0)
    Women's only (mens = 0, womens = 1)
    Both         (mens = 1, womens = 1)

Step 1  Inside each segment, compare each email with control, and the two
        emails with each other, by difference in means. Valid because
        assignment was random within every segment.

Step 2  Value whole-population policies. A policy assigns each SEGMENT one
        of {no email, men's email, women's email}. Its value is estimated
        with inverse propensity weighting over three arms:

            V(policy) = average over customers of
                        [customer's actual arm == arm the policy assigns] * spend / P(that arm)

        i.e. use only the customers who happened to receive what the policy
        would have given them, scaled up by 1 / (their chance of receiving
        it, about 1/3). "Email no one" is the mean spend of the control arm.
        Incremental revenue = V(policy) - V(email no one).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg
from .data import load_hillstrom
from .evaluation import summarize_draws
from .policy import profit_per_1000
from .stats import diff_in_means

TABLES_DIR = cfg.REPORTS_DIR / "tables"
SEGMENTS = ["Men's only", "Women's only", "Both"]
NONE = cfg.CONTROL
ARM_SHORT = {cfg.MENS: "Men's email", cfg.WOMENS: "Women's email", NONE: "No email"}

# Policies fixed in advance: {segment: arm}.
FIXED_POLICIES = {
    "Everyone gets women's email": dict.fromkeys(SEGMENTS, cfg.WOMENS),
    "Everyone gets men's email": dict.fromkeys(SEGMENTS, cfg.MENS),
    "Matched rule (men's email to men's-only, women's email to the rest)":
        {"Men's only": cfg.MENS, "Women's only": cfg.WOMENS, "Both": cfg.WOMENS},
}
BEST = "Best email per segment (chosen on this data: optimistic)"


def add_segment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["purchase_segment"] = np.select(
        [(df.mens == 1) & (df.womens == 0), (df.mens == 0) & (df.womens == 1)], SEGMENTS[:2], "Both")
    return df


# ====================================================================== step 1
def segment_counts(df) -> pd.DataFrame:
    g = df.groupby(["purchase_segment", cfg.TREATMENT_COL])
    out = g.agg(n=("spend", "size"), visitors=("visit", "sum"), buyers=("conversion", "sum"),
                mean_spend=("spend", "mean")).reset_index()
    out["purchase_segment"] = pd.Categorical(out["purchase_segment"], SEGMENTS)
    return out.sort_values(["purchase_segment", cfg.TREATMENT_COL])


def segment_effects(df) -> pd.DataFrame:
    """Every pairwise comparison, in every segment, for every outcome (27 tests)."""
    comparisons = [("Men's email vs. control", cfg.MENS, NONE), ("Women's email vs. control", cfg.WOMENS, NONE),
                   ("Men's email vs. women's email", cfg.MENS, cfg.WOMENS)]
    rows = []
    for seg in SEGMENTS + ["All customers"]:
        d = df if seg == "All customers" else df[df.purchase_segment == seg]
        for label, a, b in comparisons:
            for outcome in cfg.OUTCOMES:
                r = diff_in_means(d.loc[d[cfg.TREATMENT_COL] == a, outcome].to_numpy(float),
                                  d.loc[d[cfg.TREATMENT_COL] == b, outcome].to_numpy(float))
                rows.append({"segment": seg, "comparison": label, "outcome": outcome, "difference": r["ate"],
                             "se": r["se"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_value": r["p_value"]})
    out = pd.DataFrame(rows)
    in_family = out.segment != "All customers"          # the 27 within-segment tests
    out["p_bonferroni_27"] = np.where(in_family, (out.p_value * in_family.sum()).clip(upper=1), np.nan)
    return out


def interaction_tests(effects: pd.DataFrame) -> pd.DataFrame:
    """Does an email's effect differ BETWEEN segments? (The matched rule only
    makes sense if it does.) Segments are different customers, so the two
    estimates are independent: SE of the difference = sqrt(se1^2 + se2^2)."""
    e = effects.set_index(["segment", "comparison", "outcome"])
    contrasts = [("Men's email vs. control", "Men's only", "Women's only"),
                 ("Women's email vs. control", "Women's only", "Men's only"),
                 ("Men's email vs. women's email", "Men's only", "Women's only")]
    rows = []
    for comp, seg_a, seg_b in contrasts:
        for outcome in cfg.OUTCOMES:
            a, b = e.loc[(seg_a, comp, outcome)], e.loc[(seg_b, comp, outcome)]
            diff, se = a["difference"] - b["difference"], np.hypot(a["se"], b["se"])
            rows.append({"effect": comp, "contrast": f"{seg_a} minus {seg_b}", "outcome": outcome,
                         "difference_in_effect": diff, "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se,
                         "p_value": 2 * stats.norm.sf(abs(diff / se))})
    return pd.DataFrame(rows)


# ====================================================================== step 2
def policy_columns(df, policies: dict, arm_share: dict):
    """Per-customer IPW contribution to incremental revenue, and an 'emailed'
    flag, for each policy. Column means are the policy's value."""
    arm = df[cfg.TREATMENT_COL].to_numpy()
    seg = df["purchase_segment"].to_numpy()
    y = df["spend"].to_numpy(float)
    baseline = (arm == NONE) * y / arm_share[NONE]            # V(email no one), per customer
    contrib, emailed = {}, {}
    for name, rule in policies.items():
        assigned = np.array([rule[s] for s in SEGMENTS])[pd.Categorical(seg, SEGMENTS).codes]
        weight = np.array([1 / arm_share[a] for a in assigned])
        contrib[name] = (arm == assigned) * y * weight - baseline
        emailed[name] = (assigned != NONE).astype(float)
    return pd.DataFrame(contrib), pd.DataFrame(emailed)


def best_per_segment(effects: pd.DataFrame, margin: float, cost: float) -> dict:
    """For each segment pick the option with the highest estimated profit per
    customer: no email (0), or margin x spend lift - cost for each email.
    Chosen on the same data it is then valued on, so its value is optimistic."""
    e = effects[(effects.outcome == "spend")].set_index(["segment", "comparison"])["difference"]
    rule = {}
    for seg in SEGMENTS:
        options = {NONE: 0.0,
                   cfg.MENS: margin * e[(seg, "Men's email vs. control")] - cost,
                   cfg.WOMENS: margin * e[(seg, "Women's email vs. control")] - cost}
        rule[seg] = max(options, key=options.get)
    return rule


def bootstrap_means(df, contrib: pd.DataFrame, emailed: pd.DataFrame):
    """Resample customers within each arm; return (n_boot x n_policies)
    arrays of incremental revenue and share emailed, same resamples for all."""
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    arm = df[cfg.TREATMENT_COL].to_numpy()
    arm_rows = [np.nonzero(arm == a)[0] for a in cfg.ARMS]
    C, E = contrib.to_numpy(), emailed.to_numpy()
    inc, share = np.empty((cfg.N_BOOTSTRAP, C.shape[1])), np.empty((cfg.N_BOOTSTRAP, C.shape[1]))
    for b in range(cfg.N_BOOTSTRAP):
        rows = np.concatenate([rng.choice(r, len(r)) for r in arm_rows])
        inc[b], share[b] = C[rows].mean(axis=0), E[rows].mean(axis=0)
    return inc, share


def policy_table(names, contrib, emailed, inc_draws, share_draws, margin, cost) -> pd.DataFrame:
    profit = {n: profit_per_1000(contrib[n].mean(), emailed[n].mean(), margin, cost) for n in names}
    draws = {n: profit_per_1000(inc_draws[:, i], share_draws[:, i], margin, cost) for i, n in enumerate(names)}
    womens_all = "Everyone gets women's email"
    rows = [{"margin": margin, "cost_per_email": cost, "policy": "Email no one", "share_emailed": 0.0,
             "inc_revenue_per_1000": 0.0, "profit_per_1000": 0.0}]
    for n in names:
        row = {"margin": margin, "cost_per_email": cost, "policy": n, "share_emailed": emailed[n].mean(),
               "inc_revenue_per_1000": 1000 * contrib[n].mean(), "profit_per_1000": profit[n]}
        row.update({f"profit_{k}": v for k, v in summarize_draws(draws[n]).items()})   # CI and p vs. email no one
        # Paired differences against the two single-email policies.
        for ref, tag in [(womens_all, "vs_womens_to_all"), ("Everyone gets men's email", "vs_mens_to_all")]:
            if n != ref:
                row[tag] = profit[n] - profit[ref]
                row.update({f"{tag}_{k}": v for k, v in summarize_draws(draws[n] - draws[ref]).items()})
        rows.append(row)
    return pd.DataFrame(rows)


# ====================================================================== figure
def plot_segment_effects(effects: pd.DataFrame, path):
    cfg.apply_plot_style()
    panels = [("visit", "Visit rate", 100, "percentage points"), ("conversion", "Conversion rate", 100, "percentage points"),
              ("spend", "Spend per customer", 1, "dollars")]
    order = SEGMENTS[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for ax, (outcome, title, scale, unit) in zip(axes, panels):
        ax.axvline(0, color=cfg.INK_MUTED, lw=1)
        for arm, comp, dy in [(cfg.MENS, "Men's email vs. control", 0.14), (cfg.WOMENS, "Women's email vs. control", -0.14)]:
            e = effects[(effects.comparison == comp) & (effects.outcome == outcome)].set_index("segment").loc[order]
            y = np.arange(len(order)) + dy
            ax.errorbar(e["difference"] * scale, y, xerr=1.96 * e["se"] * scale, fmt="o", color=cfg.ARM_COLORS[arm],
                        ms=6, lw=2, markeredgecolor=cfg.SURFACE, label=ARM_SHORT[arm])
        ax.set_yticks(range(len(order)), [f"Bought {s[0].lower()}{s[1:]}" for s in order] if ax is axes[0] else [""] * len(order))
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel(f"Lift vs. no email ({unit})")
    axes[2].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Effect of each email by what the customer bought last year (95% CI)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ====================================================================== main
def save(table: pd.DataFrame, name: str, show=True):
    table.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    if show:
        print(f"\n=== {name} ===")
        print(table.round(4).to_string(index=False))


def main():
    df = add_segment(load_hillstrom())
    arm_share = df[cfg.TREATMENT_COL].value_counts(normalize=True).to_dict()   # known propensities, about 1/3 each
    print(f"customers={len(df):,}  arm shares={ {ARM_SHORT[k]: round(v, 4) for k, v in arm_share.items()} }")
    print("unclassified customers (bought neither category):", int(((df.mens == 0) & (df.womens == 0)).sum()))

    save(segment_counts(df), "phase4b_segment_counts")
    effects = segment_effects(df)
    save(effects, "phase4b_segment_effects")
    save(interaction_tests(effects), "phase4b_interactions")
    plot_segment_effects(effects, cfg.FIGURES_DIR / "phase4b_segment_effects.png")

    grid = []
    for margin in cfg.MARGIN_GRID:
        for cost in cfg.COST_GRID:
            best_rule = best_per_segment(effects, margin, cost)
            policies = {**FIXED_POLICIES, BEST: best_rule}
            contrib, emailed = policy_columns(df, policies, arm_share)
            inc_draws, share_draws = bootstrap_means(df, contrib, emailed)   # same seed -> same resamples every cell
            table = policy_table(list(policies), contrib, emailed, inc_draws, share_draws, margin, cost)
            table["best_per_segment_choice"] = "; ".join(f"{s}: {ARM_SHORT[a]}" for s, a in best_rule.items())
            grid.append(table)
            if (margin, cost) == (cfg.MARGIN, cfg.COST_PER_EMAIL):
                save(table.drop(columns="best_per_segment_choice"), "phase4b_policies_base_case")
                print("best per segment at base case:", best_rule)
    grid = pd.concat(grid)
    save(grid, "phase4b_policies_grid", show=False)
    print("\n=== grid: profit per 1,000 and paired differences ===")
    print(grid[grid.policy != "Email no one"][["margin", "cost_per_email", "policy", "profit_per_1000", "profit_ci_low",
                                                "profit_ci_high", "profit_boot_p", "vs_mens_to_all",
                                                "vs_mens_to_all_ci_low", "vs_mens_to_all_ci_high", "vs_mens_to_all_boot_p"]]
          .assign(policy=lambda d: d.policy.str[:28]).round(3).to_string(index=False))
    print("distinct best-per-segment choices across the grid:", grid["best_per_segment_choice"].unique())


if __name__ == "__main__":
    pd.set_option("display.width", 300, "display.max_columns", 40)
    main()
