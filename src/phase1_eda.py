"""Phase 1: did randomization work, and what are the average effects?

Run from the project root:

    python -m src.phase1_eda

Writes tables to reports/tables/ and figures to reports/figures/.
"""
import matplotlib

matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import config as cfg
from .data import covariate_matrix, load_hillstrom
from .stats import (bootstrap_diff_ci, diff_in_means, regression_adjusted_ate,
                    standardized_mean_difference)

TABLES_DIR = cfg.REPORTS_DIR / "tables"


# ====================================================================== balance
def balance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of every covariate in each arm, plus the standardized mean
    difference (SMD) of each email arm against control."""
    X = covariate_matrix(df, include_redundant=True)
    arm = df[cfg.TREATMENT_COL]
    rows = []
    for col in X.columns:
        ctrl = X.loc[arm == cfg.CONTROL, col].to_numpy()
        row = {"covariate": col, "mean_control": ctrl.mean()}
        for a, short in [(cfg.MENS, "mens"), (cfg.WOMENS, "womens")]:
            treated = X.loc[arm == a, col].to_numpy()
            row[f"mean_{short}_email"] = treated.mean()
            row[f"smd_{short}_vs_control"] = standardized_mean_difference(treated, ctrl)
        rows.append(row)
    return pd.DataFrame(rows)


def per_covariate_tests(df: pd.DataFrame) -> pd.DataFrame:
    """One 3-arm test per original covariate.

    Numeric -> one-way ANOVA (are the three arm means equal?).
    Categorical/binary -> chi-square test of independence between arm and level.

    Under perfect randomization these p-values are uniform on [0, 1], so with
    8 tests we expect roughly one below 0.125 by chance alone. That is why the
    table also reports a Bonferroni-adjusted p-value.
    """
    arm = df[cfg.TREATMENT_COL]
    rows = []
    for col in cfg.NUMERIC_FEATURES:
        groups = [df.loc[arm == a, col] for a in cfg.ARMS]
        stat, p = stats.f_oneway(*groups)
        rows.append({"covariate": col, "test": "ANOVA F", "statistic": stat, "p_value": p})
    for col in cfg.BINARY_FEATURES + cfg.CATEGORICAL_FEATURES + cfg.REDUNDANT_FEATURES:
        stat, p, _, _ = stats.chi2_contingency(pd.crosstab(df[col], arm))
        rows.append({"covariate": col, "test": "chi-square", "statistic": stat, "p_value": p})
    out = pd.DataFrame(rows)
    out["p_bonferroni"] = (out["p_value"] * len(out)).clip(upper=1.0)
    return out


def joint_balance_test(df: pd.DataFrame) -> dict:
    """Omnibus test: can the covariates, taken together, predict which arm a
    customer landed in?

    We fit a multinomial logit of arm on all covariates and compare it to an
    intercept-only model with a likelihood-ratio test. If assignment was truly
    random, the covariates carry no information about the arm and the test
    should NOT reject. This is one test instead of many, so there is no
    multiple-comparisons problem.
    """
    X = pd.get_dummies(df[cfg.NUMERIC_FEATURES + cfg.BINARY_FEATURES + cfg.CATEGORICAL_FEATURES],
                       drop_first=True, dtype=float)  # drop one level -> full rank
    y = pd.Categorical(df[cfg.TREATMENT_COL], categories=cfg.ARMS).codes
    fit = sm.MNLogit(y, sm.add_constant(X)).fit(disp=False)
    return {"lr_statistic": fit.llr, "df": int(fit.df_model), "p_value": fit.llr_pvalue,
            "mcfadden_pseudo_r2": fit.prsquared}


# ====================================================================== ATEs
def ate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Difference-in-means ATE of each email vs. control on each outcome."""
    arm = df[cfg.TREATMENT_COL]
    rows = []
    for a in cfg.TREATED_ARMS:
        for outcome in cfg.OUTCOMES:
            y_t = df.loc[arm == a, outcome].to_numpy(dtype=float)
            y_c = df.loc[arm == cfg.CONTROL, outcome].to_numpy(dtype=float)
            res = diff_in_means(y_t, y_c)
            row = {"arm": a, "outcome": outcome, "n_treated": len(y_t), "n_control": len(y_c), **res}
            if outcome == "spend":
                row["boot_ci_low"], row["boot_ci_high"] = bootstrap_diff_ci(y_t, y_c)
            rows.append(row)
    out = pd.DataFrame(rows)
    # Six comparisons (2 emails x 3 outcomes): show what survives Bonferroni.
    out["p_bonferroni"] = (out["p_value"] * len(out)).clip(upper=1.0)
    return out


def adjusted_ate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Same six ATEs, but regression-adjusted for pre-treatment covariates."""
    X_all = pd.get_dummies(df[cfg.NUMERIC_FEATURES + cfg.BINARY_FEATURES + cfg.CATEGORICAL_FEATURES],
                           drop_first=True, dtype=float)
    rows = []
    for a in cfg.TREATED_ARMS:
        sub = df[cfg.TREATMENT_COL].isin([a, cfg.CONTROL])
        t = (df.loc[sub, cfg.TREATMENT_COL] == a).to_numpy(dtype=float)
        for outcome in cfg.OUTCOMES:
            res = regression_adjusted_ate(df.loc[sub, outcome].to_numpy(dtype=float), t, X_all[sub])
            rows.append({"arm": a, "outcome": outcome, **res})
    return pd.DataFrame(rows)


def head_to_head_table(df: pd.DataFrame) -> pd.DataFrame:
    """Men's email minus women's email, on each outcome."""
    arm = df[cfg.TREATMENT_COL]
    rows = []
    for outcome in cfg.OUTCOMES:
        res = diff_in_means(df.loc[arm == cfg.MENS, outcome].to_numpy(dtype=float),
                            df.loc[arm == cfg.WOMENS, outcome].to_numpy(dtype=float))
        rows.append({"outcome": outcome, "womens_mean": res["control_mean"], "mens_mean": res["treated_mean"],
                     "mens_minus_womens": res["ate"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                     "p_value": res["p_value"]})
    return pd.DataFrame(rows)


def spend_anatomy(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive only: how spend per customer breaks into
    (share who buy) x (basket size among buyers).

    Basket size among buyers is NOT a causal comparison: 'bought' is itself
    affected by the email, so the buyers in each arm are different people.
    """
    g = df.groupby(cfg.TREATMENT_COL)
    buyers = df[df["conversion"] == 1].groupby(cfg.TREATMENT_COL)["spend"]
    out = pd.DataFrame({
        "n": g.size(),
        "n_buyers": g["conversion"].sum(),
        "conversion_rate": g["conversion"].mean(),
        "mean_spend_per_buyer": buyers.mean(),
        "median_spend_per_buyer": buyers.median(),
        "mean_spend_per_customer": g["spend"].mean(),
        "sd_spend_per_customer": g["spend"].std(),
    })
    return out.loc[cfg.ARMS].reset_index()


# ====================================================================== figures
def plot_balance(bal: pd.DataFrame, path):
    """Love plot: one row per covariate, one dot per email arm."""
    cfg.apply_plot_style()
    bal = bal.iloc[::-1].reset_index(drop=True)  # first covariate at the top
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.axvspan(-0.1, 0.1, color=cfg.GRID, alpha=0.45, lw=0)
    ax.axvline(0, color=cfg.INK_MUTED, lw=1)
    y = np.arange(len(bal))
    for a, short, dy in [(cfg.MENS, "mens", 0.15), (cfg.WOMENS, "womens", -0.15)]:
        ax.scatter(bal[f"smd_{short}_vs_control"], y + dy, s=42, color=cfg.ARM_COLORS[a],
                   edgecolor=cfg.SURFACE, linewidth=1, label=f"{a} vs. control", zorder=3)
    ax.set_yticks(y, bal["covariate"].str.replace("$", r"\$"))  # escape $ so it isn't read as math
    ax.set_xlim(-0.125, 0.125)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Standardized mean difference vs. no-email arm\n(shaded band: |SMD| < 0.1, conventionally 'negligible')")
    fig.suptitle("Covariate balance across arms", x=0.01, ha="left", fontsize=12, fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ates(ates: pd.DataFrame, path):
    """One panel per outcome (their units differ), a point + 95% CI per email."""
    cfg.apply_plot_style()
    panels = [("visit", "Visit rate", 100, "percentage points"),
              ("conversion", "Conversion rate", 100, "percentage points"),
              ("spend", "Spend per customer", 1, "dollars")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, (outcome, title, scale, unit) in zip(axes, panels):
        sub = ates[ates["outcome"] == outcome].set_index("arm")
        for i, a in enumerate(cfg.TREATED_ARMS):
            r = sub.loc[a]
            ax.plot([r["ci_low"] * scale, r["ci_high"] * scale], [i, i], color=cfg.ARM_COLORS[a], lw=2)
            ax.scatter(r["ate"] * scale, i, s=60, color=cfg.ARM_COLORS[a], edgecolor=cfg.SURFACE,
                       linewidth=1.5, zorder=3)
            ax.annotate(f"{r['ate'] * scale:+.2f}", (r["ate"] * scale, i), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.axvline(0, color=cfg.INK_MUTED, lw=1)
        ax.set_yticks(range(2), cfg.TREATED_ARMS if ax is axes[0] else ["", ""])
        ax.set_ylim(-0.6, 1.6)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel(f"Lift vs. no email ({unit})")
    fig.suptitle("Average treatment effects with 95% confidence intervals", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ====================================================================== main
def save(table: pd.DataFrame, name: str):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    print(f"\n=== {name} ===")
    print(table.to_string(index=False))


def main():
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_hillstrom()

    print(f"rows={len(df):,}  columns={df.shape[1]}  missing values={int(df.isna().sum().sum())}  "
          f"duplicate rows={int(df.duplicated().sum())}")
    save(df[cfg.TREATMENT_COL].value_counts().loc[cfg.ARMS].rename("n").reset_index(), "phase1_arm_sizes")
    # Were the arms meant to be equal thirds? Chi-square goodness of fit.
    counts = df[cfg.TREATMENT_COL].value_counts().loc[cfg.ARMS]
    print("arm-size chi-square vs equal thirds:", stats.chisquare(counts))

    bal = balance_table(df)
    save(bal, "phase1_balance")
    save(per_covariate_tests(df), "phase1_balance_tests")
    joint = joint_balance_test(df)
    save(pd.DataFrame([joint]), "phase1_balance_joint_test")
    plot_balance(bal, cfg.FIGURES_DIR / "phase1_balance.png")

    ates = ate_table(df)
    save(ates, "phase1_ate")
    save(adjusted_ate_table(df), "phase1_ate_adjusted")
    save(head_to_head_table(df), "phase1_mens_vs_womens")
    save(spend_anatomy(df), "phase1_spend_anatomy")
    plot_ates(ates, cfg.FIGURES_DIR / "phase1_ate.png")


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30)
    main()
