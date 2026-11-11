"""Phase 4, step 4: what would break if this were observational data?

    python -m src.phase4_simulation

*** THIS IS A SIMULATION. The "observational" data below is manufactured from
*** the real experiment to demonstrate confounding. It is not a finding
*** about the retailer.

The setup
    In the real experiment a coin flip decided who got the email. Here we
    pretend marketers chose instead, and that they favoured big spenders. We
    fake that by throwing away rows:

        an EMAILED customer is kept with probability      p(history)
        a CONTROL  customer is kept with probability  1 - p(history)

    where p rises with past-year spend. In what is left, high-history
    customers are mostly emailed and low-history customers are mostly not,
    exactly what targeting by marketers would produce.

Why the true answer is still known
    Before subsampling the arms are 50/50, so a customer with history h
    survives with probability 0.5*p + 0.5*(1-p) = 0.5 whatever h is. The fake
    dataset therefore has the same mix of customers as the real one, and its
    true ATE is still the experimental ATE. Any gap between an estimate and
    that number is caused by HOW treatment was assigned, nothing else.

Three estimators on each fake dataset
    1. Naive: mean(emailed) - mean(control). Wrong, because the emailed group
       is now richer, and richer customers visit and spend more anyway.
    2. IPW, propensity model INCLUDES history. Estimate P(emailed | x) with a
       logistic regression, then weight each customer by 1 / P(the treatment
       they actually got). That re-balances the groups: a rare low-history
       emailed customer counts for many. Should recover the truth.
    3. IPW, propensity model OMITS history (and its binned copy,
       history_segment). This stands in for an UNMEASURED confounder: the
       analyst does everything "right" but lacks the one variable that drove
       assignment. Should stay biased.

The lesson
    Adjustment (IPW, matching, regression, causal forests with an estimated
    propensity, ...) can only fix confounders you have measured. In real
    observational data you never know whether you have them all.
    Randomization fixes measured and unmeasured confounders alike, which is
    why the estimates in this project can be read causally.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from . import config as cfg
from .data import load_hillstrom, two_arm_subset

TABLES_DIR = cfg.REPORTS_DIR / "tables"
ESTIMATORS = ["Naive difference in means", "IPW, propensity includes history",
              "IPW, propensity omits history"]
EST_COLORS = dict(zip(ESTIMATORS, ["#e34948", "#2a78d6", "#eda100"]))


def assignment_probability(history: pd.Series) -> np.ndarray:
    """P(emailed) in the fake world: rises smoothly with past spend, and is
    clipped to a moderate range so no customer gets an extreme IPW weight."""
    log_h = np.log(history)
    z = (log_h - log_h.mean()) / log_h.std()
    return np.clip(expit(z), *cfg.SIM_PROPENSITY_RANGE)


def propensity_features(df: pd.DataFrame, include_history: bool) -> pd.DataFrame:
    X = df[["recency", "mens", "womens", "newbie"]].astype(float)
    X = pd.concat([X, pd.get_dummies(df[["zip_code", "channel"]], drop_first=True, dtype=float)], axis=1)
    if include_history:
        X["log_history"] = np.log(df["history"])
    return X


def ipw_ate(y, t, e_hat) -> float:
    """Normalised (Hajek) IPW estimate of the ATE: a weighted mean of the
    emailed minus a weighted mean of the controls, weights = 1 / P(own arm)."""
    w1, w0 = t / e_hat, (1 - t) / (1 - e_hat)
    return float((w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum())


def one_draw(df, p, X_with, X_without, rng) -> dict:
    keep = rng.random(len(df)) < np.where(df["treated"] == 1, p, 1 - p)
    d, t = df[keep], df.loc[keep, "treated"].to_numpy(float)

    def fitted_propensity(X):
        Xk = X[keep]
        Xk = (Xk - Xk.mean()) / Xk.std()   # standardise so the solver converges quickly
        return LogisticRegression(C=1e4, max_iter=1000).fit(Xk, t).predict_proba(Xk)[:, 1]

    e_with, e_without = fitted_propensity(X_with), fitted_propensity(X_without)
    out = {"n_kept": int(keep.sum()), "treated_share": t.mean(),
           "mean_history_emailed": d.loc[d.treated == 1, "history"].mean(),
           "mean_history_control": d.loc[d.treated == 0, "history"].mean(),
           "e_hat_min": e_with.min(), "e_hat_max": e_with.max()}
    for outcome in ("visit", "spend"):
        y = d[outcome].to_numpy(float)
        out[f"{outcome}|{ESTIMATORS[0]}"] = y[t == 1].mean() - y[t == 0].mean()
        out[f"{outcome}|{ESTIMATORS[1]}"] = ipw_ate(y, t, e_with)
        out[f"{outcome}|{ESTIMATORS[2]}"] = ipw_ate(y, t, e_without)
    return out


def plot(summary: pd.DataFrame, path):
    cfg.apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    for ax, (outcome, scale, unit) in zip(axes, [("visit", 100, "percentage points"), ("spend", 1, "dollars per customer")]):
        sub = summary[summary.outcome == outcome].set_index("estimator")
        truth = sub["true_ate"].iloc[0] * scale
        ax.axvline(truth, color=cfg.INK, lw=1.5, ls="--")
        ax.annotate("true (experimental) ATE", (truth, 2.75), xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=cfg.INK_MUTED)
        for i, est in enumerate(ESTIMATORS[::-1]):
            r = sub.loc[est]
            ax.plot([r["p2_5"] * scale, r["p97_5"] * scale], [i, i], color=EST_COLORS[est], lw=2)
            ax.scatter(r["mean"] * scale, i, s=60, color=EST_COLORS[est], edgecolor=cfg.SURFACE, linewidth=1.5, zorder=3)
        ax.set_yticks(range(3), ESTIMATORS[::-1] if ax is axes[0] else [""] * 3)
        ax.set_ylim(-0.6, 3.0)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Effect on {outcome}")
        ax.set_xlabel(f"Estimated ATE ({unit})")
    fig.suptitle("SIMULATION: estimates from fake 'observational' data where big spenders are favoured for email\n"
                 f"dot = mean, bar = middle 95% of {cfg.SIM_DRAWS} simulated datasets",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    df = two_arm_subset(load_hillstrom(), cfg.WOMENS)
    p = assignment_probability(df["history"])
    X_with, X_without = propensity_features(df, True), propensity_features(df, False)
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    draws = pd.DataFrame([one_draw(df, p, X_with, X_without, rng) for _ in range(cfg.SIM_DRAWS)])

    truth = {o: df.loc[df.treated == 1, o].mean() - df.loc[df.treated == 0, o].mean() for o in ("visit", "spend")}
    rows = []
    for outcome in ("visit", "spend"):
        for est in ESTIMATORS:
            x = draws[f"{outcome}|{est}"]
            rows.append({"outcome": outcome, "estimator": est, "true_ate": truth[outcome], "mean": x.mean(),
                         "bias": x.mean() - truth[outcome], "bias_pct_of_truth": 100 * (x.mean() / truth[outcome] - 1),
                         "sd_across_draws": x.std(), "p2_5": x.quantile(0.025), "p97_5": x.quantile(0.975),
                         # how many simulated datasets land above the truth (50% = unbiased)
                         "share_of_draws_above_truth": (x > truth[outcome]).mean()})
    summary = pd.DataFrame(rows)
    design = draws[["n_kept", "treated_share", "mean_history_emailed", "mean_history_control",
                    "e_hat_min", "e_hat_max"]].mean().to_frame("mean_across_draws").T
    design["true_p_min"], design["true_p_max"] = p.min(), p.max()
    design["share_clipped_low"], design["share_clipped_high"] = (p == p.min()).mean(), (p == p.max()).mean()

    summary.to_csv(TABLES_DIR / "phase4_simulation.csv", index=False)
    design.to_csv(TABLES_DIR / "phase4_simulation_design.csv", index=False)
    print(design.round(4).to_string(index=False))
    print(summary.round(4).to_string(index=False))
    plot(summary, cfg.FIGURES_DIR / "phase4_simulation.png")


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30)
    main()
