"""Phase 2: uplift models for the women's email vs. control.

Run from the project root:

    python -m src.phase2_uplift

For each outcome (visit, spend): fit four CATE models on the training set,
then judge them on the held-out test set by how well they RANK customers.
"""
import time
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg
from .data import load_hillstrom, model_features, two_arm_subset
from .evaluation import paired_bootstrap, summarize_draws, uplift_and_qini_curves, uplift_metrics
from .learners import build_models
from .stats import diff_in_means

TABLES_DIR = cfg.REPORTS_DIR / "tables"
# (outcome column, is it 0/1?, y-axis label, unit used in tables)
TARGETS = [("visit", True, "Incremental visits per customer", "visits per customer"),
           ("spend", False, "Incremental revenue per customer ($)", "$ per customer")]


def make_split(df: pd.DataFrame):
    """One 70/30 split, stratified on treatment x conversion.

    Stratifying on treatment keeps the treated share identical in both halves.
    Stratifying on conversion spreads the ~311 buyers proportionally, so the
    test set is not left with unusually few (or many) of the only customers
    with non-zero spend.
    """
    strata = df["treated"] * 2 + df["conversion"]
    train_idx, test_idx = train_test_split(df.index, test_size=cfg.TEST_SIZE, stratify=strata,
                                           random_state=cfg.RANDOM_SEED)
    return df.loc[train_idx], df.loc[test_idx]


def split_summary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, part in [("train", train), ("test", test)]:
        for arm, g in part.groupby("treated"):
            rows.append({"split": name, "arm": "women's email" if arm else "control", "n": len(g),
                         "visitors": int(g["visit"].sum()), "buyers": int(g["conversion"].sum()),
                         "mean_spend": g["spend"].mean()})
    return pd.DataFrame(rows)


def plot_curves(y, t, scores: dict, ylabel: str, title: str, path):
    """Uplift curve (left) and Qini curve (right) for every model."""
    cfg.apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for name, s in scores.items():
        frac, uplift, qini = uplift_and_qini_curves(y, t, s)
        axes[0].plot(frac * 100, uplift, color=cfg.MODEL_COLORS[name], lw=2, label=name)
        axes[1].plot(frac * 100, qini, color=cfg.MODEL_COLORS[name], lw=2, label=name)
    for ax, curve_name, end in [(axes[0], "Uplift curve", uplift[-1]), (axes[1], "Qini curve", qini[-1])]:
        ax.plot([0, 100], [0, end], color=cfg.INK_MUTED, lw=1.5, ls="--", label="Random targeting")
        ax.axhline(0, color=cfg.INK_MUTED, lw=0.8)
        ax.set_title(curve_name)
        ax.set_xlabel("% of customers targeted (highest predicted uplift first)")
    axes[0].set_ylabel(ylabel)
    axes[1].legend(loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def crosscheck_against_sklift(y, t, scores: dict) -> float:
    """Largest absolute gap between our curves and scikit-uplift's, for a 0/1
    outcome (scikit-uplift does not accept continuous outcomes)."""
    from sklift.metrics import qini_curve, uplift_curve

    worst = 0.0
    for s in scores.values():
        _, ours_uplift, ours_qini = uplift_and_qini_curves(y, t, s)
        _, theirs_uplift = uplift_curve(y, s, t)
        _, theirs_qini = qini_curve(y, s, t)
        worst = max(worst, np.abs(ours_uplift - theirs_uplift / len(y)).max(),
                    np.abs(ours_qini - theirs_qini / t.sum()).max())
    return worst


def evaluate_outcome(outcome, binary, X_train, t_train, train, X_test, t_test, test, propensity):
    """Fit the four models for one outcome; return test-set scores and tables."""
    y_train = train[outcome].to_numpy(float)
    y_test = test[outcome].to_numpy(float)

    scores = {}
    for name, model in build_models(binary, propensity).items():
        start = time.time()
        scores[name] = model.fit(X_train, t_train, y_train).predict(X_test)
        print(f"  fitted {name:<14} for {outcome} in {time.time() - start:5.1f}s")

    draws = paired_bootstrap(y_test, t_test, scores)

    # --- one row per model
    rows = []
    for name, s in scores.items():
        point = uplift_metrics(y_test, t_test, s)
        row = {"outcome": outcome, "model": name,
               "pred_mean": s.mean(), "pred_sd": s.std(), "pred_min": s.min(), "pred_max": s.max(),
               "auuc": point["auuc"], "auuc_random": point["auuc_random"],
               "auuc_vs_random": point["auuc_vs_random"]}
        row.update({f"auuc_vs_random_{k}": v for k, v in summarize_draws(draws[name]["auuc_vs_random"]).items()})
        row["qini"] = point["qini"]
        row.update({f"qini_{k}": v for k, v in summarize_draws(draws[name]["qini"]).items()})
        rows.append(row)

    # --- one row per pair of models: paired bootstrap of the AUUC difference
    pair_rows = []
    for a, b in combinations(scores, 2):
        diff_draws = draws[a]["auuc"] - draws[b]["auuc"]   # same resamples -> paired
        point = uplift_metrics(y_test, t_test, scores[a])["auuc"] - uplift_metrics(y_test, t_test, scores[b])["auuc"]
        pair_rows.append({"outcome": outcome, "model_a": a, "model_b": b, "auuc_a_minus_b": point,
                          **summarize_draws(diff_draws),
                          "rank_corr": pd.Series(scores[a]).corr(pd.Series(scores[b]), method="spearman")})
    return scores, pd.DataFrame(rows), pd.DataFrame(pair_rows)


def save(table: pd.DataFrame, name: str):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    print(f"\n=== {name} ===")
    print(table.to_string(index=False))


def main():
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = two_arm_subset(load_hillstrom(), cfg.WOMENS)
    train, test = make_split(df)

    X_train, X_test = model_features(train), model_features(test)
    feature_names = list(X_train.columns)
    t_train, t_test = train["treated"].to_numpy(), test["treated"].to_numpy()

    # Known propensity: the actual share assigned to the email (training set).
    propensity = float(t_train.mean())

    print(f"rows: total={len(df):,} train={len(train):,} test={len(test):,}")
    print(f"known propensity (treated share, train) = {propensity:.6f}; test share = {t_test.mean():.6f}")
    print(f"source covariates ({len(cfg.PRE_TREATMENT_COVARIATES)}): {cfg.PRE_TREATMENT_COVARIATES}")
    print(f"model columns ({len(feature_names)}): {feature_names}")
    save(split_summary(train, test), "phase2_split")

    # ATE on the test set: the end point of every curve and the random baseline.
    test_ate = pd.DataFrame([
        {"outcome": o, **diff_in_means(test.loc[test.treated == 1, o].to_numpy(float),
                                       test.loc[test.treated == 0, o].to_numpy(float))}
        for o, *_ in TARGETS])
    save(test_ate, "phase2_test_ate")

    predictions = test[["treated", "visit", "conversion", "spend"]].copy()
    model_tables, pair_tables = [], []
    for outcome, binary, ylabel, _ in TARGETS:
        print(f"\n--- outcome: {outcome}")
        scores, model_table, pair_table = evaluate_outcome(
            outcome, binary, X_train.to_numpy(), t_train, train, X_test.to_numpy(), t_test, test, propensity)
        model_tables.append(model_table)
        pair_tables.append(pair_table)
        y_test = test[outcome].to_numpy(float)
        plot_curves(y_test, t_test, scores, ylabel,
                    f"Held-out test set: models trained to predict uplift in {outcome}",
                    cfg.FIGURES_DIR / f"phase2_curves_{outcome}.png")
        for name, s in scores.items():
            predictions[f"uplift_{outcome}__{name}"] = s
        if binary:
            print(f"  max |our curve - scikit-uplift curve| = {crosscheck_against_sklift(y_test, t_test, scores):.2e}")

    # Cross-check: do the rankings learned from VISIT also capture incremental
    # REVENUE? Visit has ~18x more positive examples than spend, so a visit
    # model may be the more reliable ranker even when revenue is the goal.
    y_spend = test["spend"].to_numpy(float)
    visit_scores = {name: predictions[f"uplift_visit__{name}"].to_numpy() for name in cfg.MODEL_COLORS}
    draws = paired_bootstrap(y_spend, t_test, visit_scores)
    cross = pd.DataFrame([
        {"trained_on": "visit", "evaluated_on": "spend", "model": name,
         **{k: uplift_metrics(y_spend, t_test, s)[k] for k in ("auuc", "auuc_random", "auuc_vs_random")},
         **summarize_draws(draws[name]["auuc_vs_random"])}
        for name, s in visit_scores.items()])
    save(cross, "phase2_visit_models_on_spend")
    plot_curves(y_spend, t_test, visit_scores, "Incremental revenue per customer ($)",
                "Held-out test set: visit-trained models, judged on incremental revenue",
                cfg.FIGURES_DIR / "phase2_curves_visit_models_on_spend.png")

    save(pd.concat(model_tables), "phase2_model_metrics")
    save(pd.concat(pair_tables), "phase2_pairwise_auuc")
    # Test-set predictions, reused by Phase 3 so models need not be refit.
    predictions.to_csv(TABLES_DIR / "phase2_test_predictions.csv", index_label="row")


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30)
    main()
