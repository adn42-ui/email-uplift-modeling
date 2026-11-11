"""Loading and light preparation of the Hillstrom email dataset."""
import pandas as pd

from . import config as cfg


def load_hillstrom() -> pd.DataFrame:
    """Return the full Hillstrom dataset as one tidy DataFrame.

    64,000 customers who purchased in the previous 12 months, randomly assigned
    to one of three arms (men's email, women's email, no email). One row per
    customer: covariates, the `segment` (arm) column, and three outcomes.

    scikit-uplift downloads the CSV once and caches it under `data/`.
    """
    from sklift.datasets import fetch_hillstrom

    bunch = fetch_hillstrom(target_col="all", data_home=str(cfg.DATA_DIR))
    df = pd.concat([bunch.data, bunch.target, bunch.treatment], axis=1)

    # The raw file spells it "Surburban"; fix so tables and plots read cleanly.
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})
    return df


def model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for the uplift models, built from the 8 pre-treatment
    covariates and nothing else.

    Encoding:
      recency, history         numeric, as-is
      history_segment          ordinal 1..7 (the leading digit of the label)
      mens, womens, newbie     0/1, as-is
      zip_code, channel        one 0/1 column per level

    The outcomes (visit, conversion, spend) and the treatment column must never
    appear here: using a post-treatment variable as a feature would leak the
    answer into the model. The assert makes that impossible to do by accident.
    """
    assert not set(cfg.PRE_TREATMENT_COVARIATES) & set(cfg.OUTCOMES + [cfg.TREATMENT_COL])
    X = df[["recency", "history", "mens", "womens", "newbie"]].astype(float)
    X.insert(2, "history_segment", df["history_segment"].str[0].astype(float))
    dummies = pd.get_dummies(df[["zip_code", "channel"]], prefix_sep="=", dtype=float)
    return pd.concat([X, dummies], axis=1)


def two_arm_subset(df: pd.DataFrame, treated_arm: str) -> pd.DataFrame:
    """Rows from one email arm plus control, with a 0/1 `treated` column."""
    out = df[df[cfg.TREATMENT_COL].isin([treated_arm, cfg.CONTROL])].copy()
    out["treated"] = (out[cfg.TREATMENT_COL] == treated_arm).astype(int)
    return out.reset_index(drop=True)


def covariate_matrix(df: pd.DataFrame, include_redundant: bool = False) -> pd.DataFrame:
    """Numeric covariate matrix: numeric and binary columns as-is, categoricals
    expanded into one 0/1 column per level (e.g. `channel=Web`).

    All levels are kept (no reference level dropped) because this is used for
    balance tables, where we want to see every level. Models that need a
    full-rank matrix should drop one level per categorical themselves.
    """
    cats = list(cfg.CATEGORICAL_FEATURES)
    if include_redundant:
        cats += cfg.REDUNDANT_FEATURES
    dummies = pd.get_dummies(df[cats], prefix_sep="=", dtype=float)
    return pd.concat([df[cfg.NUMERIC_FEATURES + cfg.BINARY_FEATURES].astype(float), dummies], axis=1)
