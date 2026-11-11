"""Phase 4, step 1: out-of-fold uplift scores for EVERY customer.

    python -m src.phase4_crossfit        (slow: ~25 causal-forest fits)

Why: Phases 2-3 judged everything on one 30% test set, which turned out to
have a lucky spend ATE and was also used to pick the model. Cross-fitting
removes the dependence on that one split:

    split the customers into 5 folds;
    for each fold, train on the other 4 and score the held-out fold.

Every customer then has a score from a model that never saw them, so all
42,693 customers can be used for evaluation instead of 12,808. Repeating the
whole thing 5 times with different fold assignments shows how much results
move when only the split changes.

Models: visit-trained causal forest and X-learner, with exactly the Phase 2
hyperparameters and the known propensity. Nothing is tuned.

Output: cache/phase4_oof_scores.csv with, per repeat r:
    fold_r, forest_r, xlearner_r
"""
import time

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config as cfg
from .data import load_hillstrom, model_features, two_arm_subset
from .learners import CausalForest, XLearner

OOF_PATH = cfg.CACHE_DIR / "phase4_oof_scores.csv"


def main():
    df = two_arm_subset(load_hillstrom(), cfg.WOMENS)
    X = model_features(df).to_numpy()
    t = df["treated"].to_numpy()
    y = df["visit"].to_numpy(float)
    strata = df["treated"] * 2 + df["conversion"]   # same stratification as Phase 2

    out = pd.DataFrame(index=df.index)
    for r in range(cfg.CV_REPEATS):
        start = time.time()
        folds = StratifiedKFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_SEED + r)
        out[f"fold_{r}"] = -1
        out[f"forest_{r}"] = 0.0
        out[f"xlearner_{r}"] = 0.0
        for k, (train, held_out) in enumerate(folds.split(X, strata)):
            propensity = float(t[train].mean())   # known by design: the treated share
            seed = cfg.RANDOM_SEED + 100 * r + k
            out.loc[held_out, f"fold_{r}"] = k
            out.loc[held_out, f"forest_{r}"] = (
                CausalForest(propensity, seed).fit(X[train], t[train], y[train]).predict(X[held_out]))
            out.loc[held_out, f"xlearner_{r}"] = (
                XLearner(True, propensity, seed).fit(X[train], t[train], y[train]).predict(X[held_out]))
        print(f"repeat {r + 1}/{cfg.CV_REPEATS} done in {time.time() - start:.0f}s", flush=True)

    cfg.CACHE_DIR.mkdir(exist_ok=True)
    out.to_csv(OOF_PATH, index_label="row")
    print(f"saved {OOF_PATH.name}: {out.shape[0]:,} rows")


if __name__ == "__main__":
    main()
