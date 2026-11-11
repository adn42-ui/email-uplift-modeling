"""Small, explicit statistical helpers used across phases.

Nothing here is specific to uplift modelling: these are the textbook tools
for analysing a randomized experiment.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import config as cfg


# ------------------------------------------------------------------ balance
def standardized_mean_difference(x_treat: np.ndarray, x_ctrl: np.ndarray) -> float:
    """SMD = (mean_t - mean_c) / pooled SD.

    Unlike a p-value, the SMD does not shrink or grow with sample size, so it
    measures *how big* an imbalance is rather than whether we can detect it.
    A common rule of thumb is that |SMD| < 0.1 is negligible.
    """
    pooled_sd = np.sqrt((x_treat.var(ddof=1) + x_ctrl.var(ddof=1)) / 2)
    return (x_treat.mean() - x_ctrl.mean()) / pooled_sd


# ------------------------------------------------------------------ ATE
def diff_in_means(y_treat: np.ndarray, y_ctrl: np.ndarray, alpha: float = cfg.ALPHA) -> dict:
    """Average treatment effect as a difference in means, with a Welch
    (unequal-variance) standard error and a normal-approximation CI.

    In a randomized experiment this simple difference is an unbiased estimate
    of the ATE. For 0/1 outcomes it is the difference in proportions and the
    Welch SE is (to within n vs n-1) the usual two-proportion SE.
    """
    n_t, n_c = len(y_treat), len(y_ctrl)
    mean_t, mean_c = y_treat.mean(), y_ctrl.mean()
    ate = mean_t - mean_c
    se = np.sqrt(y_treat.var(ddof=1) / n_t + y_ctrl.var(ddof=1) / n_c)
    z = stats.norm.ppf(1 - alpha / 2)
    return {
        "control_mean": mean_c,
        "treated_mean": mean_t,
        "ate": ate,
        "se": se,
        "ci_low": ate - z * se,
        "ci_high": ate + z * se,
        "p_value": 2 * stats.norm.sf(abs(ate / se)),
        "relative_lift": ate / mean_c,
    }


def bootstrap_diff_ci(y_treat, y_ctrl, n_boot: int = 5000, alpha: float = cfg.ALPHA,
                      seed: int = cfg.RANDOM_SEED) -> tuple[float, float]:
    """Percentile bootstrap CI for a difference in means.

    Used as a cross-check for `spend`, which is ~99% zeros with a long right
    tail, so we don't rely only on the normal approximation. Each arm is
    resampled with replacement independently, mirroring the design.
    """
    rng = np.random.default_rng(seed)
    y_treat, y_ctrl = np.asarray(y_treat), np.asarray(y_ctrl)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        diffs[b] = (rng.choice(y_treat, len(y_treat)).mean()
                    - rng.choice(y_ctrl, len(y_ctrl)).mean())
    return tuple(np.quantile(diffs, [alpha / 2, 1 - alpha / 2]))


def regression_adjusted_ate(y: np.ndarray, t: np.ndarray, X: pd.DataFrame,
                            alpha: float = cfg.ALPHA) -> dict:
    """Covariate-adjusted ATE (Lin 2013): regress y on treatment, centred
    covariates, and their interactions; the treatment coefficient is the ATE.

    Why bother in an RCT? Not to remove bias (there is none) but to soak up
    outcome variance explained by pre-treatment covariates, which can tighten
    the CI. If randomization worked, the point estimate should barely move.
    Robust (HC2) standard errors, so no homoskedasticity assumption.
    """
    Xc = X - X.mean()
    design = pd.concat(
        [pd.Series(t, index=X.index, name="treat"), Xc,
         Xc.mul(t, axis=0).add_prefix("treat:")], axis=1)
    fit = sm.OLS(y, sm.add_constant(design)).fit(cov_type="HC2")
    lo, hi = fit.conf_int(alpha=alpha).loc["treat"]
    return {"ate": fit.params["treat"], "se": fit.bse["treat"],
            "ci_low": lo, "ci_high": hi, "p_value": fit.pvalues["treat"]}
