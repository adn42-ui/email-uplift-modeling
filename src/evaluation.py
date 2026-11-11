"""Evaluating uplift models on held-out data.

The core difficulty: we never observe anyone's true uplift, so there is no
"actual vs. predicted" to compare. What we *can* do, because treatment was
randomized, is this:

    1. Rank held-out customers by predicted uplift, best first.
    2. Take the top k of that ranking.
    3. Inside that top k, compare emailed vs. not-emailed customers.
       Randomization makes that comparison a valid estimate of the email's
       effect on those k customers.

A good model puts high-responders first, so the estimated incremental outcome
climbs quickly as k grows. A useless model ranks at random and the curve
climbs in a straight line. Both curves below are that idea with slightly
different bookkeeping. They work for any outcome: with a 0/1 outcome the
y-axis is incremental visits; with spend it is incremental revenue.

Scaling used throughout
    x-axis: fraction of customers targeted (0 to 1)
    y-axis: incremental outcome so far, per customer in the whole population
            (uplift curve) or per treated customer (Qini curve)
    With this scaling both curves end at the overall ATE, and a random ranking
    is the straight line from 0 to the ATE.
"""
import numpy as np

from . import config as cfg


def uplift_and_qini_curves(y, t, score):
    """Return (fraction_targeted, uplift_curve, qini_curve).

    At each cut-off k (the top-k customers by `score`):
        n_t, n_c = number of treated / control customers in the top k
        y_t, y_c = their total outcome

      uplift curve = (y_t / n_t  -  y_c / n_c) * k
          "difference in average outcome within the top k, times k"
      Qini curve   =  y_t  -  y_c * n_t / n_c
          "treated total, minus the control total rescaled to the same size"

    The two differ only in how they correct for the top k containing unequal
    numbers of treated and control customers; in a 50/50 experiment they are
    nearly identical.

    Ties: customers with identical scores cannot be ordered, so the curve is
    only evaluated at cut-offs between distinct scores (the same convention
    scikit-uplift uses). This matters for the bootstrap, which duplicates rows.
    """
    y, t, score = np.asarray(y, float), np.asarray(t, float), np.asarray(score, float)
    order = np.argsort(-score, kind="stable")
    y, t, score = y[order], t[order], score[order]

    cut = np.r_[np.nonzero(np.diff(score))[0], len(score) - 1]  # last row of each tied block
    k = cut + 1.0
    n_t = np.cumsum(t)[cut]
    n_c = k - n_t
    y_t = np.cumsum(y * t)[cut]
    y_c = np.cumsum(y * (1 - t))[cut]

    def safe_div(a, b):  # 0 where a group is still empty at the very top of the ranking
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    uplift = (safe_div(y_t, n_t) - safe_div(y_c, n_c)) * k
    qini = y_t - y_c * safe_div(n_t, n_c)

    frac = np.r_[0.0, k / len(score)]
    uplift = np.r_[0.0, uplift] / len(score)   # per customer in the population
    qini = np.r_[0.0, qini] / t.sum()          # per treated customer
    return frac, uplift, qini


def _area(x, y) -> float:
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))  # trapezoid rule


def uplift_metrics(y, t, score) -> dict:
    """Summary numbers for one model on one sample.

    auuc              Area under the uplift curve. Higher = more of the
                      incremental outcome captured early in the ranking.
    auuc_random       What a random ranking scores: the area under the
                      straight line from 0 to the ATE, i.e. ATE / 2.
    auuc_vs_random    auuc - auuc_random. THE number to look at: positive
                      means the ranking beats random targeting.
    qini              Qini coefficient: area between the Qini curve and its
                      random line. Same idea, Qini bookkeeping.
    """
    frac, uplift, qini = uplift_and_qini_curves(y, t, score)
    auuc = _area(frac, uplift)
    auuc_random = uplift[-1] / 2
    return {"auuc": auuc, "auuc_random": auuc_random, "auuc_vs_random": auuc - auuc_random,
            "qini": _area(frac, qini) - qini[-1] / 2}


def paired_bootstrap(y, t, scores: dict, n_boot: int = cfg.N_BOOTSTRAP,
                     seed: int = cfg.RANDOM_SEED) -> dict:
    """Bootstrap the test set and score EVERY model on the SAME resamples.

    Returns {model: {metric: array of length n_boot}}.

    Why paired? All models are judged on the same customers, so their scores
    move together: a resample that happens to contain a few big spenders in
    the treated group raises every model's curve. Comparing two models by
    whether their separate CIs overlap ignores that shared luck and is far too
    conservative. Taking the *difference* within each resample cancels it.

    Each arm is resampled separately (mirrors the design: arm sizes fixed).

    What this does and does not capture: the models are trained once and held
    fixed, so these intervals reflect uncertainty from the finite TEST set
    only, not from retraining on a different training sample (Phase 4).
    """
    rng = np.random.default_rng(seed)
    y, t = np.asarray(y, float), np.asarray(t)
    treated_rows, control_rows = np.nonzero(t == 1)[0], np.nonzero(t == 0)[0]
    out = {name: {m: np.empty(n_boot) for m in ("auuc", "auuc_random", "auuc_vs_random", "qini")}
           for name in scores}
    for b in range(n_boot):
        rows = np.r_[rng.choice(treated_rows, len(treated_rows)),
                     rng.choice(control_rows, len(control_rows))]
        for name, s in scores.items():
            for metric, value in uplift_metrics(y[rows], t[rows], s[rows]).items():
                out[name][metric][b] = value
    return out


def summarize_draws(draws: np.ndarray, alpha: float = cfg.ALPHA) -> dict:
    """Percentile CI and a two-sided bootstrap p-value for 'the true value is 0'."""
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return {"ci_low": lo, "ci_high": hi, "boot_p": min(p, 1.0)}
