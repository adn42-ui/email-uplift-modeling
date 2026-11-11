"""Valuing a targeting policy from randomized data.

A policy is a rule that says, for each customer, "email" or "don't email".
We want to know how much incremental revenue the policy would earn if it were
rolled out to everyone.

THE RULE: a policy is valued with the customers' ACTUAL outcomes from the
experiment, never with the model's own predicted uplift. The model only
decides WHO is targeted. (Valuing a model's policy with that same model's
predictions would be marking its own homework: an over-confident model would
simply declare itself profitable.)

How, given that each customer was either emailed or not?

  Take the customers the policy would email. Among them, some were (by the
  coin flip of the experiment) actually emailed and some were not. Because
  the coin flip ignored everything about the customer, comparing those two
  sub-groups estimates the email's effect on exactly the people the policy
  targets. Customers the policy would NOT email contribute zero: under the
  policy they get no email, so there is no incremental revenue from them.

Inverse propensity weighting (IPW) is the bookkeeping for that comparison.
With a known propensity e = P(emailed), each customer gets a signed,
re-weighted outcome

      g_i = + Y_i / e          if actually emailed
      g_i = - Y_i / (1 - e)    if actually in control

The mean of g over ALL customers is the usual treated-minus-control
difference, i.e. the ATE. The mean of g over only the customers a policy
targets (counting the rest as zero) is that policy's incremental revenue per
customer in the population. With e = 0.5 this is just "2 x spend, plus for
emailed, minus for control".
"""
import numpy as np

from . import config as cfg

EVERYONE = "Email everyone"


def ipw_contributions(y, t, propensity: float) -> np.ndarray:
    """Per-customer signed, weighted outcome g_i (see module docstring)."""
    y, t = np.asarray(y, float), np.asarray(t, float)
    return t * y / propensity - (1 - t) * y / (1 - propensity)


def top_share_mask(score, share: float) -> np.ndarray:
    """True for the `share` of customers with the highest score."""
    score = np.asarray(score)
    n_target = int(round(share * len(score)))
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(-score, kind="stable")[:n_target]] = True
    return mask


def incremental_revenue(g: np.ndarray, targeted: np.ndarray) -> float:
    """Incremental revenue per customer IN THE WHOLE POPULATION if the
    targeted customers are emailed and nobody else is."""
    return float((g * targeted).mean())


def profit_per_1000(inc_revenue_per_customer, share_emailed, margin: float = cfg.MARGIN,
                    cost_per_email: float = cfg.COST_PER_EMAIL):
    """Profit per 1,000 customers = margin x incremental revenue - email cost.

    Spend is revenue, not profit, so it is multiplied by the margin before the
    cost of sending is subtracted. Works on numbers or arrays of bootstrap draws.
    """
    return 1000 * (margin * inc_revenue_per_customer - cost_per_email * share_emailed)


def policy_incremental_revenue(y, t, scores: dict, shares=cfg.TARGET_SHARES) -> dict:
    """Point estimates: {(score_name, share): incremental revenue per customer},
    plus {EVERYONE: ATE}. Propensity = the actual treated share of this sample."""
    t = np.asarray(t)
    g = ipw_contributions(y, t, t.mean())
    out = {EVERYONE: float(g.mean())}
    for name, s in scores.items():
        for share in shares:
            out[(name, share)] = incremental_revenue(g, top_share_mask(s, share))
    return out


def bootstrap_policies(y, t, scores: dict, shares=cfg.TARGET_SHARES,
                       n_boot: int = cfg.N_BOOTSTRAP, seed: int = cfg.RANDOM_SEED) -> dict:
    """Bootstrap the test set and value EVERY policy on the SAME resamples.

    Returns {policy: array of n_boot incremental-revenue draws}, with the same
    keys as `policy_incremental_revenue`.

    Paired on purpose: the question is the DIFFERENCE between a targeted
    policy and "email everyone", and both are driven by the same handful of
    big purchases. Valuing both on each resample and subtracting cancels that
    shared noise. The random-targeting baseline is derived from the
    "email everyone" draw of the same resample (share x that value), so
    model-vs-random differences are paired too.

    Each arm is resampled separately, so the propensity is the same constant
    in every resample. Cut-offs are recomputed inside each resample, because
    "the top 30%" is defined relative to whichever customers are present.
    """
    rng = np.random.default_rng(seed)
    y, t = np.asarray(y, float), np.asarray(t)
    treated_rows, control_rows = np.nonzero(t == 1)[0], np.nonzero(t == 0)[0]
    g = ipw_contributions(y, t, t.mean())

    keys = [EVERYONE] + [(name, share) for name in scores for share in shares]
    draws = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        rows = np.r_[rng.choice(treated_rows, len(treated_rows)),
                     rng.choice(control_rows, len(control_rows))]
        g_b = g[rows]
        draws[EVERYONE][b] = g_b.mean()
        for name, s in scores.items():
            s_b = s[rows]
            for share in shares:
                draws[(name, share)][b] = incremental_revenue(g_b, top_share_mask(s_b, share))
    return draws
