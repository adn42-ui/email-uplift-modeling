"""Four ways to estimate the conditional average treatment effect (CATE):

    tau(x) = E[ Y(1) - Y(0) | X = x ]

"For customers who look like x, how much does the email change the outcome,
on average?" Nobody's individual effect is ever observed (each customer is
either emailed or not), so every method below is a different strategy for
getting at tau(x) from data where only one of Y(1), Y(0) is seen per person.

The three meta-learners are written out by hand, on top of an ordinary
gradient-boosting model, so every step is visible. All of them share the same
base learner and hyperparameters: differences in results then come from the
*strategy*, not from one model being better tuned. The causal forest comes
from econml.

All four expose the same interface:  .fit(X, t, y)  and  .predict(X) -> tau.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from . import config as cfg

# Deliberately conservative (shallow trees, big leaves, slow learning rate).
# The treatment-effect signal here is tiny relative to outcome noise, so a
# flexible base learner mostly memorises noise. Fixed up front, not tuned on
# the test set.
GBM_PARAMS = dict(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=100)


def _outcome_model(binary: bool, seed: int):
    """A model for E[Y | features]. Classifier for 0/1 outcomes, regressor otherwise."""
    Model = HistGradientBoostingClassifier if binary else HistGradientBoostingRegressor
    return Model(**GBM_PARAMS, random_state=seed)


def _expected_outcome(model, X) -> np.ndarray:
    """E[Y | X]: a probability for classifiers, a plain prediction for regressors."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


class SLearner:
    """S-learner ("single" model).

    How it works
        Fit ONE model of the outcome on the features plus the treatment flag:
        mu(x, t). To get a customer's uplift, ask the model twice, once
        pretending they were emailed and once pretending they were not:
            tau(x) = mu(x, 1) - mu(x, 0)

    When it works
        Simple, uses all the data in one model, hard to badly overfit.

    When it fails
        The treatment flag is just one feature among many. If the email's
        effect is small next to the effect of, say, purchase history, a
        regularised model can mostly ignore it, or use it only as a constant
        shift. Estimated uplift is then biased toward zero and toward being
        the same for everyone ("regularisation bias"). Heterogeneity only
        appears if the trees happen to split on treatment *inside* a
        segment.
    """

    def __init__(self, binary_outcome: bool, seed: int = cfg.RANDOM_SEED):
        self.model = _outcome_model(binary_outcome, seed)

    def fit(self, X, t, y):
        self.model.fit(np.column_stack([X, t]), y)
        return self

    def predict(self, X):
        n = len(X)
        as_treated = _expected_outcome(self.model, np.column_stack([X, np.ones(n)]))
        as_control = _expected_outcome(self.model, np.column_stack([X, np.zeros(n)]))
        return as_treated - as_control


class TLearner:
    """T-learner ("two" models).

    How it works
        Fit one outcome model on emailed customers only, mu1(x), and a
        separate one on control customers only, mu0(x). Then
            tau(x) = mu1(x) - mu0(x)

    When it works
        When the two groups really do respond differently and there is plenty
        of data in each. The treatment can never be "ignored" as in the
        S-learner.

    When it fails
        The two models are fit independently, each with its own errors, and
        nothing asks them to agree where the truth is the same. Subtracting
        them adds their noise together: places where mu1 happened to overfit
        upward and mu0 downward look like high-uplift segments. With a small
        true effect, much of the estimated "heterogeneity" can be exactly
        this noise. Each model also sees only about half the data.
    """

    def __init__(self, binary_outcome: bool, seed: int = cfg.RANDOM_SEED):
        self.model_treated = _outcome_model(binary_outcome, seed)
        self.model_control = _outcome_model(binary_outcome, seed)

    def fit(self, X, t, y):
        self.model_treated.fit(X[t == 1], y[t == 1])
        self.model_control.fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        return _expected_outcome(self.model_treated, X) - _expected_outcome(self.model_control, X)


class XLearner:
    """X-learner (Kunzel et al., 2019).

    How it works
        Stage 1: fit mu1 and mu0 exactly as in the T-learner.
        Stage 2: use them to fill in each customer's *missing* outcome and so
        build a per-customer "imputed effect":
            emailed customers:  D1 = Y - mu0(x)   (what happened minus what
                                                   we predict without email)
            control customers:  D0 = mu1(x) - Y   (what we predict with email
                                                   minus what happened)
        Stage 3: fit two regressions directly on those imputed effects,
        tau1(x) on the emailed group and tau0(x) on the control group.
        Final answer: a propensity-weighted blend
            tau(x) = e * tau0(x) + (1 - e) * tau1(x)
        where e = P(treated).

    Propensity
        This is a randomized experiment, so e is KNOWN: it is the share of
        customers assigned to the email (about 0.5). We plug in that constant
        rather than estimating a propensity model. Estimating it would only
        add noise by fitting chance imbalances.

    When it works
        Stage 3 models the effect itself as the regression target, so it can
        be smooth even when the outcome surfaces are not. It was designed for
        very unequal arm sizes, where the big arm's outcome model is accurate
        and can be lent to the small arm.

    When it fails
        The imputed effects inherit every error in the stage-1 models, so if
        mu0 and mu1 are poor, stage 3 learns their mistakes. With arms of
        equal size (as here) its main advantage is absent, and it often
        behaves like a smoothed T-learner.
    """

    def __init__(self, binary_outcome: bool, propensity: float, seed: int = cfg.RANDOM_SEED):
        self.propensity = propensity
        self.model_treated = _outcome_model(binary_outcome, seed)
        self.model_control = _outcome_model(binary_outcome, seed)
        # Imputed effects are continuous even when the outcome is 0/1.
        self.effect_model_treated = _outcome_model(False, seed)
        self.effect_model_control = _outcome_model(False, seed)

    def fit(self, X, t, y):
        X1, y1, X0, y0 = X[t == 1], y[t == 1], X[t == 0], y[t == 0]
        self.model_treated.fit(X1, y1)
        self.model_control.fit(X0, y0)
        imputed_effect_treated = y1 - _expected_outcome(self.model_control, X1)
        imputed_effect_control = _expected_outcome(self.model_treated, X0) - y0
        self.effect_model_treated.fit(X1, imputed_effect_treated)
        self.effect_model_control.fit(X0, imputed_effect_control)
        return self

    def predict(self, X):
        e = self.propensity
        return e * self.effect_model_control.predict(X) + (1 - e) * self.effect_model_treated.predict(X)


class KnownPropensity(ClassifierMixin, BaseEstimator):
    """A 'model' of P(treated | x) that always returns the same known constant.

    econml's causal forest expects a treatment model. In a randomized
    experiment the true propensity is known by design, so we hand it this
    stand-in instead of a fitted classifier.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])

    def predict(self, X):
        return np.full(len(X), self.classes_[int(self.p >= 0.5)])


class CausalForest:
    """Causal forest (Athey, Tibshirani & Wager, 2019), via econml's
    CausalForestDML.

    How it works
        Step 1 ("residualise"): predict the outcome from features alone,
        m(x) = E[Y | x], and subtract it, leaving the part of the outcome the
        features cannot explain. Do the same for treatment: T - e, with e the
        known propensity. This strips out "who buys a lot anyway" so the
        forest can concentrate on "who responds to the email".
        Step 2: grow many trees whose splits are chosen to make the
        *treatment effect* differ as much as possible between the two child
        nodes (an ordinary forest instead makes the *outcome* differ).
        Step 3 ("honesty"): each tree uses one half of its sample to choose
        splits and the other half to estimate effects in the leaves, so a
        leaf's estimate is not flattered by having been selected for looking
        extreme.
        tau(x) is a weighted treated-minus-control comparison among training
        customers that tend to share leaves with x.

    When it works
        It is the only method here built to target tau(x) directly, and
        honesty makes it conservative: it tends not to invent heterogeneity.

    When it fails
        Like any forest it gives piecewise-constant, fairly smooth estimates
        and needs a lot of data per leaf when effects are small, so it can
        miss real but subtle heterogeneity. Results depend on leaf size and
        on how good the step-1 outcome model is.
    """

    def __init__(self, propensity: float, seed: int = cfg.RANDOM_SEED):
        from econml.dml import CausalForestDML

        self.model = CausalForestDML(
            model_y=HistGradientBoostingRegressor(**GBM_PARAMS, random_state=seed),
            model_t=KnownPropensity(propensity),
            discrete_treatment=True,
            n_estimators=1000,
            min_samples_leaf=100,   # large leaves: the effect signal is weak
            cv=5,                   # cross-fitting folds for the step-1 models
            random_state=seed,
        )

    def fit(self, X, t, y):
        self.model.fit(y, t, X=X)
        return self

    def predict(self, X):
        return self.model.effect(X).ravel()


def build_models(binary_outcome: bool, propensity: float, seed: int = cfg.RANDOM_SEED) -> dict:
    """The four models, in the fixed order used in every table and figure."""
    return {
        "S-learner": SLearner(binary_outcome, seed),
        "T-learner": TLearner(binary_outcome, seed),
        "X-learner": XLearner(binary_outcome, propensity, seed),
        "Causal forest": CausalForest(propensity, seed),
    }
