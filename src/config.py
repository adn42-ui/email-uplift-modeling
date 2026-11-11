"""Project-wide constants: paths, column names, and plot styling.

Everything a later phase might want to change lives here so there is one
place to look.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------- columns
TREATMENT_COL = "segment"
CONTROL = "No E-Mail"
MENS = "Mens E-Mail"
WOMENS = "Womens E-Mail"
ARMS = [CONTROL, MENS, WOMENS]
TREATED_ARMS = [MENS, WOMENS]

# Outcomes, all measured in the two weeks after the email was sent.
OUTCOMES = ["visit", "conversion", "spend"]
BINARY_OUTCOMES = ["visit", "conversion"]

# Pre-treatment covariates (everything known about a customer before the send).
NUMERIC_FEATURES = ["recency", "history"]          # months since last purchase, $ spent in past year
BINARY_FEATURES = ["mens", "womens", "newbie"]     # bought mens / womens last year, new customer in past 12 months
CATEGORICAL_FEATURES = ["zip_code", "channel"]     # Urban/Suburban/Rural, Phone/Web/Multichannel
# `history_segment` is just a binned copy of `history`. It is left out of the
# Phase 1 regressions (it would be collinear with `history`) but is included in
# the Phase 2 uplift models, which are tree-based and unharmed by redundancy.
REDUNDANT_FEATURES = ["history_segment"]

# The 8 pre-treatment covariates: the ONLY inputs any uplift model may use.
PRE_TREATMENT_COVARIATES = ["recency", "history_segment", "history", "mens", "womens",
                            "zip_code", "newbie", "channel"]

# Phase 2 settings
TEST_SIZE = 0.30
N_BOOTSTRAP = 2000
MODEL_COLORS = {"S-learner": "#2a78d6", "T-learner": "#eb6834",
                "X-learner": "#1baf7a", "Causal forest": "#eda100"}

RANDOM_SEED = 42
ALPHA = 0.05

# ---------------------------------------------------------------- Phase 3: business assumptions
# NEITHER number is in the data. They are assumptions; change them here and
# re-run `python -m src.phase3_policy`.
MARGIN = 0.30            # profit earned per $1 of incremental revenue
COST_PER_EMAIL = 0.05    # dollars per email sent
# Sensitivity grid shown in the report.
MARGIN_GRID = [0.20, 0.30, 0.50]
COST_GRID = [0.01, 0.05, 0.10]
# Share of customers emailed under each model-targeting policy.
TARGET_SHARES = [0.30, 0.50, 0.70, 0.90]
# Group size for the persuadables profile (top vs. bottom by predicted uplift).
PROFILE_SHARE = 0.30
# Policy scores: label -> column in reports/tables/phase2_test_predictions.csv
POLICY_SCORES = {
    "Causal forest (visit)": "uplift_visit__Causal forest",   # primary
    "X-learner (visit)": "uplift_visit__X-learner",           # sensitivity check
    "Causal forest (spend)": "uplift_spend__Causal forest",   # comparison
}
POLICY_COLORS = {"Causal forest (visit)": "#eda100", "X-learner (visit)": "#1baf7a",
                 "Causal forest (spend)": "#6250d6"}

# ---------------------------------------------------------------- Phase 4: robustness
CACHE_DIR = ROOT / "cache"          # out-of-fold scores (slow to recompute)
CV_REPEATS = 5                      # each repeat reshuffles the folds with a new seed
CV_FOLDS = 5
# Policies fixed BEFORE looking at Phase 4 results: email everyone, email no
# one, model top 30/50/70/90% (TARGET_SHARES above), and the no-ML rule
# "email everyone except customers who bought only men's merchandise".
SIM_DRAWS = 200                     # repetitions of the fake-observational subsample
SIM_PROPENSITY_RANGE = (0.2, 0.8)   # keep simulated propensities moderate

# ---------------------------------------------------------------- plot style
# One fixed colour per arm, used in every figure of the project.
ARM_COLORS = {MENS: "#2a78d6", WOMENS: "#eb6834", CONTROL: "#52514e"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"


def apply_plot_style():
    """Quiet matplotlib defaults: recessive grid and axes, readable text."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED, "text.color": INK,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "font.size": 10, "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160,
    })
