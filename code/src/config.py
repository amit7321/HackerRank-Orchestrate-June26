"""Paths, model names, and rate-limit constants for the claim-review pipeline."""

from pathlib import Path

# code/src/config.py -> parents[0]=src parents[1]=code parents[2]=repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
DATASET_DIR = REPO_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"

CLAIMS_CSV = DATASET_DIR / "claims.csv"
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"

OUTPUT_CSV = REPO_ROOT / "output.csv"

CACHE_DIR = CODE_DIR / ".cache"

# --- Model configuration -----------------------------------------------
# Config A is the primary strategy used to produce output.csv.
# Config B is the comparison strategy reported in evaluation/evaluation_report.md.
#
# As of this writing, free-tier daily quotas are extremely asymmetric: the
# full "flash" tier (including the gemini-flash-latest alias, which
# currently resolves to a brand-new preview-class model) is capped at ~20
# requests/day on the free tier, while "flash-lite" models get ~500/day.
# This pipeline needs ~150-250 calls to process the full 44-row test set
# plus the sample evaluation, so both configs deliberately use flash-lite
# models -- a full "flash" model's free-tier quota can't cover a single
# real run. Pinned, dated model names are used (not the "-latest" alias)
# so the daily quota and pricing are predictable; swap these if Google
# retires either model from new-user access.
MODEL_CONFIG_A = {
    "name": "config_a_flash_lite_35",
    "model": "gemini-3.5-flash-lite",
}
MODEL_CONFIG_B = {
    "name": "config_b_flash_lite_31",
    "model": "gemini-3.1-flash-lite",
}
DEFAULT_CONFIG = MODEL_CONFIG_A

TEMPERATURE = 0.0

# Prompt/schema version. Bump this to invalidate the disk cache after a
# meaningful prompt change.
PROMPT_VERSION = "v1"

# --- Free-tier rate limiting ---------------------------------------------
# Conservative token-bucket cap, comfortably under typical free-tier RPM
# limits for the flash-lite model family. The binding constraint is the
# ~500/day-per-model quota, not RPM; this cap just smooths call bursts.
MAX_REQUESTS_PER_MINUTE = 8
MAX_RETRY_ATTEMPTS = 5
RETRY_MIN_WAIT_SECONDS = 2
RETRY_MAX_WAIT_SECONDS = 30

# Approximate paid-rate pricing (USD per 1M tokens), used only for the
# operational cost estimate in the evaluation report. Free tier costs $0;
# this is a reference figure in case free-tier quota runs out.
APPROX_PRICE_PER_1M_INPUT_TOKENS = {
    "gemini-3.5-flash-lite": 0.30,
    "gemini-3.1-flash-lite": 0.25,
}
APPROX_PRICE_PER_1M_OUTPUT_TOKENS = {
    "gemini-3.5-flash-lite": 2.50,
    "gemini-3.1-flash-lite": 1.50,
}
