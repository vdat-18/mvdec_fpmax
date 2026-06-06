"""Project paths and shared constants."""

from pathlib import Path

RANDOM_STATE = 42
H_FUSED_COLUMNS = [
    "followers",
    "revenue",
    "yearsjoined",
    "rating",
    "goodreview",
    "mediumreview",
    "badreview",
    "latent1",
    "latent2",
    "latent3",
    "latent4",
]

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed_data"
OUTPUT_DIR = PROJECT_DIR / "output"

PREPROCESSED_DATA_PATH = PREPROCESSED_DATA_DIR / "tiki_preprocessed.csv"
FUSED_REPRESENTATION_PATH = PREPROCESSED_DATA_DIR / "fused_representation.pkl"
WITHOUT_FFS_RESULTS_PATH = OUTPUT_DIR / "without_ffs_results.csv"
FFS_RESULTS_PATH = OUTPUT_DIR / "ffs_results.csv"
