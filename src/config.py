"""Project paths and shared constants."""

from pathlib import Path

RANDOM_STATE = 42
H_FUSED_COLUMNS = [f"fused_{index}" for index in range(1, 12)]

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed_data"
OUTPUT_DIR = PROJECT_DIR / "output"

PREPROCESSED_DATA_PATH = PREPROCESSED_DATA_DIR / "tiki_preprocessed.csv"
FUSED_REPRESENTATION_PATH = PREPROCESSED_DATA_DIR / "fused_representation.pkl"
WITHOUT_FFS_KPROTOTYPES_RESULTS_PATH = OUTPUT_DIR / "without_ffs_results.csv"
FFS_RESULTS_PATH = OUTPUT_DIR / "ffs_results.csv"
WITHOUT_FFS_INTUITIVE_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "without_ffs_intuitive_native_results.csv"
)
WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "without_ffs_intuitive_view_weighted_native_results.csv"
)
FFS_INTUITIVE_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "ffs_intuitive_native_results.csv"
)
FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "ffs_intuitive_view_weighted_native_results.csv"
)
WITHOUT_FFS_INTUITIVE_PAPER_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "without_ffs_intuitive_paper_native_results.csv"
)
WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_PAPER_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "without_ffs_intuitive_view_weighted_paper_native_results.csv"
)
FFS_INTUITIVE_PAPER_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "ffs_intuitive_paper_native_results.csv"
)
FFS_INTUITIVE_VIEW_WEIGHTED_PAPER_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "ffs_intuitive_view_weighted_paper_native_results.csv"
)
WITHOUT_FFS_INTUITIVE_EXHAUSTIVE_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "without_ffs_intuitive_exhaustive_native_results.csv"
)
WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_EXHAUSTIVE_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "without_ffs_intuitive_view_weighted_exhaustive_native_results.csv"
)
FFS_INTUITIVE_EXHAUSTIVE_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "ffs_intuitive_exhaustive_native_results.csv"
)
FFS_INTUITIVE_VIEW_WEIGHTED_EXHAUSTIVE_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "ffs_intuitive_view_weighted_exhaustive_native_results.csv"
)
WITHOUT_FFS_INTUITIVE_PAPER_EXHAUSTIVE_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "without_ffs_intuitive_paper_exhaustive_native_results.csv"
)
WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_PAPER_EXHAUSTIVE_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR
    / "without_ffs_intuitive_view_weighted_paper_exhaustive_native_results.csv"
)
FFS_INTUITIVE_PAPER_EXHAUSTIVE_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "ffs_intuitive_paper_exhaustive_native_results.csv"
)
FFS_INTUITIVE_VIEW_WEIGHTED_PAPER_EXHAUSTIVE_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "ffs_intuitive_view_weighted_paper_exhaustive_native_results.csv"
)
POST_FFS_INTUITIVE_RESULTS_PATH = OUTPUT_DIR / "post_ffs_intuitive_results.csv"
POST_FFS_INTUITIVE_VIEW_WEIGHTED_RESULTS_PATH = (
    OUTPUT_DIR / "post_ffs_intuitive_view_weighted_results.csv"
)
POST_WITHOUT_FFS_INTUITIVE_RESULTS_PATH = (
    OUTPUT_DIR / "post_without_ffs_intuitive_results.csv"
)
POST_WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_RESULTS_PATH = (
    OUTPUT_DIR / "post_without_ffs_intuitive_view_weighted_results.csv"
)
POST_FFS_INTUITIVE_BEST_RESULTS_PATH = (
    OUTPUT_DIR / "post_ffs_intuitive_best_results.csv"
)
POST_FFS_INTUITIVE_VIEW_WEIGHTED_BEST_RESULTS_PATH = (
    OUTPUT_DIR / "post_ffs_intuitive_view_weighted_best_results.csv"
)
POST_FFS_INTUITIVE_EXHAUSTIVE_BEST_RESULTS_PATH = (
    OUTPUT_DIR / "post_ffs_intuitive_exhaustive_best_results.csv"
)
POST_FFS_INTUITIVE_VIEW_WEIGHTED_EXHAUSTIVE_BEST_RESULTS_PATH = (
    OUTPUT_DIR / "post_ffs_intuitive_view_weighted_exhaustive_best_results.csv"
)
POST_WITHOUT_FFS_INTUITIVE_BEST_RESULTS_PATH = (
    OUTPUT_DIR / "post_without_ffs_intuitive_best_results.csv"
)
POST_WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_BEST_RESULTS_PATH = (
    OUTPUT_DIR / "post_without_ffs_intuitive_view_weighted_best_results.csv"
)
POST_WITHOUT_FFS_INTUITIVE_EXHAUSTIVE_BEST_RESULTS_PATH = (
    OUTPUT_DIR / "post_without_ffs_intuitive_exhaustive_best_results.csv"
)
POST_WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_EXHAUSTIVE_BEST_RESULTS_PATH = (
    OUTPUT_DIR / "post_without_ffs_intuitive_view_weighted_exhaustive_best_results.csv"
)
