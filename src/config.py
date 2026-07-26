"""Project paths and shared constants."""

from pathlib import Path

RANDOM_STATE = 42
KPROTOTYPES_N_INIT = 10
H_FUSED_COLUMNS = [f"fused_{index}" for index in range(1, 12)]

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed_data"
OUTPUT_DIR = PROJECT_DIR / "output"
DEKM_DATASET_DIR = PROJECT_DIR / "external_repos" / "DEKM" / "datasets"
PUBLIC_BENCHMARK_OUTPUT_DIR = OUTPUT_DIR / "public_benchmark"

PREPROCESSED_DATA_PATH = PREPROCESSED_DATA_DIR / "tiki_preprocessed.csv"
WITHOUT_FFS_KPROTOTYPES_RESULTS_PATH = OUTPUT_DIR / "mimvdec_without_ffs_results.csv"
FFS_RESULTS_PATH = OUTPUT_DIR / "mimvdec_with_ffs_results.csv"
WITHOUT_FFS_INTUITIVE_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "mimvdec_intuitive_without_ffs_results.csv"
)
WITHOUT_FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "mimvdec_intuitive_view_weighted_without_ffs_results.csv"
)
FFS_INTUITIVE_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "mimvdec_intuitive_with_ffs_results.csv"
)
FFS_INTUITIVE_VIEW_WEIGHTED_NATIVE_RESULTS_PATH = (
    OUTPUT_DIR / "mimvdec_intuitive_view_weighted_with_ffs_results.csv"
)
