import pandas as pd

from pipeline.fpmax import extract_fpmax_features


def test_extract_fpmax_features_builds_itemset_columns():
    df = pd.DataFrame(
        {
            "x": [0.0, 0.1, 0.2, 10.0, 10.1, 10.2],
            "y": [0.0, 0.1, 0.2, 10.0, 10.1, 10.2],
        }
    )

    result = extract_fpmax_features(
        df=df,
        n_bins=3,
        strategy="uniform",
        min_support=0.4,
        drop_original_numeric=True,
    )

    assert not result.itemsets.empty
    assert not result.features.empty
    assert result.features.shape[0] == len(df)
    assert all(column in {0, 1} for column in result.features.to_numpy().ravel())
