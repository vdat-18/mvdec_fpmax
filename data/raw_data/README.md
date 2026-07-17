# Raw Tiki storefront data — provenance notes

## Files

- `seller_store_urls.csv` — seller storefront URLs used during collection.
- `tiki_merged_data_repaired.xlsx` (~36 MB) — recovered raw crawl, originally
  named `Merged Data (Repaired).xlsx`. Data were collected from publicly
  accessible Tiki storefront pages. Three sheets:
  - `Sheet1` — 1,823 shops x 16 columns, one row per shop (aggregated). This is
    the sheet the preprocessing pipeline reads.
  - `Sheet4` — 274,154 rows, product-level records joined with shop attributes
    (per-product revenue; star columns spelled `1 Star..5 Stars`).
  - `Sheet6` — 274,154 rows, product-level details (name, link, price, sold
    quantity, product rating).
  Verified: per-shop sums of Sheet4 `Product Revenue` reproduce Sheet1
  `Total Revenue` exactly.

## Known defect — do NOT regenerate the canonical CSV from this file

`Sheet1` row 139 ("House Of Luggage") has `Year Joined = 0` (likely an Excel
repair artifact). Because of this and small NaN-pattern differences, running
`data_preprocessing` on this file yields 1,798 rows and slightly shifted
z-scores — it does **not** reproduce the canonical
`data/preprocessed_data/tiki_preprocessed.csv` (1,799 rows), which was built
from the original `Tiki_raw_data.xlsx` (no longer available). Keep the
canonical CSV as the source of truth.

## Row mapping for cluster interpretation

`data/preprocessed_data/tiki_row_mapping.csv` maps every row of the canonical
`tiki_preprocessed.csv` to its shop in this workbook: 1,799/1,799 rows matched
1-to-1 (nearest-neighbour on the six stable z-scored features, max distance
0.0011; ties resolved by integer `Years Joined` recovered by inverting the
z-score's affine transform, zero conflicts). Columns include `csv_row_index`,
`merged_row_index`, `Store Name`, `Seller Link`, and raw feature values — use
it to profile clusters in real units.
