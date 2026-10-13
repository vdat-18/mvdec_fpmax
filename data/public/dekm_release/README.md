# Bundled DEKM release fallbacks

This directory contains the two small RCV1 split-index files published by the
DEKM 2021 authors. They are retained byte-for-byte as a verified fallback when
Google Drive throttles automated downloads.

| File | Google Drive file ID | SHA-256 |
| --- | --- | --- |
| `RCV1/test` | `1WzfCSj8yAgnHsb5tiIQRmw6uwz03td5K` | `26c22ad47eda2c283a5481c2f705da61f862c0b7b457941a1564b07f605133bf` |
| `RCV1/validation` | `1d-ZJ5SxneLkTYiRUAQ5y1Ov0ZGuAyLvQ` | `a1a689eb7f808e80d57dab694e57d98fe92a9ba3ef2b787e7f33547a03e9cf92` |

The full RCV1 feature matrix is still obtained through scikit-learn. These files
only preserve the paper release's sample selection and order.
