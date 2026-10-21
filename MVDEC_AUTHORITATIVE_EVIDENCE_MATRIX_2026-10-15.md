# MvDEC Authoritative Evidence Matrix (2026-10-15)

## Decision

Exact-paper reproduction is not currently possible from the disclosed material. This is an evidence limitation, not a remaining local execution bug.

The blocking reasons are independent and cumulative:

- The paper gives mutually incompatible fusion definitions: Figure 1 averages two full per-view outputs, Figure 2 concatenates two 23-dimensional outputs into 46 dimensions, and Equation 4 averages equal-dimensional encoded representations.
- The paper says the autoencoders use different attribute sets, but Equation 3 and Figure 2 feed the same input to both branches and disclose no feature partition.
- Public-dataset experimental settings are incomplete: no public seed policy, run count, initialization, learning rate, checkpoint-selection rule, variance, or reproducible definition of "best performances" is provided.
- Table 1 visibly prints an inconsistent REUTERS sample count of 210 despite naming REUTERS-10K and reusing DEKM results based on the full benchmark.
- The air-pollution row-level matrix, preprocessing outputs, fitted models, cluster assignments, and plotting inputs are not public. The paper only says data will be available on request.
- The supplied Google Drive is the user's personal reference folder, not an author release. It contains 16 PDF files and therefore does not supply the missing code, data, notebooks, logs, weights, manifests, or preprocessing artifacts.

The repository does have a valid controlled reproduction of one explicitly declared reconstruction protocol. It does not reproduce the paper result and must not be relabeled as exact MvDEC.

## Evidence Sources

- MvDEC PDF visually inspected page by page: tmp/pdfs/mvdec_paper/page-01.png through page-18.png.
- DEKM PDF visually inspected for the method and experimental protocol: tmp/pdfs/dekm_paper/page-6.png through page-9.png.
- User reference-folder inventory: docs/audit_evidence/mvdec_public_folder_12nHBje7FdWom7WVAUK9254gHyvFdt33G/inventory.json.
- Controlled REUTERS run: artifacts/kaggle/mvdec-2025-reuters-release-l2-seed42/controlled-reproduction-report.md.
- Post-training fusion diagnostic: artifacts/kaggle/mvdec-2025-reuters-paper-figure-audit-seed42/version7/mvdec-reuters-paper-figure-audit-seed42/paper_figure_diagnostic_metrics.csv.
- Repository behavior: src/pipeline/mvdec_contract.py, src/pipeline/mvdec_fusion_audit.py, and src/pipeline/mvdec_paper_figure_audit.py.

Status vocabulary:

- exact: directly verified from the rendered paper, an immutable artifact, or an independently recomputed result.
- copied: the MvDEC value is reused from DEKM rather than demonstrated as a newly disclosed MvDEC run.
- inferred: a reasonable interpretation, but not explicitly or uniquely specified.
- contradictory: two disclosed claims cannot both define the same implementation or dataset.
- missing: the required artifact or setting was not disclosed.
- unreproducible: the claim cannot be regenerated exactly from the disclosed evidence.

## Public Benchmarks And Method

| Paper element | Paper claim | DEKM evidence | Available artifact evidence | Repository behavior | Reproduction result | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Page 5, Figure 1 | Two model outputs enter an AVG operation. The diagram visually treats the complete outputs as the averaged objects. | DEKM has one view and does not resolve multi-view fusion. | No implementation artifact; PDFs only. | Encodes this only as the named diagnostic mvdec2025_figure1_full_output_average_diagnostic. | On the centered REUTERS seed-42 model: ACC 0.5473, NMI 0.435088; reconstruction-only average is already 0.5465, 0.431989. | exact; contradictory |
| Page 6, Figure 2 | Two 23-dimensional outputs are concatenated into a 46-dimensional final output for the 13-feature example. | DEKM does not contain this second branch or fusion layer. | No code or model graph. | Encodes this only as the named diagnostic mvdec2025_figure2_full_output_concatenation_diagnostic. | On the same model: ACC 0.5407, NMI 0.374664. | exact; contradictory |
| Page 8, Equation 4 | h_i = 1/2 (h_i^(1) + h_i^(2)); the encoded representations must have the same width. | A single 10-dimensional DEKM embedding is compatible with the dimensional premise but cannot resolve fusion. | No implementation artifact. | Canonical declared fusion contract is mvdec2025_encoder_average. | Controlled release-L2 run: fused ACC 0.7194, NMI 0.528645 versus paper 0.7813, 0.6141. | exact; contradictory |
| Page 7 prose versus Equation 3 | Prose says each autoencoder captures a different set of features, while Equation 3 sends the same x_i to both encoders. No feature indices or partition rule are disclosed. | Public DEKM provides one feature matrix per dataset. | No alternate view matrices or feature maps. | Explicitly records same_full_feature_matrix_both_views_v1; no silent feature split is invented. | No exact split-view experiment can be reconstructed. | contradictory; missing; unreproducible |
| Page 6, Figure 2 architecture | View1 visibly contains a 10-dimensional latent. View2 ends in a 23-dimensional head without clearly labeling a corresponding 10-dimensional latent. | DEKM uses d-500-500-2000-10 with a conventional mirrored decoder. | No source model or weights. | The direct-head protocol interprets the first 10 columns as latent and the final 13 as reconstruction; this is versioned as an assumption. | Architecture diagnostics show that the interpretation materially affects training and fusion. | inferred; missing |
| Pages 5 and 8, eigen direction | Page 5 says the last eigenvector contributes least; page 8 says the least-informative direction corresponds to the smallest eigenvalue. | DEKM orders eigenvalues ascending but updates the last transformed dimension; its prose is also confusing about importance. | No author clarification or source. | Largest/last and smallest/first choices are separate immutable protocols. | No direction can be identified as the official implementation from published evidence. | contradictory; unreproducible |
| Page 8, Table 1 | Printed sample counts are REUTERS 210, 20NEWS 2,000, RCV1 10,000. This is visible in the rendered PDF, not a text-extraction artifact. | DEKM defines REUTERS-10K as 10,000 rows and uses full benchmark data for its published controls. | No subset arrays, indices, or sampling manifest. | Canonical REUTERS runs use the independently fingerprinted 10,000-row arrays. | Controlled REUTERS run contains 10,000 samples; no 210-row protocol is disclosed. | exact; contradictory; missing |
| Page 9, Table 2, REUTERS | View1 76.42/59.99, View2 65.01/51.29, fused 78.13/61.41. | DEKM Table III reports DEKM_F 76.42/59.99. | No logs, assignments, weights, or run manifest supporting the MvDEC table. | Reports jointly trained View1 separately and never calls it the frozen DEKM_F control. | Controlled run gives View1 71.93/52.85, View2 46.31/18.19, fused 71.94/52.86. | exact; copied; unreproducible |
| Page 9, Table 2, 20NEWS | View1 41.20/43.76, View2 40.86/40.32, fused 41.63/45.43. | DEKM Table III reports DEKM_F 41.20/43.76. | No supporting artifacts. | Public protocols require explicit source identity and fail-closed provenance. | No disclosed evidence proves the fused row or a common run-selection policy. | exact; copied; unreproducible |
| Page 9, Table 2, RCV1 | View1 62.12/35.98, View2 62.35/36.59, fused 71.93/48.85. | DEKM Table III reports DEKM_F 62.12/35.98. | No supporting artifacts. | Public protocols keep standalone DEKM_F distinct from jointly trained View1. | No disclosed evidence proves the fused row or a common run-selection policy. | exact; copied; unreproducible |
| Table 2 View1 semantics | The prose explicitly equates Proposed method_View1 with DEKM_F. All three View1 rows exactly duplicate DEKM_F values. | DEKM reports those values as its own three-run averages. | No separate MvDEC View1 run artifacts. | Jointly trained View1 is labeled as a representation diagnostic, not DEKM_F. | Repository View1 values differ because the branch is jointly updated. | copied; contradictory |
| Public result selection | Paper describes the table as best performances but does not disclose trials, seeds, checkpoint search, variance, or label-use policy. | DEKM says datasets are run three times and Table III reports averages. | No logs or selection manifest. | Uses explicit seeds, immutable configs, provenance, and mean/sample standard deviation for preregistered matrices. | The one completed release-L2 seed-42 run is a controlled reproduction only. | missing; unreproducible |
| DEKM inherited setup | MvDEC relies on DEKM context without restating a complete public protocol. | DEKM discloses three runs, mean reporting, squared row norm divided by d is approximately 1, d-500-500-2000-10, Xavier initialization, Adam lr 0.001 with beta1 0.9 and beta2 0.999, and stopping below 0.1% assignment changes. | The user reference folder contains papers only and is not evidence of an author implementation. | Each repository protocol records the choices it actually uses rather than silently assuming all DEKM details. | Inherited settings do not resolve MvDEC-specific architecture, fusion, schedule, or selection ambiguity. | exact; inferred |
| Controlled REUTERS release-L2 run | Not a paper claim; this is the repository's explicit reconstruction gate. | Uses the DEKM REUTERS-10K source identity. | No official author artifact has been obtained to confirm equivalence. | Protocol mvdec_2025_paper_resolved_release_l2_full_batch_v1; commit c6252ad95d3fe1395ceaee3e6f26e10043c23c0f; seed 42; complete manifest and output hashes. | View1 0.7193/0.528496, View2 0.4631/0.181907, fused 0.7194/0.528645. Gap to paper: -0.0619 ACC and about -0.08546 NMI. | exact; inferred |
| Alternative post-training fusion controls | No unique paper rule covers normalization, coordinate alignment, or concatenation controls. | Not applicable. | No implementation evidence. | Runs row-L2 average, z-score average, Procrustes average, and concatenation only as diagnostics. | None recovers the paper result; best listed diagnostic is Procrustes 0.7196/0.528915 on the release-L2 run. | exact; unreproducible |

## User Reference Folder And Artifact Availability

| Paper element | Paper claim | DEKM evidence | Available artifact evidence | Repository behavior | Reproduction result | Status |
| --- | --- | --- | --- | --- | --- | --- |
| User Google Drive reference folder | Personal working/reference folder supplied by the user; it is not an author release location. | Not applicable. | Exactly 16 files, all PDFs; total 48,190,879 bytes. No code, CSV, NPY/NPZ, notebooks, logs, weights, manifests, or preprocessing outputs. | Stores a durable name/ID/size/SHA-256 inventory under docs/audit_evidence. | The folder helps review the papers but provides no arrays, labels, model states, or execution logs for reproduction. | exact |
| MvDEC PDF in user reference folder | The folder copy is a publication PDF, not a source package. | Not applicable. | Google file ID 1Xz3NnlwzGV-KmuqLp7uwODcHch24ab0I; size 5,273,595; SHA-256 6dc04788c15e13637eac09e033b5d2134c0fdc0f446e9ef4ebd3cd692b1c90d3. | The Downloads copy has the same size and SHA-256. | This proves file identity only; it adds no executable evidence. | exact; missing |
| Repository PDF copy | Not a paper scientific claim. | Not applicable. | User-folder copy hash is listed above. | docs/2025_Multi-view Deep Embedded Clustering- Exploring a new dimension of air pollution.pdf is bytewise different: size 5,509,400; SHA-256 5992c9e3e773431661fab3b2615278e3d18ba6b334867fb629b57dbbb63de3cb. | No semantic difference is claimed from the byte mismatch alone. | exact |

## Air-Pollution Application

| Paper element | Paper claim | DEKM evidence | Available artifact evidence | Repository behavior | Reproduction result | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Page 13, evaluation data | The Luxembourg dataset contains 3,765 samples with 13 features. | DEKM does not define this dataset. | No row-level data or schema file. | Air-pollution exact reproduction is deferred; no surrogate is labeled as the author's dataset. | Sample matrix and labels/assignments cannot be verified. | exact; missing; unreproducible |
| Page 11, Table 3 | The feature-description table visibly contains 17 rows. | Not applicable. | No preprocessing table or feature manifest. | No code claims that Table 3 is the exact 13-column model schema. | The 17-row description cannot be mapped one-to-one to the 13 model columns. | exact; contradictory |
| Page 14, Table 4 | The sample matrix has 13 columns: four traffic distances; three industrial-neighbor features; one residential-buffer feature; one forest-buffer feature; three neighbor-NO2 features; and one NO2 value. | Not applicable. | No matrix is supplied. | No fabricated schema is accepted as exact. | Table 4 is visually readable, but the full 3,765-row matrix is unavailable. | exact; missing; unreproducible |
| Table 3 versus Table 4 | Table 3 lists raw Industrial zone, Residential zone, and Forest rows that are not separate Table 4 columns. Table 4 includes neighbor-NO2 distance to max, which Table 3 does not list. | Not applicable. | No schema resolution artifact. | Treats the feature definition as internally inconsistent. | Exact preprocessing cannot be reconstructed. | contradictory; unreproducible |
| Page 14, Table 5 | Silhouette scores are K-means 28.29, DEKM 50.48, Proposed single-view 52.76, Proposed multi-view 61.76. | DEKM does not provide the air-pollution run. | No assignments, distances, or metric logs. | Does not claim to reproduce these values. | Values are readable but cannot be independently recomputed. | exact; unreproducible |
| Page 14, Figure 9 | Elbow annotation selects k = 4 with score 149.646. | Not applicable. | No inertia series, fitted KMeans object, or preprocessing matrix. | No exact plot regeneration claim. | The rendered point is readable; exact computation is unavailable. | exact; unreproducible |
| Page 13, Figure 8 | Caption says Land-use Map, but the figure and legend display primary roads and traffic points. | Not applicable. | No source figure data. | Records this as a paper caption mismatch, not a data source. | Exact figure can only be viewed, not regenerated. | contradictory |
| Pages 15-16, Figures 10-13 | PCA 2-D/3-D and t-SNE 2-D/3-D visualizations are presented as evidence of cluster separation. | Not applicable. | No coordinates, random seed, perplexity, initialization, PCA parameters, or cluster assignments. | No exact plot regeneration claim. | Rendered plots are inspectable, but coordinates cannot be reproduced. | missing; unreproducible |
| Page 16, Figures 14-16 | Traffic-distance, land-use, and cluster-size percentages are shown as charts. | Not applicable. | No row-level category table or aggregation code. | No exact aggregation claim. | Percentages can be estimated/read from the charts but not independently recomputed. | missing; unreproducible |
| Pages 16-17, Figures 17-20 | Four NO2 boxplots are used to validate cluster interpretations. | Not applicable. | No raw values, quartiles, whisker rule, outlier list, or assignments. | No exact boxplot regeneration claim. | Exact quartiles and distributions cannot be reconstructed. | missing; unreproducible |
| NO2 validation logic | NO2 value and three neighbor-NO2 features are inputs, then NO2 distributions are used to validate the resulting clusters. | Not applicable. | No held-out target or leakage-controlled protocol. | Limits the conclusion to descriptive clustering; does not treat the boxplots as independent predictive validation. | The figures demonstrate preservation of an input variable, not independent hotspot prediction. | inferred |
| Page 17, data availability | Data will be made available on request. | Not applicable. | The audited public folder does not contain the data. | Exact reproduction remains blocked pending author-provided data and preprocessing artifacts. | No exact air-pollution reproduction is possible now. | exact; missing; unreproducible |

## Reproducibility Boundary

### Reproducible now

- The printed equations, tables, diagrams, captions, and numerical values listed above.
- The byte-level inventory of the user's reference folder.
- The repository's explicitly versioned protocol contracts and their local tests.
- The complete REUTERS release-L2 seed-42 controlled run and its independent metric/hash audit.
- The post-training fusion diagnostics, explicitly labeled as diagnostics rather than paper truth.

### Not reproducible now

- The exact implementation that produced MvDEC Table 2.
- The undocumented trial/seed/checkpoint selection behind "best performances".
- A unique two-view feature construction for public datasets.
- The air-pollution 3,765 x 13 matrix and its preprocessing pipeline.
- Exact PCA/t-SNE coordinates, chart aggregates, boxplot statistics, cluster assignments, or fitted air-pollution weights.

## Required Evidence For An Exact Claim

An exact-paper reproduction claim requires, at minimum:

1. Author source code or an immutable implementation commit.
2. Exact public-dataset arrays or source/version/preprocessing fingerprints.
3. View-construction rules and feature indices for each dataset.
4. Full architecture definitions, especially the View2 latent/reconstruction boundary.
5. The actual fusion rule used for Table 2.
6. Seeds, initialization, optimizer settings, batch schedule, update interval, stopping rule, trial count, and model-selection rule.
7. Air-pollution raw and processed matrices, coordinates, time ranges, category thresholds, raster/grid parameters, convolution kernels, nearest-neighbor rules, and cluster assignments.
8. Run logs, checkpoints, final weights, assignments, and metric code for the published rows.

Until those items exist, future work should continue only as transparent, versioned controlled reconstructions. It should not spend additional GPU quota on blind parameter searches selected by proximity to the paper's labels.
