# MvDEC Paper-Code Audit

## Scope

This audit compares the implemented MvDEC reproduction against:

- Kassem et al., *Multi-view Deep Embedded Clustering: Exploring a new dimension of air pollution* (Engineering Applications of Artificial Intelligence 139, 2025, 109509).
- Guo et al., *Deep Embedded K-Means Clustering* (2021).
- The official DEKM release at `external_repos/DEKM/DEKM_dense.py`.

The immutable baseline `mvdec_2025_equation11_literal_v1` remains unchanged. Every alternative interpretation is a separately named protocol and must not be described as official MvDEC code.

## Main Conclusion

MvDEC extends DEKM, but the two papers do not define one uniquely reproducible implementation. They agree on the broad alternating K-Means/eigenvector refinement idea, while MvDEC changes the architecture and objective and contains internal contradictions. DEKM can resolve some missing schedule details, but it cannot resolve statements where MvDEC explicitly conflicts with itself.

Exact reproduction of MvDEC Table 2 cannot be claimed without the authors' source code, seeds, initialization, model-selection rule, and air-pollution data.

## Ambiguity Matrix

| Topic | MvDEC statement | Conflicting MvDEC statement | DEKM paper/release evidence | Literal implementation | Classification / confidence | Named experiment |
| --- | --- | --- | --- | --- | --- | --- |
| Eigen direction | Section 3.1 repeats that the last eigenvector contributes least; Eq. 10 also says the last dimension is modified. | Section 3.3.4 says the least-informative direction has the smallest eigenvalue. | DEKM sorts eigenvalues ascending and updates the last/largest direction. | Largest/last. | Internal MvDEC contradiction; high confidence. | B1 smallest+frozen, B3 smallest+selected, P0 paper-resolved. |
| Greedy target | Eq. 10 defines a distance to adjusted `y'`, but does not state whether non-selected dimensions are frozen anchors. | No explicit update rule for the other transformed dimensions. | The DEKM release copies the full transformed embedding and replaces only the selected last coordinate. | Frozen full-vector target. | Underspecified in MvDEC; high confidence. | B2/B3 selected-dimension-only. |
| Refinement batch | Table 2 states Proposed View1 is the same as `DEKM_F`; all View1 numbers exactly equal `DEKM_F`. | No batch size or update schedule is stated in the MvDEC method section. | DEKM defines `DEKM_F` as full-batch representation updating, while its released REUTERS code also uses a 10,000-row batch. | Sequential batch size 256 inherited from the previously validated release-style schedule. | Strong table-based inference, not explicit pseudocode; medium-high confidence. | F0 largest+frozen full batch; P0 smallest+selected full batch. |
| Reconstruction during refinement | Eq. 11 explicitly optimizes `L1 + L2 + L3 + L4`. | MvDEC's DEKM review says the decoder is discarded, but its own final objective retains reconstruction. | DEKM discards the decoder after pretraining and stops optimizing reconstruction. | Retains both decoders and L1. | Deliberate MvDEC change; high confidence. | All Eq. 11 protocols retain L1. |
| L2 versus L3 | Eq. 6 defines K-Means SSE and Eq. 9 defines `Tr(V Sw V^T)`. | No explanation is given for optimizing both. | For square orthonormal `V`, trace invariance makes L3 equal L2. | Both terms have unit weight, so SSE is counted twice. | Mathematical redundancy; high confidence. | Kept literal in B0-B3/F0/P0 to isolate other ambiguities. |
| View semantics | Prose says each view uses a specific set of attributes. | Eq. 3 and Figure 2 feed the same `x_i`/13-dimensional input to both networks; no feature partition is defined. | Public DEKM REUTERS contains one 2,000-dimensional feature matrix. | Two architectures consume the same input. | No reproducible split is specified; high confidence. | No silent feature split is introduced. |
| Fusion | Eq. 4 averages equal-dimensional latent embeddings. | Figure 2 ends with a 46-wide concatenation of two 23-wide outputs; page 6 incorrectly calls averaging Eq. 6. | DEKM has one latent representation and cannot resolve multi-view fusion. | Average the first 10 latent dimensions. | Eq. 4 is the clearest mathematical definition; medium-high confidence. | Architecture/fusion changes require separate future protocols. |
| View2 latent | Figure 2 ends View2 with one linear 23-wide head for the 13-feature example. | No 10-dimensional latent node is labeled in that head. | DEKM does not contain this U-Net branch. | Interpret first 10 outputs as latent and last 13 as reconstruction. | Implementation assumption; high confidence that the paper is incomplete. | Current direct-head protocol plus existing architecture diagnostics. |
| View2 pretraining identifiability | Reconstruction loss uses only the last 13 outputs of the joint head. | The paper does not explain how the first 10 output columns are pretrained. | Not applicable to DEKM's separate latent and reconstruction heads. | Latent-specific columns receive zero direct reconstruction gradient during pretraining. | Confirmed architecture consequence; high confidence. | Documented by a regression test; any fix must be a new architecture protocol. |
| Dataset size | Table 1 text extraction reports REUTERS sample count as `210`. | Dataset name is REUTERS-10K and the official DEKM arrays contain 10,000 rows. | Official DEKM data has 10,000 REUTERS samples. | Uses the official 10,000-row arrays. | Typesetting/extraction error; high confidence. | None. |
| Reproducibility details | Adam and 200 pretraining epochs can be inherited from DEKM context. | MvDEC gives no seed, learning-rate schedule, initialization, exact alternating schedule, variance, or model-selection rule. | DEKM release supplies some defaults but not MvDEC-specific choices. | Uses the documented inherited schedule. | Exact Table 2 reproduction is not identifiable; high confidence. | Named protocols make each assumption auditable. |

## View2 Gradient Finding

For the Figure 2 direct joint head, the output is split as:

```text
[first hidden_units columns | final input_shape columns]
[interpreted latent          | reconstruction]
```

Reconstruction pretraining slices only the reconstruction columns. Therefore the final head's latent columns have exactly zero direct gradient during pretraining, while the reconstruction columns have non-zero gradient. Shared earlier U-Net layers still train. This is a paper architecture ambiguity, not silently corrected in the literal protocol.

## Protocol Matrix

| Label | Protocol ID | Eigen direction | Greedy target | Refinement batch |
| --- | --- | --- | --- | --- |
| B0 | `mvdec_2025_equation11_literal_v1` | largest | frozen snapshot | fixed 256 |
| B1 | `mvdec_2025_equation11_smallest_frozen_v1` | smallest | frozen snapshot | fixed 256 |
| B2 | `mvdec_2025_equation11_largest_selected_only_v1` | largest | selected dimension only | fixed 256 |
| B3 | `mvdec_2025_equation11_smallest_selected_only_v1` | smallest | selected dimension only | fixed 256 |
| F0 | `mvdec_2025_equation11_largest_frozen_full_batch_v1` | largest | frozen snapshot | full dataset |
| P0 | `mvdec_2025_paper_resolved_full_batch_v1` | smallest | selected dimension only | full dataset |

B0 remains the immutable literal baseline. P0 means "paper-resolved interpretation," not literal, official, or ground truth.

## Data and Code Availability

- The official DEKM repository and public benchmark arrays are available and are already vendored/fetched for reproducible REUTERS, 20NEWS, and RCV1 experiments.
- No official MvDEC repository, supplementary source package, Zenodo/DataCite software record, or credible third-party reproduction was found in the completed repository/DOI/author searches.
- The MvDEC paper states: "Data will be made available on request."
- The air-pollution dataset therefore cannot be reconstructed exactly from a public artifact currently known to this project. The defensible options are to request it from the authors or build a separately documented reconstruction from EEA, Sentinel-5P, land-use, traffic, and neighborhood sources. A reconstruction must not be labeled as the authors' original dataset.

## Execution Rule

Run local lint/unit tests only. GPU training must run on Colab or Kaggle. Start with seed 42. Evaluate stability, cluster sizes, embedding norms/variances, all four loss components, unsupervised inertia/silhouette, and only then external ACC/NMI. Promote a protocol to seeds 42-44 only after it is justified; do not choose a protocol merely because it happens to match Table 2.
