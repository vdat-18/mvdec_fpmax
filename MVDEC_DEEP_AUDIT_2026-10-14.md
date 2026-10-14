# MvDEC 2025 Deep Paper-Code Audit (2026-10-14)

## Current decision

All code and notebook blockers C1-C6, stale-resume provenance, batch-count
validation, and invalid-dataset fallback are remediated. The exact training code
is commit `9b63446a0dd535a8ab94afc1e5457ab1594fb9ef`. The canonical candidate protocol
is `mvdec_2025_paper_informed_full_batch_v2`; historical P0 v1 artifacts remain
unchanged and explicitly historical. The five MvDEC notebooks pin and verify the
training commit. The three-dataset notebooks are fail-closed and stage 20NEWS
seed 42, then 20NEWS seeds 43-44, REUTERS, and RCV1 last.

Local validation after remediation: `155 passed, 74 skipped`; all skips are
TensorFlow-dependent. MvDEC is code-ready, but it is **not statistically frozen
as the canonical three-dataset protocol** until the pinned GPU matrix completes
without protocol substitution. A full-batch OOM is a failed gate and must not
trigger an automatic smaller-batch fallback.

DEKM is closed and remains the canonical verified control defined in
`DEKM_CANONICAL_BASELINE.md`. The repository contains no algorithm or protocol
named `DEKC`; this audit interprets that spelling as DEKM.

The existing REUTERS P0 artifacts remain valid for their recorded contract, but
`mvdec_2025_paper_resolved_full_batch_v1` is a historical paper-informed
reconstruction, not an official, literal, or uniquely resolved implementation
of MvDEC 2025. No code-level release blocker remains; the next release gate is
the pinned 20NEWS seed-42 GPU run against the canonical 18,846-row source.

## Original audit snapshot (pre-remediation)

At audit time, MvDEC was not ready to be frozen because the CLI used the
18,057-row release NPZ for 20NEWS while the frozen DEKM control and official
DEKM loader used the 18,846-row scikit-learn source. The critical findings below
describe that pre-remediation state and are retained as the audit trail.

## Scope and verification

The audit read the complete 18-page MvDEC paper, including every method
paragraph, equations 1-11, Figures 1-20, Tables 1-5, footnotes, data-availability
statement, and references. The implementation review covered:

- `src/representation_learning/MVDEC_dense.py`
- `src/pipeline/mvdec_contract.py`
- `src/pipeline/mvdec_runs.py`
- `src/pipeline/mvdec_audit.py`
- `src/pipeline/public_reproduction_matrix.py`
- the MvDEC artifact loader/validator in `src/pipeline/data.py`
- all five tracked MvDEC Kaggle/Colab notebooks
- the focused MvDEC contract, loss, architecture, preprocessing, run, audit,
  reclustering, and public-matrix tests

Validation during the original audit:

- `uv run ruff check ...`: passed.
- Focused MvDEC pytest suite: `28 passed, 74 skipped`.
- The 74 skipped tests require TensorFlow, so the local run did not execute most
  architecture, gradient, loss, and training-contract tests.

## Critical findings

### C1 - MvDEC 20NEWS silently uses the wrong canonical source

`load_dekm_public_dataset()` defaults 20NEWS to
`dekm_google_drive_npz_18057`. `MVDEC_dense.py` calls it without a
`twenty_news_source` argument. Consequently, the current MvDEC three-dataset
notebooks do not use the official DEKM source path
`fetch_20newsgroups(subset="all")` plus `TfidfVectorizer(max_features=2000)`,
which has 18,846 rows and is the frozen DEKM comparison source.

Impact: any new MvDEC 20NEWS result produced by the current CLI is not directly
comparable with the canonical DEKM 20NEWS baseline. This blocks the planned
P0 seed-42 gate.

Required correction: make the MvDEC public protocol store and enforce an
explicit `source_id`, select the 18,846-row source for canonical 20NEWS runs,
and add a regression test asserting row count, source ID, and source hash.

### C2 - Public artifact audit does not prove public-dataset provenance

The MvDEC artifact stores a top-level `source_sha256`, and the manifest config
also stores a source hash, but the loader does not compare those two values.
For public runs, `mvdec_audit` passes `assignments.csv` to the artifact loader;
public artifacts have no preprocessing dictionary, so source-file validation is
skipped. The original feature matrix is therefore not reopened or independently
fingerprinted during audit. MvDEC also omits the public dataset `source_id` that
DEKM records.

Impact: hashes can be internally valid while the artifact/manifest source
identity is inconsistent or ambiguous. The audit proves output integrity, not
full input provenance.

Required correction: store `source_id` in config, manifest, artifact, summaries,
and assignments metadata; require equality of artifact and manifest source
fields; independently rebuild or load the canonical public source during audit
and compare its fingerprint.

### C3 - Official-run notebooks are not commit-pinned and may package failures

The tracked MvDEC notebooks clone the moving `dev` branch rather than checking
out an immutable training commit. The three-dataset notebooks catch per-run
exceptions, append them to `failures.json`, and still create the final ZIP.
They also launch all three datasets and all three seeds immediately rather than
using a bounded 20NEWS seed-42 gate.

Impact: two executions of the same notebook can use different code, and a
partial matrix can look like a completed deliverable.

Required correction: pin and verify a full commit SHA, record it in every run
manifest and package manifest, fail the notebook if the required gate fails,
and run seeds 43-44 only after seed 42 passes all integrity checks.

### C4 - P0 is not uniquely "paper-resolved"

P0 infers full-dataset refinement from Table 2 because the table says Proposed
View1 equals DEKM_F. The paper does not state the MvDEC batch size, update
interval, number of optimization steps, or alternating schedule. Table 2 may
simply copy the standalone DEKM_F result as the View1 comparator; it does not
prove that joint two-view MvDEC uses full-batch gradient updates.

Impact: `mvdec_2025_paper_resolved_full_batch_v1` overstates confidence. Its
REUTERS results are valid as an explicit ablation, but the name must not imply
that the paper uniquely determines this schedule.

Required correction: retain the protocol ID for artifact compatibility, but
change its claim/status wording to "paper-informed full-batch reconstruction"
or introduce a new versioned protocol after preregistration. Never choose the
schedule by closeness to paper ACC/NMI.

### C5 - The implemented view construction is an undocumented assumption

The paper prose says each autoencoder learns a different set of attributes, but
Eq. 3, Eq. 5, and Figure 2 feed the same 13-dimensional input to both views. The
code pretrains both models from the same `(x, x)` dataset and changes only the
network architecture. The protocol contract records architecture and fusion,
but does not explicitly record the view-input mapping.

Impact: the implementation is architecture-view learning, not a demonstrated
split-feature multi-view construction. This is defensible because the paper is
internally inconsistent, but it must be versioned as an assumption.

Required correction: add a `view_input_contract` field such as
`same_full_feature_matrix_both_views_v1` to protocol, config, manifest, and
artifact validation.

### C6 - Reported View1 is not the paper's DEKM_F control

At final evaluation the code applies a new independent K-means fit to View1
after View1 has been jointly updated with View2 and the fused MvDEC objective.
The paper's Table 2 says Proposed View1 is DEKM_F and prints exactly the DEKM_F
numbers. Those quantities are not algorithmically equivalent.

Impact: repository `view1_acc/view1_nmi` must be called "jointly trained View1",
not "DEKM_F" or a parity check against the paper.

Required correction: report the frozen standalone DEKM_F control separately;
keep jointly trained View1 only as a representation diagnostic.

## High findings

### H1 - Equation 11 double-counts the same centroid distance

For a full square orthonormal matrix V,
`Tr(V S_w V^T) = Tr(S_w)`. In the implementation this means L3 is numerically
and differentially equivalent to L2 while V is held fixed during each update.
The literal unweighted Eq. 11 objective therefore applies effective weight 2 to
K-means centroid distance before adding L1 and L4.

The code correctly records this equivalence, but papers and result tables must
not describe L2 and L3 as two independent clustering mechanisms.

### H2 - The paper contradicts itself on the greedy eigen direction

Section 3.1 says eigenvalues are ascending and the last eigenvector contributes
least. Eq. 10 again modifies the last dimension. Section 3.3.4 instead says the
least-informative direction corresponds to the smallest eigenvalue. These imply
largest/last versus smallest/first respectively.

The repository correctly isolates both choices. No direction can be called the
official truth without author code or clarification.

### H3 - Public benchmark sample counts are internally inconsistent

Table 1 labels REUTERS-10K and 20NEWS but prints 210 and 2,000 samples. Table 2
copies DEKM/DEKM_F values associated with the full benchmark datasets. The
paper does not disclose subset indices, seeds, or a sampling rule.

Exact Table 2 reproduction cannot be claimed. Canonical repository experiments
must use independently fingerprinted full sources and describe Table 2 only as
a published reference.

### H4 - "Best performance" is not a reproducible evaluation rule

The paper reports best metrics, copies competitor values from prior papers, and
does not disclose the number of trials, seeds, stopping checkpoints, variance,
or model-selection rule. Selecting a seed or protocol because it approaches the
paper would leak ground-truth labels into method selection.

The repository's mean and sample standard deviation over preregistered seeds is
the correct primary reporting policy; best ACC/NMI may appear only as a clearly
labeled sensitivity statistic.

### H5 - Public-matrix resume can reuse stale runs or archives

Manifest reuse is matched only by dataset, protocol ID, and seed. The matrix
runner can reuse a single older complete manifest without requiring the current
training commit. Archive resume checks only the archive checksum, not its
contained config, source, code revision, or presence in the current summary.
Archive names also omit config hash.

Required correction: bind reuse to full config hash, source ID/hash, and commit;
validate archive contents before skipping; include config hash in archive
identity or package metadata.

### H6 - Air-pollution interpretation is circular

The 13-feature air-pollution matrix includes `satellite_no2` and three derived
neighbor-NO2 features. The paper then interprets and "validates" clusters using
NO2 distributions. This demonstrates that clustering preserved an input
variable; it is not independent validation of pollution discovery or hotspot
prediction.

Claims must be limited to descriptive clustering. A predictive or hotspot claim
requires held-out spatial/temporal NO2 targets and leakage-controlled fitting.

## Medium findings

### M1 - Batch-count validation disagrees with training for some sizes

Balanced training uses nearest-integer batch count. The public artifact validator
uses floor division for the same policy. For official 20NEWS, 18,846 / 256 gives
74 training batches but 73 validator batches, so a public DEKM-consistent MvDEC
artifact can be produced and then rejected by its own loader.

### M2 - Invalid dataset names fall back to AIRPOLLUTION

The CLI silently replaces an unsupported dataset name with AIRPOLLUTION. A typo
can therefore start the wrong experiment instead of failing at the trust
boundary. The positional argument should use explicit choices and reject invalid
values.

### M3 - Core TensorFlow tests were not executed locally

The focused test command passed only non-TensorFlow tests; 74 architecture,
loss, gradient, and training tests skipped because TensorFlow is absent in the
local environment. Kaggle GPU execution must run the complete focused suite and
record the result before training.

### M4 - Full-batch P0 has an explicit memory-risk contract

P0 resolves refinement batch size to all rows. This is manageable only if the
runtime can hold both large dense autoencoders, activations, gradients, two view
outputs, and objective tensors for the complete dataset. An OOM must be reported
as an OOM; silently changing to mini-batches would create a different protocol.

## Paper-to-code mapping

| Paper element | Repository implementation | Audit judgment |
|---|---|---|
| Eq. 3 two encoders | Two dense models over the same input matrix | Paper-consistent with equations/figure; conflicts with prose about attribute sets |
| Eq. 4 average fusion | Equal average of 10-D latent slices | Correct for the declared direct-head interpretation |
| Eq. 5 reconstruction | Sum of both view squared reconstruction errors, mean over batch | Equivalent up to the paper's unspecified global reduction |
| Eq. 6 K-means | Assigned-centroid squared distance in fused latent | Correct |
| Eq. 7-9 transform/scatter | Full orthonormal eigensystem of within-cluster scatter | Correct, but L3 is trace-invariant and equals L2 |
| Eq. 10 greedy target | Frozen-full-vector or selected-coordinate named variants | Properly isolates an underspecified choice |
| Eq. 11 joint loss | Literal protocols use unit L1+L2+L3+L4 | Literal algebra, with explicit L2/L3 double counting |
| Figure 2 View1 | 13-500-500-2000-10 and mirrored decoder | Matches the illustrated air-pollution architecture |
| Figure 2 View2 | Dense U-Net skips and a direct 23-wide head split as 10 latent + 13 reconstruction for Eq. 11 | Plausible implementation assumption; figure does not label the split |
| Table 2 View1 | Jointly trained View1 evaluation | Not equivalent to standalone DEKM_F |
| Public source | DEKM release loader, currently old 20NEWS NPZ by default | REUTERS/RCV1 bounded; 20NEWS canonical source is wrong |

## Freeze policy

### Frozen now

- DEKM canonical baseline and its three-dataset seed 42-44 aggregates.
- MvDEC REUTERS ambiguity-matrix artifacts as historical, immutable experiments
  under their recorded protocol/config/source hashes.
- `mvdec_2025_equation11_literal_v1` as the immutable literal-control identity,
  not as an official implementation claim.

### Not frozen

- A canonical MvDEC primary protocol across REUTERS, 20NEWS, and RCV1.
- The P0 label "paper-resolved" as a methodological conclusion.
- Any MvDEC 20NEWS result produced through the current default source path.
- Any package produced from an unpinned notebook or with non-empty failures.

## Required closure sequence

1. Fix and test the canonical 20NEWS source selector and source provenance.
2. Bind artifacts, manifests, summaries, notebooks, and archives to a commit SHA.
3. Encode the same-input view construction and all schedule assumptions in the
   protocol contract.
4. Fix batch-count validation and reject invalid dataset names.
5. Make the 20NEWS notebook a seed-42-only fail-closed gate.
6. Run the full TensorFlow test suite in the pinned GPU environment.
7. Run and independently audit 20NEWS seed 42, including View1, View2, fused
   ACC/NMI, assignments, source ID/hash, manifests, output hashes, and logs.
8. Only after a valid seed-42 gate, authorize seeds 43-44. Run RCV1 last and
   preserve/report OOM rather than changing the protocol.
9. Freeze the final MvDEC status only after all three datasets have one common,
   preregistered, provenance-complete contract.
