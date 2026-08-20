# BanglaCyberBench

**A dual-script benchmark and a script-aware ensemble for Bengali cyberbullying detection**

This repository contains the code, notebooks, manifests, paper assets, and
evaluation artifacts for the BanglaCyberBench study. The benchmark contains
94,323 deduplicated comments in Bangla and Romanized Bangla.
The proposed system is a four-backbone ensemble trained with cross-entropy and
the Fast Gradient Method (FGM) and evaluated with random-split,
script-stratified, and source-held-out protocols.

The manuscript describes the taxonomy, deduplication rule, splits, training
configuration, ablations, and limitations. The published comparator is the
eight-backbone transformer-stacking study and does not use FGM. FGM results in
this repository refer only to the proposed system.

## Reproducibility levels

### Level 1: reproduce the reported metrics

This path does not require a GPU, model checkpoints, or raw comment text. It
uses the retained prediction and probability arrays to recompute the headline
metrics, confidence intervals, and script-specific scores.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -r requirements-repro.txt

python scripts/reproduce_metrics.py \
  --pred outputs/test_pred.npy \
  --proba outputs/test_proba.npy \
  --meta outputs/fusion_meta.npz \
  --bootstrap 2000 \
  --seed 42 \
  --check
```

The script should report values within `1e-4` of:

| Metric | Expected value |
|---|---:|
| Macro-F1 | 0.8225 |
| Weighted-F1 | 0.8332 |
| Accuracy | 0.8339 |
| MCC | 0.7452 |
| Macro-AUROC | 0.9626 |

The retained test split contains 18,865 examples. The script reports
five-class Romanized Macro-F1 using the paper's explicit convention: a class
with zero gold support contributes F1 = 0. It also reports the
supported-class-only value. Romanized Threat support is zero in this split.

### Level 2: verify a release snapshot

Use the checksum file committed with a tagged release. Do not regenerate the
manifest before checking it, because that would replace the values being
verified. On macOS, run:

```bash
shasum -a 256 -c SHA256SUMS
```

The next release must contain a newly generated `SHA256SUMS` that covers only
the intended public files. Development files, environment files, caches,
third-party papers, and local manuscript archives must not be included.

### Level 3: rebuild the benchmark

The four source datasets are third-party materials. Download them from the
authoritative references in `DATA_LICENSE.md`, confirm their licences, and
place them in the paths documented by the preprocessing notebook. Raw or
derived comment text must not be redistributed unless the applicable source
terms explicitly permit it.

The repository snapshot uses these reproducibility invariants:

```text
raw rows before consolidation:       135,575
unique cleaned comments:              94,323
train / validation / test:       66,026 / 9,432 / 18,865
UID intersection across splits:             0
Bangla-script comments:                56,989
Romanized comments:                    37,334
```

Both split manifests are intentional:

- `data/splits/split_manifest.json` describes the random in-domain split.
- `data/splits/source_holdout_bangla_only/split_manifest.json` describes the
  Bangla-script source-held-out experiments.

The manifests describe different evaluation protocols and should both be kept.

### Level 4: retrain the models

Retraining requires a CUDA-capable GPU, the original source data, the four
pretrained model revisions, and the recorded training environment. Use
`requirements-full.txt` as the functional environment specification and
`requirements-mac-lock.txt` as the retained environment snapshot. Do not claim
bit-for-bit training reproducibility unless the environment and model revisions
used for the reported run are archived.

The original training sequence is:

```text
01_dataset_inventory.ipynb
02_preprocessing_and_consolidation.ipynb
03_data_splits.ipynb
04_baselines.ipynb
05_advanced_finetuning.ipynb
06_ensemble.ipynb
07_robustness.ipynb
08_ablation_upd.ipynb
09_basepaper_comparison.ipynb
10_analysis_and_assets.ipynb
12_paper_asset_creation.ipynb
13_annotation_sample.ipynb
14a_kappa_and_llm_baseline.ipynb
```

After the retained model outputs exist, run the CPU review notebook:

```text
15_cpu_review_experiments_and_fixed_figures.ipynb
```

Notebook 15 computes bootstrap confidence intervals, annotation-agreement
uncertainty, script-stratified metrics, class supports, and source-held-out
split summaries from retained project outputs. It also regenerates
code-generated paper figures. It does not retrain transformer models, compare
the project data with external source counts, or modify diagrams.

The less-data exploratory notebook is not part of the final evidence chain and
should be labelled as exploratory if retained.

## Repository map

```text
notebooks/                     numbered experiments and CPU review audit
src/                           reusable preprocessing and evaluation code
data/                          source references, metadata, and split manifests
outputs/                       retained predictions and fusion metadata
finalized_paper_q1/figures/    code-generated manuscript figures
finalized_paper_q1/tables/     manuscript tables and CPU audit outputs
scripts/reproduce_metrics.py   deterministic metric and bootstrap check
scripts/make_checksums.py      SHA-256 manifest generator
DATA_LICENSE.md                third-party data and model restrictions
DATASHEET.md                   benchmark documentation and limitations
MODEL_CARD.md                  reference-model documentation
CITATION.cff                   citation metadata
```

## Exact evaluation conventions

- Label order is `[abusive, none, religious, sexual, threat]`.
- Overall Macro-F1 is the unweighted mean over all five classes.
- Script-specific Macro-F1 also uses all five classes and `zero_division=0`.
- Supported-class Macro-F1 excludes classes with zero gold support in the
  evaluated subset and is always labelled separately.
- Bootstrap intervals use 2,000 resamples, NumPy `default_rng(42)`, and the
  2.5th and 97.5th percentiles.
- Test predictions are not used to fit ensemble weights or early stopping.
- Source-held-out results are compound source-shift measurements. The BanTH
  holdout is not interpreted as a pure script-transfer experiment because it
  couples source and script.

## Data and model licensing

Original repository code is MIT-licensed. Dataset text, pretrained models, and
third-party material are not covered by that licence. Read `DATA_LICENSE.md`
before downloading, redistributing, or publishing comment text. The presence
of a file in this repository does not grant additional rights to the underlying
dataset. A public release should contain raw or derived text only when explicit
redistribution permission has been verified for that source.

## Release integrity

Before creating the next release:

1. Reproduce the reported metrics from a clean CPU environment.
2. Verify that the public tree excludes local archives, third-party papers,
   environment files, editor settings, caches, and other development files.
3. Confirm source-by-source redistribution permissions.
4. Generate `SHA256SUMS` from the final public tree and verify it independently.
5. Check the release archive anonymously before citing it in the manuscript.

Use a tagged release rather than the mutable `main` branch in the paper.

## Citation

Use `CITATION.cff` or cite the paper associated with the tagged release. If the
repository is archived through Zenodo, add the DOI here and to the paper's Data
and Code Availability statement.

## Responsible use

The corpus contains offensive language. It is intended for research on abuse
detection and moderation, not for harassment, profiling, surveillance, or
generation of abusive content. Users are responsible for complying with the
source licences, applicable law, and institutional policies.

## Governance artifacts

The datasheet (`DATASHEET.md`) and model card (`MODEL_CARD.md`) document the
benchmark, reference ensemble, Romanized performance gap, and cross-source
transfer drop. Per-source licensing information is maintained in
`DATA_LICENSE.md`.

## Third-party processing notice

The historical zero-shot language-model evaluation sent the retained test
comments to a third-party API. This step is not part of benchmark construction,
and the local metric-reproduction workflow does not contact an external
service. Before repeating that evaluation, researchers must verify the source
dataset licences and the provider's current privacy, retention, and processing
terms.
