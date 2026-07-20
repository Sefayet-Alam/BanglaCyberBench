# BanglaCyberBench

**A dual-script benchmark and a script-aware ensemble for Bengali cyberbullying detection**

This repository contains the code, notebooks, manifests, paper assets, and
evaluation artefacts for the BanglaCyberBench study. The benchmark consolidates
four public Bengali sources into 94,323 deduplicated comments covering Bangla
and Romanized Bangla. The proposed system is a four-backbone ensemble trained
with cross-entropy plus FGM and evaluated with random-split, script-aware, and
source-held-out protocols.

The accompanying paper is the authoritative description of the taxonomy,
deduplication rule, splits, training configuration, ablations, and limitations.
The published comparator is the eight-backbone transformer-stacking study; it
does not use FGM. The FGM results in this repository refer to the proposed
system only.

## Reproducibility levels

### Level 1: reproduce the reported metrics (recommended first check)

This path does not require a GPU, model checkpoints, or raw comment text. It
uses the retained final prediction/probability arrays and recomputes the
headline metrics, confidence intervals, and script-specific scores.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements-repro.txt

python scripts/reproduce_metrics.py \\
  --pred outputs/test_pred.npy \\
  --proba outputs/test_proba.npy \\
  --meta outputs/fusion_meta.npz \\
  --bootstrap 2000 \\
  --seed 42 \\
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

The overall test split contains 18,865 examples. The script also reports the
five-class Romanized Macro-F1 using the paper's explicit convention: a class
with zero gold support contributes F1 = 0. It additionally prints the
supported-class-only value so the two quantities cannot be confused.

### Level 2: verify the release snapshot

After cloning the tagged release, verify the files and record their hashes:

```bash
git checkout v1.0.1
python scripts/make_checksums.py --root . --output SHA256SUMS
git status --short
```

The release used for the paper should contain a committed `SHA256SUMS` file.
Do not regenerate it after publication unless you create a new version tag.

### Level 3: rebuild the benchmark

The four source datasets remain third-party materials. Download them from the
authoritative links listed in `DATA_LICENSE.md`, place them in the paths
documented by the preprocessing notebook, and execute the notebooks in the
numbered order shown below. Do not commit raw text unless the original source
licence explicitly permits redistribution.

The rebuild must reproduce these invariants:

```text
raw rows before consolidation:       135,575
unique cleaned comments:              94,323
train / validation / test:       66,026 / 9,432 / 18,865
UID intersection across splits:             0
Bangla-script comments:                56,989
Romanized comments:                    37,334
```

The deduplication audit should also report 1,287 cleaned-text groups spanning
multiple sources, 661 groups with conflicting five-class labels, and 271 groups
with conflicting binary flags. The retained source is the first source in the
deterministic merge order; it is a provenance field, not a claim of original
authorship.

### Level 4: retrain the models

Retraining requires a CUDA-capable GPU, the original source data, the four
pretrained model identifiers, and the exact training environment. Use
`requirements-full.txt` as a functional environment specification. Before a
camera-ready release, export the actual RunPod environment and commit it as
`requirements-mac-lock.txt`:

```bash
python -m pip freeze > requirements-mac-lock.txt
```

Do not claim bit-for-bit training reproducibility unless that lock file and the
pretrained model revisions are archived. Notebook execution order is:

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

The less-data exploratory notebook is not part of the final evidence chain and
should be labelled as exploratory if retained in the repository.

## Repository map

```text
paper/ or root                 LaTeX manuscript and bibliography
project_sources/               numbered notebooks and analysis code
data/                          source data or download instructions
outputs/                       predictions, probabilities, fusion metadata,
                              tables, figures, and model-selection records
scripts/reproduce_metrics.py  deterministic metric and bootstrap check
scripts/make_checksums.py     SHA-256 manifest generator
DATA_LICENSE.md               third-party data/model restrictions
CITATION.cff                  citation metadata
```

If the current repository keeps files in different directories, preserve the
same logical names in the command examples or update the paths in this README.

## Exact evaluation conventions

- Label order is `[abusive, none, religious, sexual, threat]`.
- Overall Macro-F1 is the unweighted mean over all five classes.
- Script-specific Macro-F1 also uses all five classes and `zero_division=0`.
- Bootstrap intervals use 2,000 resamples, NumPy `default_rng(42)`, and the
  2.5th/97.5th percentiles.
- Test predictions are never used to fit ensemble weights or early stopping.
- Source-held-out results are compound source-shift measurements. The BanTH
  hold-out is not interpreted as a pure script-transfer experiment because it
  couples source and script.

## Data and model licensing

Original repository code is MIT-licensed. Dataset text, pretrained models, and
third-party material are not covered by that licence. Read `DATA_LICENSE.md`
before downloading, redistributing, or publishing any raw comment text.

## Release procedure

1. Merge the final manuscript and reproducibility files into the release branch.
2. Run Level 1 from a clean environment and save the JSON output.
3. Run the notebook/output consistency checks and inspect the compiled PDF.
4. Commit `requirements-mac-lock.txt`, `SHA256SUMS`, and the final paper.
5. Create an annotated tag and GitHub release:

```bash
git add README.md LICENSE DATA_LICENSE.md CITATION.cff \\
  requirements-repro.txt requirements-full.txt environment.yml \\
  scripts/ requirements-mac-lock.txt SHA256SUMS
git commit -m "Prepare BanglaCyberBench v1.0.1 reproducibility release"
git tag -a v1.0.1 -m "Camera-ready reproducibility release"
git push origin main
git push origin v1.0.1
gh release create v1.0.1 \\
  --title "BanglaCyberBench v1.0.1" \\
  --notes-file RELEASE_NOTES.md
```

Use the release URL, not the mutable `main` branch, in the paper. If GitHub
immutable releases are enabled for the repository, publish the release only
after all assets and notes are correct.

## Citation

Use the `CITATION.cff` file or cite the paper associated with the tagged
release. After enabling the repository in Zenodo, add the DOI to this section
and to the paper's Data and Code Availability statement.

## Responsible use

The corpus contains offensive language. It is intended for research on abuse
detection and moderation, not for harassment, profiling, surveillance, or
generation of abusive content. Users are responsible for complying with the
source licences, applicable law, and institutional policies.
!!