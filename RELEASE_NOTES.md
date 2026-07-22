# BanglaCyberBench v1.0.3

Camera-ready reproducibility release accompanying:

**BanglaCyberBench: A Dual-Script Benchmark and a Script-Aware Ensemble for
Bengali Cyberbullying Detection**

## Included

- final manuscript and bibliography;
- numbered preprocessing, split, training, robustness, ablation, comparison,
  annotation, and asset-generation notebooks;
- split manifests, label mappings, prediction arrays, probabilities, and fusion
  metadata;
- deterministic metric and bootstrap reproduction script;
- environment specifications, citation metadata, and checksums;
- source-data download and licensing instructions.

## Reference test result

The retained arrays reproduce Macro-F1 0.8225, Weighted-F1 0.8332, Accuracy
0.8339, MCC 0.7452, and Macro-AUROC 0.9626 on the 18,865-example test split.

## Important data notice

The repository licence applies only to original code and documentation. Raw
third-party dataset text and pretrained model weights remain subject to their
original terms. See `DATA_LICENSE.md` before redistribution.

## Governance artifacts

This release adds a datasheet (`DATASHEET.md`) and a model card (`MODEL_CARD.md`)
documenting the benchmark and the reference ensemble, including the known Romanized
performance gap and the cross-source transfer drop. Per-source licensing is in
`DATA_LICENSE.md`; file integrity is covered by `SHA256SUMS`.

## Third-party processing notice

The zero-shot large-language-model reference in the paper sent the 18,865 test
comments to Google's Gemini (`google/gemini-2.5-flash`) through the OpenRouter API.
This is a one-off evaluation step, not part of benchmark construction: reconstructing
BanglaCyberBench never contacts a third-party API. The transfer was assessed against
the source-dataset licences (`DATA_LICENSE.md`) and the provider's stated retention
terms. Users who redistribute or re-run this baseline are responsible for the same
check under their own agreements.