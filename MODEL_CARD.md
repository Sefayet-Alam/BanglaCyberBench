# Model Card — BanglaCyberBench Script-Aware Ensemble

Follows the *Model Cards for Model Reporting* framework (Mitchell et al., 2019).
The accompanying paper is authoritative; this card summarises and points back to it.

## Model details

- **What it is.** A script-aware ensemble of four transformer encoders for five-class
  Bengali harmful-content typing: BanglishBERT (bilingual), BanglaBERT (Bangla-script
  specialist), MuRIL, and XLM-R. Each is fine-tuned under three seeds (42, 123, 456),
  giving twelve members fused by a validation-optimised weighted-logit rule.
- **Training objective.** Cross-entropy (label smoothing 0.03) plus the Fast Gradient
  Method (FGM, ε = 1.0) — one adversarial forward/backward pass per batch. No focal
  loss, class-balanced sampler, R-Drop, EMA, or logit adjustment in the shipped model.
- **Script routing.** BanglaBERT is masked off Romanized comments (emits a neutral
  logit, zero fusion weight there); the other three decide Romanized cases. On this
  benchmark the script tag is source-derived, not detected per comment.
- **Fusion.** Nelder–Mead search over the 12-member simplex, fit on validation,
  frozen before the test read-out.
- **Reference / contact.** See the paper and repository README.

## Intended use

- **Intended.** A fully specified *reference system* for the benchmark's evaluation
  protocol and a baseline for research on Bengali abuse detection and transfer.
  It is offered as a reference, not as an algorithmic-novelty claim.
- **Not intended.** Production moderation without target-domain validation;
  Romanized-heavy streams (see limitations); any use implying it detects
  interactional cyberbullying rather than harmful comment content.

## Factors

Performance varies sharply by **script** and by **source**. Both are reported
explicitly because a single headline number hides them.

## Metrics and evaluation data

- **In-domain (20% test, n = 18,865):** Macro-F1 0.8225, Weighted-F1 0.8332,
  Accuracy 0.8339, MCC 0.7452, Macro-AUROC 0.9626. Percentile-bootstrap 95% CIs
  (B = 2,000, seed 42) accompany the headline and every per-class F1. These CIs
  capture test-resampling only — not training-seed, split, or annotation variance.
- **By script (final test arrays):** Bangla-script five-class Macro-F1 0.8303
  (n = 11,406); Romanized five-class Macro-F1 **0.3900** (n = 7,459), with a
  four-supported-class mean of 0.4875. Romanized gold has no `threat` support, which
  caps its five-class Macro-F1 at 0.80 by construction.
- **Cross-source (held-out source):** Macro-F1 collapses to 0.46–0.59
  (Facebook-44K 0.5850, Multilabel-12.5K 0.5601, BD-SHS 0.4612). This is a compound
  source shift (platform, period, sampling, annotation policy), not a single factor.
- **Comparisons are point estimates, not paired significance tests.** The ensemble
  beats its best full-scope single member (MuRIL 0.8084) by +0.0141; uniform
  averaging matches the learned fusion to within 0.0003, so the gain is attributable
  to encoder diversity rather than the weighting rule or the script mask (which is
  overall-neutral, −0.0007).

## Training data

BanglaCyberBench (see `DATASHEET.md`). Backbones are public pretrained encoders used
under their own licences.

## Ethical considerations

The corpus contains abusive, sexual, religious, and threatening language. The model
can mislabel intent-ambiguous comments (the `abusive`↔`none` boundary is the dominant
error) and underperforms badly on Romanized input. It should not be used to make
consequential moderation decisions on its own. No fairness or PII audit was run.

## Caveats and recommendations

- Validate on the target source before any deployment; in-domain accuracy overstates
  readiness.
- Do not use on Romanized-dominant traffic without improvement — the Romanized score
  is near the floor.
- Treat the `+0.0141` ensemble gain and the base-paper `Threat` delta as unpaired
  point estimates; released per-example predictions allow third parties to run paired
  tests.
