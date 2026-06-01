# BanglaCyberBench — Research Plan (v3, 5-class advanced pipeline)

## 1. Motivation & contribution
A unified, **deduplicated, cross-source, cross-script** Bangla cyberbullying benchmark with a
**5-class fine-grained abuse taxonomy**, plus a **robustness-regularized multi-encoder framework**
that is methodologically distinct from the base paper (Hoque & Seddiqui 2025, *Frontiers in AI*,
which uses plain transformer stacking on a single Facebook source).

Three contributions:
1. **Benchmark** — 4 sources, 2 scripts (Bangla + Romanized), deduplicated, 5-class.
2. **Method** — adversarial + consistency-regularized, class-imbalance-aware, script-aware
   multi-encoder ensemble with inference-time logit adjustment.
3. **Evaluation** — clean 20% held-out test, source/script held-out robustness, head-to-head
   replication against the base paper's own protocol.

## 2. Dataset
| source | rows (post-dedup) | script | notes |
|---|---|---|---|
| banth | ~37.3K | Romanized | underscore-compound labels |
| facebook_44001 | ~43.0K | Bangla | comma-compound labels (base-paper source) |
| multilabel_12557 | ~9.0K | Bangla | comma-compound |
| bd_shs | ~5.0K | Bangla | simple |

Raw merged = 135,575 → **deduplicated on `text_clean` = 94,337**.

## 3. The 5-class taxonomy (NEW)
Collapses 89 raw multi-label variants into 5 well-populated, semantically coherent classes.

| class | merged from | ~train count |
|---|---|---|
| **none** | none, not-bully | ~39K |
| **abusive** | abusive/violence, personal offense, political, troll, slander, spam, origin, body-shaming, misc | ~19K |
| **sexual** | sexual, gender (gender-based / misogynistic harassment) | ~8.7K |
| **religious** | religious, religion | ~6.5K |
| **threat** | threat, callToViolence | ~3.2K |

**Consolidation rule.** Each raw label is split on `,` and `_`, each component mapped to a bucket,
and the final class chosen by priority **threat > sexual > religious > abusive > none**. Consistency
with `label_binary` is enforced (binary=0 → none; binary=1 & resolved-none → abusive).
*(Priority order is tunable; this order keeps explicit-violence and identity-targeted abuse from being
absorbed into the generic `abusive` bucket.)*

Consolidation now lives in **NB02** so the splits carry the final `label5` directly.

## 4. Split protocol
- **70 / 10 / 20** (train / val / test), **stratified on `label5`**, deduplicated.
- **Val (10%)**: early stopping, ensemble-weight optimization, logit-adjustment τ tuning. *Tuning only.*
- **Test (20%)**: pure held-out. **Reported metric. Never used for any tuning** → zero contamination.
- **Source-held-out** (×4): train on 3 sources (+ internal val), test on the 4th unseen source.
- **Script-held-out** (×2): train on one script, test on the other unseen script.
- Hard assertion in every held-out run: `intersection(train, heldout_test) == 0`.

## 5. Methodology (the framework)
Single-task 5-class. For each encoder (BanglaBERT, MuRIL, XLM-R), fine-tune with:

| component | role | citable basis |
|---|---|---|
| Class-balanced focal loss | down-weight easy/majority | Cui 2019 / Lin 2017 |
| Balanced sampler (√-inverse-freq) | minority recall | — |
| **FGM adversarial training** | embedding-space robustness & generalization | Miyato 2017 |
| **R-Drop** | dropout-consistency regularization | Liang 2021 |
| **EMA weights** | stable, well-generalized eval weights | — |
| Multi-sample dropout (head) | variance reduction | Inoue 2019 |
| **Logit adjustment** (inference) | residual long-tail correction | Menon 2021 |
| Script-aware specialization | BanglaBERT ← Bangla script only | — |
| Confidence-weighted stacking ensemble | fuse the 3 encoders | (distinct from base paper) |

Every component is a config toggle → directly produces the **ablation table** in NB08.
(Optional extension reserved for ablation: supervised contrastive auxiliary loss, default off.)

**Fixed constraints honored:** dedup ✔, stratify ✔, preprocessing unchanged (priority reorder only) ✔,
no leakage ✔, 20% test everywhere ✔, LR-decay = False ✔, binary head omitted (none-class F1 is the
binary-equivalent; reinstated only if it provably helps) ✔.

## 6. Metrics
Primary **Macro-F1** + **Weighted-F1** (reported together — defensible for imbalanced multi-class),
plus **Accuracy, MCC, Macro-AUROC**, full per-class report, and `none`-class F1 as binary-equivalent.

## 7. The 10-notebook pipeline
| nb | name | status | key output | runtime (4060Ti) |
|---|---|---|---|---|
| 01 | dataset_inventory | keep | `data/merged/benchmark_raw.csv` | ~2 min |
| 02 | preprocessing_and_consolidation | **new** | `data/processed/benchmark_cleaned.csv` (+`label5`) | ~3 min |
| 03 | data_splits | **new** | `data/splits/*` (random 70/10/20 + 4 source + 2 script held-outs) | ~1 min |
| 04 | baselines | update | TF-IDF→LR/LinearSVM + char-BiLSTM on `label5`, test-20% | ~30 min |
| 05 | advanced_finetuning | **new** | `outputs/models_main/*` (3 enc × 3 seed) | ~14–18 h |
| 06 | ensemble | update | `outputs/ensemble/*` (weighted + stacking + LA) | ~20 min |
| 07 | robustness | update | `outputs/robustness/*` (4 source + 2 script held-outs) | ~25–35 h |
| 08 | ablation | update | `outputs/ablation/*` (component + 5-vs-9-class + HP) | ~6–8 h |
| 09 | basepaper_comparison | **new** | `outputs/basepaper/*` (facebook 44K, base 5-class, 70/15/15) | ~2–3 h |
| 10 | analysis_and_assets | update | all tables, figures, LaTeX | ~25 min |

**Data flow:** 01→02→03 are the data spine; 05 trains; 06 ensembles; 07/08/09 are independent
experiment legs that read 03 (and 02); 10 reads everyone's `outputs/` JSON/CSV.

## 8. Build order for delivery
1. plans.md, NB02, NB03, NB05  ← this batch
2. NB04, NB06, NB07            ← next
3. NB08, NB09, NB10            ← final
NB01 unchanged (only its merge cell matters).
