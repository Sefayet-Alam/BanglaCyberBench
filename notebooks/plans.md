# BanglaCyberBench — Research Plan (v4, 5-class advanced pipeline)

> **What changed since v3**
> 1. **BanglishBERT added** (`csebuetnlp/banglishbert`) as a 4th encoder → the framework is now a
>    **4-encoder** ensemble (BanglishBERT, BanglaBERT, MuRIL, XLM-R).
> 2. **BanglaBERT is now script-isolated**: it **never trains, validates, or tests on Romanized
>    Bangla**. It is a Bangla-script specialist; on Romanized rows it emits neutral zero logits and
>    receives zero ensemble weight. BanglishBERT is the bilingual (Bangla + Romanized) encoder.
> 3. **New pre-flight notebook** `check_on_less_data.ipynb` — a fast, label-stratified miniature of
>    the whole pipeline (fine-tune → ensemble → ablation) run **before** the multi-hour main
>    experiments, to confirm the methodology is sound and that no ablated component yields a
>    positive delta (i.e., no component is silently hurting). If any component shows a non-positive
>    contribution here, the final method is revised before committing GPU-days to NB05/07/08.

## 1. Motivation & contribution
A unified, **deduplicated, cross-source, cross-script** Bangla cyberbullying benchmark with a
**5-class fine-grained abuse taxonomy**, plus a **robustness-regularized multi-encoder framework**
that is methodologically distinct from the base paper (Hoque & Seddiqui 2025, *Frontiers in AI*,
which uses plain transformer stacking on a single Facebook source).

Three contributions:
1. **Benchmark** — 4 sources, 2 scripts (Bangla + Romanized), deduplicated, 5-class.
2. **Method** — adversarial + consistency-regularized, class-imbalance-aware, **script-aware**
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
Script mix ≈ 45% Bangla / 55% Romanized.

## 3. The 5-class taxonomy
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

Consolidation lives in **NB02** so the splits carry the final `label5` directly.

## 4. Split protocol
- **70 / 10 / 20** (train / val / test), **stratified on `label5`**, deduplicated.
- **Val (10%)**: early stopping, ensemble-weight optimization, logit-adjustment τ tuning. *Tuning only.*
- **Test (20%)**: pure held-out. **Reported metric. Never used for any tuning** → zero contamination.
- **Source-held-out** (×4): train on 3 sources (+ internal val), test on the 4th unseen source.
- **Script-held-out** (×2): train on one script, test on the other unseen script.
- Hard assertion in every held-out run: `intersection(train, heldout_test) == 0` on `uid`.

## 5. Methodology (the framework)
Single-task 5-class. For **each encoder** (BanglishBERT, BanglaBERT, MuRIL, XLM-R), fine-tune with:

| component | role | citable basis |
|---|---|---|
| Class-balanced focal loss | down-weight easy/majority | Cui 2019 / Lin 2017 |
| Balanced sampler (√-inverse-freq) | minority recall | — |
| **FGM adversarial training** | embedding-space robustness & generalization | Miyato 2017 |
| **R-Drop** | dropout-consistency regularization | Liang 2021 |
| **EMA weights** | stable, well-generalized eval weights | — |
| Multi-sample dropout (head) | variance reduction | Inoue 2019 |
| **Logit adjustment** (inference) | residual long-tail correction | Menon 2021 |
| **Script-aware specialization** | **BanglaBERT ← Bangla script only**; BanglishBERT ← both scripts | — |
| Confidence-weighted stacking ensemble | fuse the 4 encoders (script-masked) | (distinct from base paper) |

**Script-aware contract (the v4 change).**
- **BanglaBERT** trains/validates/tests **only on Bangla-script rows**. Its saved val/test logits are
  full-size `[N, C]`, with **neutral zero logits on Romanized rows**. In the ensemble it gets
  **zero row-weight on Romanized rows** (weights renormalize over the active encoders per row).
- **BanglishBERT** is the bilingual workhorse — trains/tests on **all** rows (Bangla + Romanized).
- MuRIL and XLM-R remain full-scope (both scripts).
- Rationale: BanglaBERT's pretraining is Bangla-script only; feeding it Romanized Bangla degrades it
  and contaminates the script-holdout story. Isolating it makes each encoder's competence explicit
  and gives the ensemble a clean specialist + generalists split.

Every component is a config toggle → directly produces the **ablation table** in NB08.
(Optional extension reserved for ablation: supervised contrastive auxiliary loss, default off.)

**Fixed constraints honored:** dedup ✔, stratify ✔, preprocessing unchanged (priority reorder only) ✔,
no leakage ✔, 20% test everywhere ✔, LR-decay = False ✔, binary head omitted (none-class F1 is the
binary-equivalent; reinstated only if it provably helps) ✔, **BanglaBERT never sees Romanized** ✔.

## 6. Metrics
Primary **Macro-F1** + **Weighted-F1** (reported together — defensible for imbalanced multi-class),
plus **Accuracy, MCC, Macro-AUROC**, full per-class report, and `none`-class F1 as binary-equivalent.

## 7. The pipeline
| nb | name | status | key output | runtime (4060Ti) |
|---|---|---|---|---|
| 00 | **check_on_less_data** | **new (pre-flight)** | `outputs/precheck/*` (mini fine-tune + ensemble + ablation + verdict) | ~1.5–2.5 h |
| 01 | dataset_inventory | keep | `data/merged/benchmark_raw.csv` | ~2 min |
| 02 | preprocessing_and_consolidation | keep | `data/processed/benchmark_cleaned.csv` (+`label5`) | ~3 min |
| 03 | data_splits | keep | `data/splits/*` (random 70/10/20 + 4 source + 2 script held-outs) | ~1 min |
| 04 | baselines | keep | TF-IDF→LR/LinearSVM + char-BiLSTM on `label5`, test-20% | ~30 min |
| 05 | advanced_finetuning | **updated** | `outputs/models_main/*` (**4 enc × 3 seed = 12**) | ~16–22 h |
| 06 | ensemble | **updated** | `outputs/ensemble/*` (script-masked weighted + stacking + LA) | ~20 min |
| 07 | robustness | **updated** | `outputs/robustness/*` (4 source + 2 script held-outs, 4 enc) | ~30–45 h |
| 08 | ablation | **updated** | `outputs/ablation/*` (component + 5-vs-9-class), anchor = BanglishBERT | ~6–9 h |
| 09 | basepaper_comparison | keep | `outputs/basepaper/*` (facebook 44K, base 5-class, 70/15/15) | ~2–3 h |
| 10 | analysis_and_assets | keep | all tables, figures, LaTeX | ~25 min |

**Data flow:** 00 is an independent pre-flight gate (reads `03` outputs, writes only to
`outputs/precheck/`, touches nothing else). 01→02→03 are the data spine; 05 trains; 06 ensembles;
07/08/09 are independent experiment legs that read 03 (and 02); 10 reads everyone's `outputs/`.

## 7a. Pre-flight gate — `check_on_less_data.ipynb` (NB00)
A faithful miniature of the full pipeline on a **label-stratified sub-sample** (default
~1.5K/class train, ~0.3K/class val, ~0.5K/class test). **All settings are identical to the main
notebooks** — same architecture, losses, toggles, uniform LR, script-aware BanglaBERT — only the
data volume shrinks and seeds are fixed to one (seed=42), since the goal is a directional signal,
not variance.

It runs three legs and an automatic verdict:
1. **Fine-tune** all 4 encoders (1 seed) → per-model 20%-test metrics + in-memory logits.
2. **Ensemble** (script-masked weighted, Nelder-Mead on val) → fused test metrics.
3. **Ablation** (additive, BanglishBERT): base → +focal+CW → +sampler → +MSD → +R-Drop → +FGM →
   +EMA(full); plus 5-class vs 9-class taxonomy.
4. **Verdict** → PASS / REVISE. Flags raised if:
   - the ensemble underperforms the best single encoder,
   - **any additive component contributes a non-positive macro-F1 delta** (the "plus-delta" check —
     if removing/omitting a component would score ≥ keeping it, that component is revised or dropped
     before the main run),
   - 5-class is not at least competitive with 9-class on the shared metrics.

**Decision rule:** if the verdict is REVISE, edit §5 (drop/replace the offending component or reorder
the taxonomy) and re-run NB00 until PASS, *then* launch NB05/06/07/08. This converts a potential
multi-day mistake into a ~2-hour check.

## 8. Build order for delivery
0. **check_on_less_data (NB00)** ← run first; gate the rest on its verdict
1. NB02, NB03, NB05  (data spine + main trainer)
2. NB04, NB06, NB07
3. NB08, NB09, NB10
NB01 unchanged (only its merge cell matters).