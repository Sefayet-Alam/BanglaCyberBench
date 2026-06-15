# BanglaCyberBench — Experiment Log (v4, CURRENT pipeline)

**Project:** BanglaCyberBench — A Multi-Source, Dual-Script Benchmark and Two Script-Aware Transformer Ensembles for Robust Fine-Grained Bengali Cyberbullying Detection
**Authors:** Sefayet Alam (sefayetalam14@gmail.com),  Naim Parvez and A. F. M. Minhazur Rahman
**Date:** June 2026
**Repo:** github.com/Sefayet-Alam/Sarcasm_detection  (outputs under `04_outputs/`)
**Target venue:** Q1/Q2 — IP&M / ESWA / Neurocomputing, or ACL/EMNLP Findings
**Status:** all experiments complete; final tables being assembled in NB10

> **Read-me note on numbers.** Every result reported for **Proposed Model B** (the full-stack
> BanglishBERT, notebook 07b) is **final** and taken directly from the completed run. Results marked
> **⟨pending — `path`⟩** are produced by notebooks that ran on cloud GPUs and live in the committed
> output files at the path shown; drop them in before circulating. No number from the previous
> (multi-task, 9-class, duplicate-retaining) draft has been carried over — that was a different
> experiment and reusing its numbers here would be incorrect.

---

## 0. One-Page Summary

Bengali cyberbullying detection has been studied almost entirely on **single sources**, in **a single
script (Bangla)**, with **coarse labels**, and **without robustness testing**. This work delivers
three things:

1. **BanglaCyberBench** — a **deduplicated, four-source, dual-script** benchmark of **94,337** unique
   comments (from 135,575 raw), spanning **Bangla script and Romanized Bangla**, consolidated into a
   clean **5-class** abuse taxonomy (`none, abusive, sexual, religious, threat`).

2. **Two distinct proposed systems**, deliberately contrasted:
   - **Proposed Model A — Script-Aware Ensemble (main).** Four encoders (BanglishBERT, a
     **Bangla-script-specialist** BanglaBERT, MuRIL, XLM-R), each fine-tuned with a **minimal**
     recipe (cross-entropy + **FGM** adversarial training only), fused by a **script-masked,
     validation-optimised weighted-logit ensemble**.
   - **Proposed Model B — Full-Stack BanglishBERT (alternate).** A single bilingual encoder trained
     with the **complete regularisation stack** (class-balanced focal loss, balanced sampler,
     multi-sample dropout, R-Drop, FGM, EMA), **seed-ensembled** across three seeds.

3. **A cross-distribution robustness study** the base paper does not attempt: **in-domain**,
   **4× source-held-out**, and **2× script-held-out** evaluation.

**Headline (Model B, 20% in-domain test):** Macro-F1 **0.8135**, Weighted-F1 0.8224, Accuracy 0.8222,
MCC 0.7291, Macro-AUROC 0.9534.
**Against the base paper (Hoque & Seddiqui 2025) on its own Facebook-44K, 5-class protocol:** Model B
reaches Macro-F1 **0.8670** vs the base paper's 0.8923 (Δ −0.0253) **while improving the hardest class,
`Threat`, to 0.8337 vs 0.7579 (+0.0758).**
**Robustness:** in-domain is strong, source transfer is moderate, and **cross-script transfer is the
open wall** (Macro-F1 0.16–0.22) — the empirical justification for Model A's script-aware design.

---

## 1. Title, in Plain Words

**BanglaCyberBench: A Multi-Source, Dual-Script Benchmark and Two Script-Aware Transformer Ensembles
for Robust Fine-Grained Bengali Cyberbullying Detection.**

- **BanglaCyberBench** — the benchmark we release.
- **Multi-Source** — four public datasets merged into one.
- **Dual-Script** — both Bangla script (বাংলা) and Romanized Bangla (*tumi kemon acho*).
- **Two … Ensembles** — we propose and compare two different systems, not one.
- **Script-Aware** — the design explicitly accounts for which script a comment is written in.
- **Robust / Fine-Grained** — evaluated beyond a random split, on a 5-class taxonomy.

The paper is simultaneously a **resource paper** (the benchmark) and a **methods paper** (two systems
+ a robustness study + a base-paper comparison).

---

## 2. Research Questions & Contributions

**RQ1.** Can heterogeneous Bengali cyberbullying datasets be unified into one clean, deduplicated,
dual-script, fine-grained benchmark?
**RQ2.** Does a *minimal* adversarially-trained, **script-aware ensemble** (Model A) match or beat a
*heavily regularised single-encoder* system (Model B)?
**RQ3.** How well do these systems generalise **across sources** and **across scripts**, and how do
they compare to the current SOTA transformer-stacking base paper?

**Contributions.**
1. **BanglaCyberBench**: 4 sources, 2 scripts, **deduplicated** (94,337), 5-class taxonomy with a
   documented priority-based consolidation of 89 raw label strings.
2. **Two contrasting proposed systems** (minimal script-aware ensemble vs. full-stack single encoder),
   evaluated head-to-head and against the base paper on its own protocol.
3. **Script-aware specialisation**: a Bangla-only BanglaBERT specialist that is *masked off* Romanized
   rows in the ensemble — a design directly motivated by our robustness findings.
4. **A systematic robustness study** (source- and script-held-out) absent from prior Bengali work.
5. **A component ablation** that drives Model A's minimalism, plus a **5-vs-9-class taxonomy ablation**.

---

## 3. Positioning vs. the Base Paper and Prior Work

**Base paper — Hoque & Seddiqui (2025), *Frontiers in AI*.** *Transformer-stacking*: XLM-R + mBERT +
Bangla-Bert-Base feeding an MLP meta-classifier, on the 44,001-comment Facebook dataset, 5 classes
(Not Bully, Sexual, Troll, Religious, Threat). Reported **multiclass F1 89.23 / Acc 89.23**, **binary
F1 93.61 / Acc 93.62**. Single source, single script, no cross-source/script robustness test.

| Axis | Base paper (H&S 2025) | This work |
|---|---|---|
| Sources | 1 (Facebook 44K) | **4 merged** |
| Scripts | Bangla only | **Bangla + Romanized** |
| Deduplication | n/a | **Yes (135,575 → 94,337)** |
| Task | 5-class | **5-class (+ 9-class taxonomy ablation)** |
| Systems | 1 (stacking) | **2 (script-aware ensemble + full-stack single encoder)** |
| Robustness | none | **in-domain + 4 source-holdout + 2 script-holdout** |
| Specialisation | none | **script-aware (Bangla specialist masked on Romanized)** |

**Honest stance.** On the base paper's *own* clean, single-source Facebook split we are **competitive,
slightly below** on overall Macro-F1 (Model B 0.8670 vs 0.8923) but **better on the minority `Threat`
class**. Our contribution is **not** "beat SOTA on one dataset"; it is the **benchmark + dual-script
coverage + robustness evaluation + the two-system comparison**, which are new evaluation axes the base
paper does not address.

---

## 4. Dataset: BanglaCyberBench

### 4.1 Sources (post-deduplication)

| Source | Script | Origin | ~Samples (dedup) | Raw label style |
|---|---|---|---:|---|
| `banth` | Romanized | Kaggle | ~37,334 | binary column + 0/1 type columns |
| `facebook_44001` | Bangla | Mendeley | ~43,000 | single label column |
| `multilabel_12557` | Bangla | Kaggle | ~9,000 | separate 0/1 type columns |
| `bd_shs` | Bangla | Mendeley | ~5,000 | one harmful column + one type column |
| **Total (unique)** | — | — | **94,337** | — |

### 4.2 Multi-source & multi-script
- **Romanized Bangla:** 37,334 comments (**39.6%**) — all from `banth`.
- **Bangla script:** 56,989 comments (**60.4%**) — from the other three sources.

`banth` is the **only** Romanized source, which makes `source_holdout_banth` and
`script_holdout_romanized` the *same* experiment — an important caveat used throughout the analysis.

**Figure — Class distribution (5-class benchmark)**
![Class distribution](https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/finalized_outputs/figures/01_class_distribution.png)

**Plain summary:** The benchmark is dominated by `none`, with `threat` the rarest class. This is the
core difficulty of the task: Macro-F1 (which weights every class equally) is governed by the rare
classes, so a model that simply predicts the majority class scores poorly even at high accuracy.

### 4.3 The 5-class taxonomy

`none · abusive · sexual · religious · threat`. The 89 raw `label_type` strings are split on `,` and
`_`, each piece mapped to a bucket, and the final class chosen by **priority**:

> **threat > sexual > religious > abusive > none**

| Final class | Raw labels folded in | ~Train count |
|---|---|---:|
| `none` | none, not-bully | ~39,000 |
| `abusive` | abusive/violence, troll, personal offense, body-shaming, origin, slander, spam, political, misc | ~19,000 |
| `sexual` | sexual, gender | ~8,700 |
| `religious` | religious, religion | ~6,500 |
| `threat` | threat, callToViolence | ~3,200 |

```mermaid
flowchart LR
  R["89 raw label_type strings<br/>(comma/underscore compounds)"] --> S["split + map each piece"]
  S --> P{"priority resolve<br/>threat &gt; sexual &gt; religious &gt; abusive &gt; none"}
  P --> N[none]
  P --> A[abusive]
  P --> X[sexual]
  P --> RE[religious]
  P --> T[threat]
```

A **9-class** variant (adds `personal, political, gender, other`) is built only for the **taxonomy
ablation** (Section 13), not for the headline.

### 4.4 Deduplication (changed from the previous draft)
The earlier draft **kept** duplicates; this version **removes** them. We deduplicate on the cleaned
text field, collapsing **135,575 → 94,337** unique comments. `banth` shrank the most (it carried many
near-identical scrapes), which is why its post-dedup share fell sharply. Deduplication makes the
held-out evaluations honest (no train/test overlap via duplicates) and is enforced by a hard `uid`
intersection assert in every split.

---

## 5. Schema Unification

The four sources used incompatible schemas, standardised to one:
`text · text_clean · label_binary · label_type · label5 · source · script · uid`.

| Source | Text col | Binary signal | Type signal | Mapping |
|---|---|---|---|---|
| `banth` | `Text` | `Label` | 0/1 type cols | active type cols joined; none → `none` |
| `bd_shs` | `sentence` | `hate speech` | `type` | `label_type = type` |
| `facebook_44001` | `comment` | derived from `label` | `label` | not bully → 0; else → 1 |
| `multilabel_12557` | `comment` | `bully` | 0/1 type cols | active type cols joined; none → `none` |

---

## 6. Splits & Evaluation Design

```mermaid
flowchart TD
  D["Deduplicated benchmark<br/>94,337"] --> R1["Random 70/10/20<br/>stratified on label5<br/>train 66,026 · val 9,432 · test 18,865"]
  D --> R2["Source-held-out ×4<br/>train on 3 sources → test the 4th"]
  D --> R3["Script-held-out ×2<br/>train on one script → test the other"]
  R1 --> Q1["Q: unseen examples,<br/>same distribution?"]
  R2 --> Q2["Q: unseen source?"]
  R3 --> Q3["Q: unseen script?"]
```

- **Random 70/10/20**, stratified on `label5`. Validation is for early stopping, ensemble-weight
  optimisation, and threshold/τ tuning **only**; the 20% test is never touched during tuning.
- **Source-held-out (×4):** `banth`, `bd_shs`, `facebook_44001`, `multilabel_12557`.
- **Script-held-out (×2):** `bangla`, `romanized`.
- Every held-out config asserts `intersection(train∪val, test) == 0` on `uid`.

---

## 7. Preprocessing

Light, script-safe normalisation (heavy rewriting destroys abuse signal carried in emoji, elongation,
hashtags, and mentions):

1. NFKC Unicode normalisation
2. strip zero-width / invisible characters
3. mask URLs → `[URL]`, mentions → `[USER]`
4. normalise hashtags → `[HASHTAG] topic`
5. map emojis/emoticons → `[EMOJI]`
6. collapse character elongation (`খারাপপপপপ` → shorter)
7. whitespace normalisation → `text_clean`

---

## 8. Model Architecture (shared)

All encoders use the same classification head.

```mermaid
flowchart TD
  T["comment → text_clean"] --> E["Encoder backbone"]
  E --> PL["pool = 0.5·CLS + 0.5·mean"]
  PL --> L1["Linear(h→384) → GELU → LayerNorm"]
  L1 --> MSD["Multi-Sample Dropout ×N (train)"]
  MSD --> O["Linear(384 → 5 logits)"]
```

| Component | Setting |
|---|---|
| Pooling | 0.5·CLS + 0.5·mean |
| Head | Linear(h→384) → GELU → LayerNorm → MSD → Linear(384→5) |
| `token_type_ids` | used for BERT-family (BanglaBERT/BanglishBERT/MuRIL), skipped for XLM-R |
| Precision | fp16 mixed |

| Backbone | HF id | Family | Role |
|---|---|---|---|
| BanglishBERT | `csebuetnlp/banglishbert` | ELECTRA | bilingual (Bangla + Romanized) |
| BanglaBERT | `csebuetnlp/banglabert` | ELECTRA | **Bangla-script specialist (script-isolated)** |
| MuRIL | `google/muril-base-cased` | BERT | multilingual / transliteration |
| XLM-R | `xlm-roberta-base` | RoBERTa | multilingual baseline |

---

## 9. Proposed Model A — Script-Aware Ensemble (main)

A deliberately **minimal** per-encoder recipe — **cross-entropy + FGM adversarial training only** —
across four encoders × three seeds, fused by a **script-masked** weighted-logit ensemble. The
minimalism is **ablation-driven** (Section 13): on this benchmark, FGM was the component that
consistently earned its place, so Model A keeps it and drops the rest, trading a heavier recipe for
speed and simplicity.

```mermaid
flowchart TD
  subgraph FT["Fine-tune (CE + FGM), 3 seeds each"]
    B1["BanglishBERT<br/>both scripts"]
    B2["BanglaBERT<br/>Bangla rows only"]
    B3["MuRIL"]
    B4["XLM-R"]
  end
  B1 & B2 & B3 & B4 --> LG["per-run val/test logits (12 sets)"]
  LG --> MK["script mask:<br/>BanglaBERT = 0 on Romanized rows"]
  MK --> W["Nelder-Mead weights on val<br/>(per-row renormalised)"]
  W --> P["weighted-logit fusion → prediction"]
```

**Script-aware contract.** BanglaBERT trains/validates/tests on **Bangla rows only**; on Romanized
rows it emits neutral (zero) logits and receives **zero ensemble weight**, with the remaining encoders
re-normalised per row. BanglishBERT is the bilingual workhorse; MuRIL and XLM-R are full-scope.

---

## 10. Proposed Model B — Full-Stack BanglishBERT (alternate)

A single bilingual encoder trained with the **complete** stack, then **seed-ensembled**.

```mermaid
flowchart TD
  X["BanglishBERT (bilingual)"] --> S["full stack:<br/>focal+CW · balanced sampler · MSD×4<br/>· R-Drop(α=0.5) · FGM · EMA (val-guarded)"]
  S --> Z["train 3 seeds (42/123/456)"]
  Z --> EN["seed-ensemble<br/>(Nelder-Mead on val)"]
  EN --> PR["prediction"]
```

Because BanglishBERT is bilingual, Model B uses **no script isolation** — it is the natural "single
strong model" counterpoint to Model A's "specialised committee". The two systems therefore differ on
**two** axes at once: *architecture* (one encoder vs. four) and *recipe* (full stack vs. CE+FGM),
which is exactly the contrast RQ2 asks about.

---

## 11. Training Configuration

| Hyperparameter | Value |
|---|---:|
| max_length | 128 |
| batch_size / grad_accum / effective | 32 / 1 / **32** |
| epochs | 8 (early stopping, patience 3) |
| encoder LR / head LR | 2e-5 / 8e-5 |
| LR decay | **none (uniform)** — see ablation |
| label smoothing | 0.03 |
| dropout / head hidden | 0.25 / 384 |
| class_weight_beta | 0.999 |
| focal γ | 2.0 *(Model B only)* |
| R-Drop α | 0.5 *(Model B only)* |
| FGM ε | 1.0 *(both models)* |
| EMA decay | 0.999 *(Model B only, val-guarded)* |
| sampler α | 0.5 *(Model B only)* |
| precision | fp16 |
| num_workers | 4 (cloud) / 0 (Windows local) |

Effective batch is held at 32 across GPUs (physical 32 × accum 1 on 24–48 GB cards). **EMA is
val-guarded** — kept only when it beats raw weights on validation, so it can never hurt.

---

## 12. Baselines (NB04)

TF-IDF + Logistic Regression, TF-IDF + Linear SVM, and a character-level BiLSTM, all on the 5-class
20% test.

| Model | Macro-F1 | Accuracy | MCC | AUROC |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | ⟨pending — `04_outputs/baselines/baseline_results.csv`⟩ | | | |
| TF-IDF + Linear SVM | ⟨pending⟩ | | | |
| char-BiLSTM | ⟨pending⟩ | | | |

---

## 13. Ablation (NB08)

**Component ablation** — reference = **CE + FGM**, then each component added individually, plus the
full stack:

| Configuration | Macro-F1 | Δ vs CE+FGM |
|---|---:|---:|
| CE + FGM (Model A recipe) | ⟨pending — `04_outputs/ablation/component_ablation.csv`⟩ | — |
| + focal + class weights | ⟨pending⟩ | |
| + balanced sampler | ⟨pending⟩ | |
| + multi-sample dropout | ⟨pending⟩ | |
| + R-Drop | ⟨pending⟩ | |
| + EMA | ⟨pending⟩ | |
| ALL (Model B recipe) | ⟨pending⟩ | |

**Taxonomy ablation** — 5-class vs 9-class, same recipe:

| Taxonomy | Macro-F1 | Weighted-F1 | Accuracy | MCC |
|---|---:|---:|---:|---:|
| 5-class (headline) | ⟨pending — `04_outputs/ablation/taxonomy_ablation.csv`⟩ | | | |
| 9-class | ⟨pending⟩ | | | |

**Figure — Component ablation**
![Ablation](https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/finalized_outputs/figures/04_ablation.png)

**Plain summary:** Each bar shows what adding one component does on top of the CE+FGM reference. This is
the evidence behind Model A's "less-is-more" recipe and behind keeping Model B as a separate
full-stack contrast rather than the single headline system.

> **Methodology note (honest).** A *pre-flight* sanity check on a small **balanced** sample suggested
> only FGM helped; we flagged that a balanced sample structurally hides the benefit of the imbalance
> components, so the **real, imbalanced** NB08 numbers above are the authoritative ones that decide
> the final recipe. A surprising earlier finding — that **uniform LR beats LR decay** — is why the
> final config uses uniform learning rates.

---

## 14. Main Results — Benchmark (20% in-domain test)

5-class benchmark scheme (`none/abusive/sexual/religious/threat`).

| System | Macro-F1 | Weighted-F1 | Accuracy | MCC | Macro-AUROC |
|---|---:|---:|---:|---:|---:|
| Best baseline (RF/SVM) | ⟨pending — `baselines/`⟩ | | | | |
| **Model A — Script-Aware Ensemble** | ⟨pending — `04_outputs/ensemble/ensemble_test_metrics.json`⟩ | | | | |
| **Model B — Full-Stack BanglishBERT** | **0.8135** | **0.8224** | **0.8222** | **0.7291** | **0.9534** |

Per-encoder single-model scores for Model A: ⟨pending — `04_outputs/models_main/per_run_summary.csv`⟩.
Model B per-seed validation Macro-F1: 0.8034 / 0.8018 / 0.8056; **seed-ensemble** lifts test Macro-F1
to 0.8135 (best single seed 0.8086).

### 14.1 Per-class F1 — Model B (final)

| Class | F1 |
|---|---:|
| none | 0.8675 |
| religious | 0.8953 |
| sexual | 0.8226 |
| threat | 0.7608 |
| abusive | 0.7214 |

**Figure — Per-class F1**
![Per-class F1](https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/finalized_outputs/figures/03_per_class_f1.png)

**Plain summary:** `religious` and `none` are the strongest; `abusive` is the hardest in our scheme
because it is the catch-all bucket that absorbs many heterogeneous behaviours (troll, personal,
political, spam), making its boundary fuzzier than the semantically tight classes.

**Figure — Confusion matrix (ensemble, 20% test)**
![Confusion matrix](https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/finalized_outputs/figures/02_confusion_matrix.png)

**Plain summary:** Most mass sits on the diagonal. The off-diagonal mass concentrates in the
`abusive ↔ none` and `abusive ↔ sexual` cells, which is the same semantic-overlap difficulty the base
paper reports between Troll and Not-Bully.

---

## 15. Robustness — the central study

**Model B, held-out evaluation (1 seed per config):**

| Split | n_test | Macro-F1 | Weighted-F1 | Accuracy | MCC |
|---|---:|---:|---:|---:|---:|
| in-domain (20% test) | 18,865 | **0.8135** | 0.8224 | 0.8222 | 0.7291 |
| source_holdout_facebook_44001 | 43,078 | 0.5828 | 0.6256 | 0.6344 | 0.5186 |
| source_holdout_multilabel_12557 | 8,882 | 0.5579 | 0.5640 | 0.5907 | 0.4444 |
| source_holdout_bd_shs | 5,029 | 0.4549 | 0.5657 | 0.5383 | 0.3943 |
| source_holdout_banth | 37,334 | 0.2165 | 0.6051 | 0.6425 | 0.0845 |
| script_holdout_romanized | 37,334 | 0.2165 | 0.6051 | 0.6425 | 0.0845 |
| script_holdout_bangla | 56,989 | 0.1631 | 0.2617 | 0.3898 | 0.1088 |
| **mean held-out** | — | **0.3653** | 0.5379 | — | — |

**Model A (Script-Aware Ensemble) robustness:** ⟨pending — `04_outputs/robustness/robustness_summary.csv`⟩.

**Key observations.**
1. **In-domain is strong; transfer degrades by *type* of shift.** Source shift (same script) costs a
   lot but stays usable (0.45–0.58); **script shift collapses to 0.16–0.22.**
2. **Cross-script transfer is the open wall.** Note the tell-tale gap on `script_holdout_romanized`:
   Accuracy/Weighted-F1 ≈ 0.64/0.61 but Macro-F1 = 0.22. The model **defaults to the majority class**
   on the unfamiliar script, so accuracy looks fine while minority-class F1 falls to near zero.
3. `source_holdout_banth` **≡** `script_holdout_romanized` (identical numbers) because `banth` is the
   only Romanized source — reported transparently, **not** counted as two independent results.
4. This is exactly why **Model A is script-aware**: a single model cannot bridge the scripts, so we
   split the work between a Bangla specialist (BanglaBERT) and a bilingual encoder (BanglishBERT).

**Figure — Robustness across held-out splits**
![Robustness](https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/finalized_outputs/figures/05_robustness.png)

**Plain summary:** The drop from in-domain to script-held-out is the paper's most important empirical
message: dual-script generalisation, not in-domain accuracy, is the real frontier for Bengali
cyberbullying detection.

---

## 16. Base-Paper Comparison (Facebook-44K, native 5-class)

Protocol matched to the base paper: Facebook 44,001 → within-source dedup **43,112** → native classes
(Not Bully, Sexual, Troll, Religious, Threat) → **70/15/15** (train 30,178 / val 6,467 / test 6,467).
Both proposed systems are retrained on this split.

| Class | Base paper (H&S 2025) | **Model B (ours)** | Model A (ours) |
|---|---:|---:|---:|
| Not Bully | 0.9151 | 0.8813 | ⟨pending — `04_outputs/basepaper/comparison.json`⟩ |
| Sexual | 0.8845 | 0.8711 | ⟨pending⟩ |
| Troll | 0.8446 | 0.8122 | ⟨pending⟩ |
| Religious | 0.9374 | 0.9366 | ⟨pending⟩ |
| **Threat** | 0.7579 | **0.8337** | ⟨pending⟩ |
| **Macro-F1** | **0.8923** | **0.8670** | ⟨pending⟩ |

Model B: Weighted-F1 0.8703, Accuracy 0.8703, MCC 0.8266, Macro-AUROC 0.9719; per-seed val Macro-F1
0.8496 / 0.8508 / 0.8512.

**Figure — Base-paper comparison**
![Base-paper comparison](https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/finalized_outputs/figures/06_basepaper.png)

**Plain summary:** On the base paper's own dataset our general-purpose system is a touch below on
overall Macro-F1 (0.867 vs 0.892) but **beats it by ~7.6 points on the hardest class, `Threat`** — the
class that matters most for real-world harm. We frame this honestly: the base paper is tuned to one
clean source; our value is breadth (four sources, two scripts) and robustness, not a single-dataset
win.

---

## 17. Error Analysis

- Macro vs. accuracy gap under script shift = **majority-class collapse** (Section 15).
- In-domain confusions concentrate on the `abusive` catch-all (overlaps `none`, `sexual`).
- Hardest sources: `bd_shs` (small, different collection context) and `multilabel_12557` (multi-label
  annotation converted to single-label introduces label noise on transfer).
- Full per-source / per-script error breakdown: ⟨pending — derived in NB10 from the committed metrics⟩.

---

## 18. Key Design Decisions

| Decision | Rationale |
|---|---|
| Deduplicate (vs. keep) | honest held-out evaluation; no duplicate leakage |
| 5-class (headline), 9-class (ablation only) | 5 classes are well-populated and comparable to the base paper; 9 are sparse |
| Priority `threat > sexual > religious > abusive > none` | keep explicit-violence and identity-targeted abuse from being absorbed into the generic bucket |
| Two proposed systems | RQ2: minimal script-aware committee vs. heavy single encoder |
| Script-isolated BanglaBERT | robustness shows cross-script transfer fails; specialise instead |
| CE + FGM for Model A | ablation-driven minimalism (FGM the consistent winner) |
| Uniform LR (no decay) | ablation showed decay hurt |
| Macro-F1 primary | equal weight to rare classes (`threat`, `religious`) |
| 3 seeds + ensemble | stability + a genuine ensemble signal |

---

## 19. Limitations & Honest Claims

1. **Single-task, 5-class.** The system predicts a 5-class abuse type; do not claim severity prediction
   or a separate binary head as a headline (binary-equivalent = `none` F1).
2. `source_holdout_banth` **≡** `script_holdout_romanized`; reported as one finding, not two.
3. On the base paper's single clean dataset we are **slightly below** on overall Macro-F1; our claim is
   breadth + robustness + the Threat-class gain, not a single-dataset SOTA.
4. Robustness for Model B uses **1 seed per held-out config** (tractability); the in-domain and
   base-paper numbers use the full 3-seed ensemble.
5. The 0.5/0.5 CLS+mean pooling is a balanced empirical choice, not a proven optimum.
6. Some result cells are **⟨pending⟩** in this log until the committed output files are pasted in; they
   are not estimates.

---

## 20. Viva / Seminar Q&A

**Q1. What is genuinely new here?** A deduplicated four-source, dual-script Bengali benchmark; two
contrasting proposed systems; and the first source-/script-held-out robustness study for this task.

**Q2. Why two models instead of one?** To answer whether a *minimal, script-aware committee* (A) can
rival a *heavily regularised single encoder* (B). They differ on architecture and recipe at once,
which is the comparison the paper is built around.

**Q3. Why is BanglaBERT restricted to Bangla?** Because robustness shows cross-script transfer fails;
feeding a Bangla-pretrained model Romanized text degrades it and pollutes the ensemble, so we mask it
off Romanized rows.

**Q4. Why does cross-script Macro-F1 crater while accuracy stays ~0.64?** Majority-class collapse: on
an unfamiliar script the model predicts `none` for most inputs, which is often correct (high accuracy)
but kills minority-class recall (low Macro-F1).

**Q5. You're below the base paper on Facebook — isn't that bad?** On *overall* Macro-F1, marginally;
but we **improve the hardest class (`Threat`)** and, unlike the base paper, generalise across four
sources and two scripts. Benchmarks are judged on rigour and new axes, not one dataset's leaderboard.

**Q6. Why Macro-F1?** It weights every class equally, so the rare `threat`/`religious` classes are not
hidden by the dominant `none` class.

**Q7. Why uniform LR?** The ablation showed LR decay reduced performance on this multi-source,
dual-script mixture.

**Q8. Why deduplicate now?** Duplicates (especially in `banth`) would leak across held-out splits and
inflate robustness numbers; removing them makes the evaluation trustworthy.

---

## 21. Reproducibility — Pipeline & Outputs

```mermaid
flowchart TD
  N1["NB01 inventory"] --> N2["NB02 clean + consolidate<br/>dedup 135,575→94,337 · 89→5 classes"]
  N2 --> N3["NB03 splits<br/>random 70/10/20 · 4 source · 2 script"]
  N3 --> N4["NB04 baselines"]
  N3 --> N5["NB05 fine-tune<br/>4 encoders × 3 seeds · CE+FGM"]
  N5 --> N6["NB06 ensemble (Model A)<br/>script-masked weighted fusion"]
  N3 --> N8["NB08 ablation<br/>components · 5-vs-9 class"]
  N3 --> N7["NB07 robustness (Model A)"]
  N3 --> N7b["NB07b Model B (alt)<br/>benchmark + robustness + base-paper"]
  N3 --> N9["NB09 base-paper (Model A)"]
  N4 & N6 & N7 & N7b & N8 & N9 --> N10["NB10 tables + figures"]
```

**Output tree (`04_outputs/`, per `output_dir.txt`):**
```
04_outputs/
├── baselines/baseline_results.csv
├── models_main/                 # NB05: 4 enc × 3 seeds (results.json), per_run_summary.csv
├── ensemble/                    # NB06 (Model A): ensemble_test_metrics.json, test_pred/proba.npy, cm_ensemble_20pct.png
├── ablation/                    # NB08: component_ablation.csv, taxonomy_ablation.csv
├── robustness/                  # NB07 (Model A): robustness_summary.csv + per-config
├── altmethod/                   # NB07b (Model B): benchmark / robust_* / basepaper / *_summary.json
├── basepaper/comparison.json    # NB09 (Model A) vs base paper
├── fig_label5_distribution.png
├── paper/                       # NB10: table1..6 (.csv/.tex), fig_per_class/ablation/robustness/basepaper.png
└── finalized_outputs/figures/   # curated, renamed figures (01_..06_) used in this log
```

---

## 22. Asset Index (figures used above)

| # | Figure | Source file (`04_outputs/…`) | Status |
|---|---|---|---|
| 01 | Class distribution | `fig_label5_distribution.png` | exists |
| 02 | Confusion matrix (ensemble, 20%) | `ensemble/cm_ensemble_20pct.png` | exists |
| 03 | Per-class F1 | `paper/fig_per_class_f1.png` | exists |
| 04 | Component ablation | `paper/fig_ablation.png` | exists |
| 05 | Robustness | `paper/fig_robustness.png` | exists |
| 06 | Base-paper comparison | `paper/fig_basepaper.png` | exists |

Figures are embedded from `…/04_outputs/finalized_outputs/figures/0N_*.png`. If your curated filenames
differ, adjust the six URLs (or point them at the source paths in the table above). All **diagrams**
in this log are Mermaid placeholders to be replaced with rendered PNGs later.

---

## 23. Changelog vs. the previous draft

| Previous draft | This version (current project) |
|---|---|
| Multi-task (binary + 9-class) | **Single-task 5-class** (9-class only in ablation) |
| 3 encoders | **4 encoders** (adds BanglishBERT; BanglaBERT script-isolated) |
| Kept duplicates | **Deduplicated** (135,575 → 94,337) |
| 80/10/10 | **70/10/20** |
| One system | **Two proposed systems** (A: script-aware CE+FGM ensemble; B: full-stack BanglishBERT) |
| No script-aware masking | **Script-aware ensemble** (BanglaBERT masked on Romanized) |
| Heavier default recipe | Model A is **ablation-driven minimal (CE+FGM)**; Model B keeps the full stack |

---

*BanglaCyberBench — Experiment Log v4 | June 2026 | Sefayet Alam, Naim Parvez and A. F. M. Minhazur Rahman*