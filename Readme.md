# BanglaCyberBench — Fix & Upgrade Plan (Journal-Worthy)

**Project:** BanglaCyberBench: A Robust Multi-Source Benchmark and Transformer Ensemble for Cyberbullying Detection in Bengali
**Document purpose:** Single source of truth for the revision pass. Resolves all 6 issues raised, in dependency order. From the next prompt we fix notebooks one at a time.
**Status legend:** ✅ Done · 🔄 In Progress · ⬚ Pending · ❌ Blocked

---

## 0. Why this revision exists

The current pipeline produces strong headline numbers but carries three structural problems that a Q1 reviewer would flag, plus three smaller presentation gaps. The most serious is a **fine-tuning leakage in the source-holdout robustness claim** (Issue 5) — if not fixed, the central "robustness" contribution is invalid. This document sequences the fixes so that each one is done before anything that depends on it.

---

## 1. The Six Issues — Decisions Locked

| # | Issue | Decision | Requires retrain? |
|---|---|---|---|
| 1 | Binary classification over-common in literature | Do **not** run a separate extensive binary study. Use the `none` class F1 (~0.93) as implicit evidence the model handles binary detection. One framing sentence in Results/Discussion. | No |
| 2 | ~40,590 duplicates kept in merged set | **Deduplicate** the merged dataset before splitting. Make the benchmark unique. | Yes (everything downstream) |
| 3 | `gender` F1 too low (0.56) + final test size | Two parts: (a) fix stratification so rare classes are preserved; (b) keep 80/10/10 but report **final metrics on val+test combined (20%)** after everything is frozen. | Yes (re-split + re-eval) |
| 4 | `lr_decay` helps when removed (ablation) | Treat "no layer-wise LR decay" as a **fixed design choice**, not a tunable hyperparameter. Final models trained **without** lr_decay. | Yes |
| 5 | Source/script-holdout leakage in NB08 | **Critical fix.** Train dedicated holdout models on the per-source/per-script `*_train.csv` splits (which exclude the held-out source), then evaluate those models on the held-out test. No more reusing the random-split checkpoints for holdout claims. | Yes (4 source + 1 script retrains) |
| 6 | Need individual-model metrics (MCC, Acc, etc.) | Add a complete per-model metric table (Acc, Macro-F1, Weighted-F1, MCC, AUROC for both tasks) so ensemble-vs-individual gain is explicit. | No |

---

## 2. Detailed Resolution Per Issue

### Issue 1 — Binary framing (no retrain)
**What's true now:** `label_binary` exists; `none` maps to "not harmful" (label 0). In the 9-class ensemble report, `none` has Precision/Recall/F1 ≈ 0.93 over 7,830 support.
**Action:** In the paper, add one sentence in Results: the model's ability to separate harmful vs. non-harmful is captured by the `none` class performance (~0.93 F1), so binary detection is handled implicitly without a dedicated binary benchmark. Drop any plan for an extensive standalone binary comparison table.
**Where:** Paper text only (NB10 / writing phase). No code change beyond optionally surfacing the `none`-as-binary note in NB07 commentary.

### Issue 2 — Deduplication (retrain trigger)
**What's true now:** Experiment log §8 records 40,590 duplicates **intentionally kept**. Merged total = 135,575.
**Action:**
- In **NB02 (preprocessing)**, add a deduplication step. Decide dedup key: exact match on cleaned text, OR (text + label) pair. Recommended: dedup on `text_clean` after normalization, keeping the first occurrence and its label. If the same text carries conflicting labels across sources, resolve by the existing priority rule (threat > sexual > religious > gender > political > abusive > personal > other > none) for `label_type`, and majority/any-positive for `label_binary`.
- Record the exact post-dedup count and the per-source / per-class distribution. Update experiment log §8 to reflect that the final benchmark is now deduplicated (reverse the old "we kept them" statement).
- Re-save `benchmark_cleaned.csv`.
**Downstream consequence:** NB03 split must re-run, then NB04–NB09 all re-run.
**Honesty note for paper:** This strengthens the benchmark — a deduplicated multi-source set is a cleaner contribution than one with cross-source overlap.

### Issue 3 — Stratification fix + final eval on 20% (retrain trigger)
**Problem A — `gender` stratification.** Current splits stratify **only on `label_binary`**. With `gender` having ~585 train / 90 test samples, binary-only stratification lets the 9-class minorities scatter unevenly, hurting `gender` F1.
**Action A:** In **NB03**, change the random-split stratification to stratify on a **combined key** or on `label_type` (the 9-class column) so every rare class is proportionally represented in train/val/test. Practical approach: stratify on `label_type` (9 classes) — this automatically keeps `none` (and therefore binary balance) roughly preserved since `none` dominates. Verify post-split per-class counts for all 9 classes in each split and print them.
- Guard: if any class has too few samples to stratify cleanly after dedup, log it explicitly and document in limitations.

**Problem B — final test pool.** You want a larger, more stable final evaluation.
**Action B (confirmed acceptable):** Keep 80/10/10. Use the 10% val during training (early stopping) and ensemble tuning (NB06 Nelder-Mead) **as now**. After ALL training, ensemble weights, and threshold are **frozen**, run a **one-shot final evaluation on val+test combined (~20%)**.
**Hard rule:** Nothing may be tuned after seeing the combined-20% numbers. If any decision is revisited, the val portion is contaminated and this is void. Report in the paper: "validation set used for early stopping and ensemble weight optimization; final metrics reported on the combined held-out 20% (val+test) after all components frozen."
- The 10%-only test number can still be reported as a secondary, fully-clean figure for transparency.
- Source/script holdouts are unaffected — they already use the held-out source as test.

### Issue 4 — Remove lr_decay as a design choice (retrain trigger)
**What's true now:** NB09 ablation: removing layer-wise LR decay improved Binary Macro-F1 0.9081 → 0.9266 (+0.0185) and abuse-type 0.7424 → 0.7826 (+0.0402) — the largest single ablation effect.
**Action:**
- In **NB05 (fine-tuning)**, set the optimizer to the **uniform-LR path** (no layer-wise decay) as the default/final configuration. Concretely, use the `no_lrdecay` branch (encoder_lr + head_lr, no per-layer multiplier) for the production runs that feed the ensemble.
- Keep `lr_decay` only inside NB09 as the ablation comparison ("w/ vs w/o layer-wise LR decay"). Do not expose it as a tuning knob in the main training narrative.
- Update experiment log §16 to state lr_decay was evaluated and **rejected** based on ablation evidence; the final model uses uniform LR.
**Consequence:** All 9 ensemble runs (3 models × 3 seeds) retrain without lr_decay. Expect the reported transformer + ensemble numbers to rise.

### Issue 5 — Proper source/script holdout (CRITICAL retrain trigger)
**Root cause (confirmed):** NB08 loads the 9 checkpoints from `outputs/models_v2_fix/`, which were fine-tuned in NB05 on the **random 80% split** drawn from the full merged set — i.e., it already contains samples from banth, bd_shs, facebook, multilabel. Evaluating those same checkpoints on `source_holdout_banth_test.csv` is **not** a held-out test: the model saw banth during fine-tuning. Same logic invalidates the script holdout.
**Pretraining check (resolved, not a problem):** The backbones come from standard public weights — BanglaBERT = `csebuetnlp/banglabert` (BUET, ELECTRA), MuRIL = `google/muril-base-cased`, XLM-R = `xlm-roberta-base`. None were pretrained on your 4 datasets, so there's no pretraining contamination. The contamination is purely at the **fine-tuning** stage.
**The fix — train dedicated holdout models:**
- NB03 already produced the correct splits: `source_holdout_{src}_train.csv` / `_val.csv` (rest-of-data, holdout source excluded) and `source_holdout_{src}_test.csv` (the held-out source). Same for `script_holdout_*`.
- For each of the **4 source holdouts + 1 script holdout (5 settings)**, fine-tune the model(s) on that setting's `_train` split and evaluate on its `_test` split. The held-out source/script is now genuinely unseen during fine-tuning.
- **Cost-control decision needed (ask user):** do we run the full 3-models × 3-seeds ensemble for each of the 5 holdouts (= 45 extra runs), or a reduced protocol (e.g., best single model BanglaBERT × 1–3 seeds per holdout)? Recommendation for journal credibility vs. compute: **BanglaBERT × 3 seeds per holdout** (15 runs) as the primary robustness result, optionally the full ensemble for 1–2 representative holdouts. Lock this before NB05/NB08 rework.
- **NB08 rework:** point each holdout evaluation at its **own** dedicated checkpoints (trained on the matching `_train` split), not the random-split checkpoints. Add an explicit assertion/printout per holdout confirming "training split excluded source = X" so the no-leakage property is self-documenting.
- **Old numbers are discarded.** The current robustness table (banth 0.9777, etc.) is leakage-inflated and must be regenerated. Expect genuine holdout numbers to be **lower** — that is correct and more credible.
**Paper framing:** This becomes a genuine cross-source / cross-script generalization study. Lower-but-honest numbers with a proper protocol are far more publishable than inflated ones.

### Issue 6 — Individual-model metrics (no retrain, but regenerate after retrains)
**What's true now:** NB07 Table 1 shows transformers with Acc / Macro-F1 / Weighted-F1 / MCC (binary, averaged over seeds), AUROC shown as `--`. Abuse-type per-model shows only Macro-F1 in the headline; per-seed JSON has abuse_type MCC.
**Action:** In **NB07 (paper assets)**, build a complete per-model table:
- For each backbone (BanglaBERT, MuRIL, XLM-R), report **both tasks**: Accuracy, Macro-F1, Weighted-F1, MCC, and AUROC (binary; AUROC N/A or one-vs-rest macro for 9-class) — mean ± std across the 3 seeds.
- Add a clear **Ensemble vs. best-single-model delta** row/column so the ensemble gain is quantified (e.g., binary Macro-F1 best single 0.9095 → ensemble 0.9247).
- Compute AUROC for the individual transformers (currently missing) using saved logits, so the column is no longer `--`.
**Note:** This must be regenerated AFTER Issues 2/4/5 retrains, since the underlying numbers change.

---

## 3. Execution Order (dependency-sorted)

Each step gates the next. Do not start a step until the prior one is confirmed.

| Step | Notebook | What changes | Depends on | Status |
|---|---|---|---|---|
| S1 | **NB02** preprocessing | Add deduplication (Issue 2). Re-save `benchmark_cleaned.csv`. Print post-dedup counts & per-class/per-source distribution. | — | ⬚ |
| S2 | **NB03** data splits | Re-split on deduped data. Stratify random split on `label_type` (9-class) not just binary (Issue 3A). Verify all 9 classes present in each split. Regenerate source/script holdout splits. Confirm `random_state=42` reproducibility. | S1 | ⬚ |
| S3 | **NB04** baselines | Re-run on new splits (numbers shift after dedup). | S2 | ⬚ |
| S4 | **NB05** fine-tuning (main) | Set **uniform LR / no lr_decay** as production config (Issue 4). Retrain 3×3 = 9 runs on new random split. Save val+test logits as before. | S2 | ⬚ |
| S5 | **NB05-holdout runs** | Train dedicated holdout models on each `*_holdout_*_train.csv` (Issue 5). Protocol (full ensemble vs BanglaBERT×3) per locked decision. Save per-holdout checkpoints to a distinct dir. | S2 | ⬚ |
| S6 | **NB06** ensemble + threshold | Re-optimize ensemble weights (Nelder-Mead on new val logits) and threshold on the new 9 main-run checkpoints. Freeze everything here. | S4 | ⬚ |
| S7 | **NB08** robustness | Rework to load **per-holdout** checkpoints from S5; evaluate each on its matching held-out test. Add no-leakage assertion print per holdout. Regenerate robustness table. | S5, S6 | ⬚ |
| S8 | **Final eval** | One-shot evaluation of frozen ensemble on **val+test combined (20%)** for headline numbers (Issue 3B). Also keep clean 10%-test figure. | S6 | ⬚ |
| S9 | **NB07/NB09** assets + ablations | Complete per-model metric table incl. AUROC + ensemble-vs-single delta (Issue 6). Re-run ablations on new data; keep lr_decay only as ablation comparison. | S6, S7, S8 | ⬚ |
| S10 | **NB10** paper assets / writing | Regenerate all figures/tables. Add binary-via-`none` framing sentence (Issue 1). Update prior-work comparison with honest new robustness numbers. | S7, S8, S9 | ⬚ |

---

## 4. Things to Confirm Before We Start Fixing (open questions for next prompt)

1. **Dedup key (Issue 2):** dedup on `text_clean` alone, or on `(text_clean, label)`? And conflict-resolution rule when same text has different labels across sources — confirm priority-rule approach.
2. **Holdout protocol (Issue 5):** full 3×3 ensemble per holdout (45 runs) vs. BanglaBERT×3 per holdout (15 runs) vs. another reduced scheme. Compute budget? (GPU = RTX 4060 Ti 8 GB per logs; each full run ≈ 18 min/epoch × 8 epochs.)
3. **Environment:** local (4060 Ti) or Kaggle T4 for the retrains?
4. **Checkpoint dir naming:** confirm where main-run vs holdout-run checkpoints are saved so NB08 can disambiguate (e.g., `models_main/` vs `models_holdout_<src>/`).
5. **AUROC for 9-class (Issue 6):** report one-vs-rest macro AUROC for abuse-type, or restrict AUROC to the binary task only?

---

## 5. Expected Net Effect on the Paper

- **Headline numbers** will move: dedup may lower raw counts; removing lr_decay should raise transformer + ensemble F1; honest holdouts will lower the robustness numbers.
- **Credibility** rises sharply: a deduplicated benchmark, proper cross-source/cross-script protocol, fixed minority-class stratification, transparent val+test final reporting, and explicit ensemble-vs-single gains are exactly the rigor markers Q1 reviewers look for.
- **Narrative** stays intact: multi-source, dual-script, 9-class, multi-task, ensemble — all preserved. The lr_decay ablation becomes a genuine finding rather than an inconsistency.

---

## 6. Issues Log (running)

| # | Issue | Root cause | Resolution | Status |
|---|---|---|---|---|
| 1 | Binary study redundant | Field saturation | Use `none` F1 as implicit binary evidence | Decided |
| 2 | 40,590 duplicates | Intentionally kept in v2 | Deduplicate in NB02 | Planned |
| 3a | `gender` F1 = 0.56 | Stratified on binary only | Stratify on 9-class `label_type` | Planned |
| 3b | Small final test set | Only 10% reported | Final eval on frozen val+test (20%) | Planned |
| 4 | lr_decay inconsistency | Tuned but ablation says remove | Fix as design choice (uniform LR) | Planned |
| 5 | Holdout leakage | NB08 reused random-split checkpoints; held-out source seen during fine-tuning | Train dedicated per-holdout models on excluded-source splits | Planned (critical) |
| 6 | Missing per-model metrics | Only partial table | Full Acc/F1/W-F1/MCC/AUROC + ensemble delta | Planned |

---

*Next prompt: we start at S1 (NB02 deduplication). I'll ask the Section-4 confirmation questions first, then deliver surgical fixes one notebook at a time.*