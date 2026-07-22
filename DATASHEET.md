# Datasheet — BanglaCyberBench

This datasheet follows the *Datasheets for Datasets* framework (Gebru et al., 2021).
It documents BanglaCyberBench, the consolidated benchmark described in the paper
*BanglaCyberBench: A Dual-Script Benchmark and a Script-Aware Ensemble for Bengali
Cyberbullying Detection*. The paper is the authoritative source; this file is a
navigational summary and points back to it for anything it does not settle.

## Motivation

**Why was the dataset created?** Bengali abuse-detection research has mostly worked
on one dataset, in one script, evaluated on a random split of that single corpus.
That leaves the deployment question — does a classifier trained on one collection
work on comments from somewhere else — untested. BanglaCyberBench consolidates four
public Bengali sources into one deduplicated, dual-script, five-class resource so
that in-domain and cross-source performance can be measured on the same footing.

**Who created it and who funded it?** The authors of the accompanying paper
(Department of CSE, RUET). The research received no specific external grant.

## Composition

- **What do the instances represent?** Individual social-media comments in Bengali,
  each labelled with one of five harmful-content classes.
- **How many instances?** 94,323 unique comments after deduplication (from 135,575
  raw rows; 41,252 duplicates removed).
- **Classes and distribution.** `none` 47,312 (50.16%), `abusive` 24,963 (26.47%),
  `sexual` 10,822 (11.47%), `religious` 8,032 (8.52%), `threat` 3,194 (3.39%).
- **Script coverage.** Bangla script 56,989 (60.4%); Romanized Bangla / Banglish
  37,334 (39.6%). The four sources are script-homogeneous, so the script tag is
  inherited from source provenance rather than detected per comment.
- **Sources.** `facebook_44001` (43,078; Bangla; Mendeley), `banth` (37,334;
  Romanized; NAACL 2025 release), `multilabel_12557` (8,882; Bangla; Mendeley),
  `bd_shs` (5,029; Bangla; LREC 2022). See `DATA_LICENSE.md` for terms.
- **What does each instance consist of?** Fields: `text`, `text_clean`,
  `label_binary`, `label_type`, `label5`, `source`, `script`, `uid`.
- **Label semantics — read this carefully.** The labels type the harmful *content*
  of an isolated comment. They do **not** capture cyberbullying in its interactional
  sense (repetition, power imbalance, relationship history, victim impact); the
  comment-level sources record none of those signals. Treat the task as harmful-
  language typing, not cyberbullying detection in the fuller sense.
- **Is there label noise / disagreement?** Yes, and it is documented rather than
  hidden. Among 1,287 cleaned-text groups that span multiple sources, 661 (1,721
  rows) carry conflicting five-class labels across sources and 271 (767 rows)
  carry conflicting binary flags. Consolidation resolves conflicts by a fixed
  priority rule (`threat > sexual > religious > abusive > none`); this is a
  deterministic construction policy, not a claim of a universal hierarchy.
- **Recommended splits.** Stratified random 70/10/20 on `label5`
  (66,026 / 9,432 / 18,865). Source-held-out configurations hold out one
  Bangla-script source at a time and train on the other two Bangla sources; the
  sole Romanized source (`banth`) is excluded from those configurations because it
  couples source and script shift.

## Collection process

The dataset is a consolidation of four *previously collected* public corpora, not a
new data-collection effort. Original collection methods are those of the source
providers; consult the cited source papers. No new comments were scraped or
solicited for this benchmark.

## Preprocessing / cleaning / labeling

- **Cleaning is deliberately light and script-safe:** NFKC normalisation; zero-width
  and invisible characters stripped; URLs/mentions masked to `[URL]`/`[USER]`;
  hashtags rewritten to `[HASHTAG]` keeping the tag word; emojis demojised to
  bracketed tokens; runs of 3+ identical characters collapsed to 2; whitespace
  normalised. No lowercasing beyond Unicode, no punctuation stripping, no stopword
  removal — repetition, punctuation, and emoji all carry abuse signal.
- **Deduplication** is exact-match on `text_clean`, before any split. Fuzzy /
  paraphrase near-duplicates are **not** removed; any residual near-duplicate that
  crosses the train/test boundary would inflate in-domain scores.
- **Label consolidation:** 89 raw label strings folded into five classes by the
  priority rule above; full map in the paper (Table 5) and the released code.
- **Human check:** two of the authors independently re-labelled a 500-item
  stratified sample (Cohen's κ = 0.709). This is an internal consistency check on
  the scheme, **not** external gold validation of all labels, and it has no
  adjudication and no interval estimate.

## Uses

- **Intended:** research on Bengali/low-resource abuse detection, benchmark
  construction, and cross-distribution transfer evaluation.
- **Out of scope / discouraged:** harassment, profiling, surveillance, generation
  of abusive content, or deployment as a moderation system without target-domain
  validation. Source-held-out Macro-F1 falls to 0.46–0.59, so in-domain accuracy is
  not a proxy for deployment readiness. Romanized five-class Macro-F1 is ~0.39, so
  the resource exposes a script gap rather than demonstrating dual-script competence.
- **Known biases:** heavy class imbalance (~14.8:1 between the most and least
  frequent class); label definitions inherited from four different annotation teams;
  `gender` folded under `sexual` and `origin`-based slurs folded into generic
  `abusive`, which some governance contexts would classify differently.

## Distribution

- **How is it distributed?** As code, manifests, and derived outputs in the public
  GitHub repository, plus a tagged release. Raw comment text is redistributed only
  where the source licence permits; otherwise the deterministic pipeline
  reconstructs the benchmark from the cited source downloads.
- **Licence:** original code and documentation MIT; dataset text and pretrained
  models remain under their own terms (`DATA_LICENSE.md`).
- **PII / fairness:** no comprehensive PII audit, fairness analysis, or
  counterfactual identity-term testing was performed; this is flagged as a
  prerequisite before any deployment use.

## Maintenance

- **Contact / takedown:** source-specific removal requests are honoured through the
  documented route; deletions propagate to the redistributed manifests and the
  reconstruction index so a removed item does not reappear on rebuild.
- **Versioning:** the artifact is released under version tags; the paper cites the
  tagged release rather than the mutable `main` branch. A Zenodo DOI will be added.
