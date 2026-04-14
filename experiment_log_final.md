
# BanglaCyberBench — Experiment Log (MASTERCLASS UPDATED)

**Project:** BanglaCyberBench: Transformer Ensemble for Cyberbullying Detection in Bengali  
**Authors:** Sefayet Alam (sefayetalam14@gmail.com), Arafat Hossain  
**Date:** April 2026  
**Repo:** github.com/Sefayet-Alam/CyberBully_Detection_Paper  
**Target Venue:** Q1 journal (IPM / ESWA) or ACL/EMNLP findings  
**Status:** ALL EXPERIMENTS COMPLETE ✅  
**Purpose of this file:** This is the single master reference for the whole research. A reader who knows nothing about the project should be able to understand the motivation, data, modeling, results, design decisions, limitations, and likely viva/seminar questions from this file alone.

---

## 0. One-Page Summary

This research tackles **cyberbullying detection in Bengali/Bangla**, where earlier work mostly used smaller datasets, single-source setups, coarser label schemes, and little or no robustness evaluation. We built **BanglaCyberBench**, a **135,575-sample**, **4-source**, **multi-script** benchmark covering both **Bangla script** and **Romanized Bangla**. We standardized heterogeneous datasets into one common schema, consolidated **89 raw abuse-type labels** into a cleaner **9-class taxonomy**, and trained a **multi-task transformer system** that predicts:

1. **Binary harmfulness** — harmful vs not harmful  
2. **Abuse type** — one of 9 abuse categories

We fine-tuned **BanglaBERT, MuRIL, and XLM-R**, each with **3 seeds** (9 total runs), then combined all 9 checkpoints using a **weighted-logit ensemble** optimized on the validation set. The final system achieved:

- **Binary:** Accuracy = **0.9256**, Macro-F1 = **0.9247**
- **Abuse type (9-class):** Accuracy = **0.8688**, Macro-F1 = **0.7746**

The benchmark also includes **source-held-out** and **script-held-out** evaluation to test robustness beyond a normal random split.

---

## 1. Title Explained in Simple Words

### Full title
**BanglaCyberBench: Transformer Ensemble for Cyberbullying Detection in Bengali**

### What each part means

- **BanglaCyberBench**  
  The name of the benchmark introduced in this work.  
  - **Bangla** = language focus  
  - **Cyber** = online/social media setting  
  - **Bench** = benchmark

- **Transformer Ensemble**  
  The final system is not a single model. It combines several transformer models together.

- **Cyberbullying Detection**  
  The core task is to detect harmful online comments and, in the final setup, also identify the abuse type.

- **in Bengali**  
  The language focus is Bengali/Bangla.

### Why the title is appropriate
The paper is both:
1. a **benchmark/data paper** (because we build BanglaCyberBench), and  
2. a **method paper** (because we build and evaluate a strong transformer ensemble).

---

## 2. Beginner Glossary

### 2.1 Dataset / benchmark terms

- **Source**  
  Where the data came from. One public dataset = one source.

- **Multi-source**  
  Uses data from multiple dataset sources instead of only one.

- **Script**  
  The writing system used in the text.

- **Bangla script**  
  Native Bengali writing, e.g., বাংলা

- **Romanized Bangla**  
  Bangla written using English/Latin letters, e.g., *tumi kemon acho*

- **Multi-script**  
  The benchmark contains more than one writing form/script.

- **Label**  
  The answer attached to a text sample.

- **Binary classification**  
  Two classes only. Here: harmful vs not harmful.

- **Multi-class classification**  
  One text must be assigned one class out of several.

- **Multi-label classification**  
  One text may belong to multiple classes at the same time.

- **Taxonomy**  
  The final organized class system used in the paper.

### 2.2 Split / evaluation terms

- **Train set**  
  The data the model learns from.

- **Validation set**  
  The data used to tune the model and choose settings.  
  Fine-tuning means taking a pretrained model and training it further on a task-specific dataset.

- **Test set**  
  Final unseen data for reporting performance.

- **In-domain testing**  
  Train and test come from the same overall mixed benchmark distribution. This is the normal random test split.

- **Source-held-out testing**  
  Hold out one full dataset source from training and test only on that source. This checks **cross-source generalization**.

- **Script-held-out testing**  
  Hold out one writing form/script and test on it. This checks **cross-script generalization**.

### 2.3 Model / metric terms

- **CLS pooling**  
  Uses the transformer's special summary token as a sentence representation.

- **Mean pooling**  
  Averages all token representations.

- **Ensemble**  
  Combines several models to get a stronger final prediction.

- **Threshold tuning**  
  Chooses the decision cutoff for a binary prediction.

- **Logit**  
  The model's raw score before converting to a probability.

- **Accuracy**  
  Fraction of predictions that are correct.

- **Precision**  
  Of the items predicted as a class, how many are actually correct.

- **Recall**  
  Of the true items in a class, how many the model finds.

- **F1-score**  
  Balance between precision and recall.

- **Macro-F1**  
  Compute F1 for each class separately, then average equally. Small classes matter as much as large classes.

- **Weighted-F1**  
  Like F1, but larger classes get more weight.

- **MCC**  
  A balanced binary classification score that uses TP, TN, FP, and FN.

- **AUROC**  
  Measures how well the model separates classes across thresholds.

- **AUPRC**  
  Precision-recall area; useful for positive-class quality.

---

## 3. Research Problem and Motivation

Cyberbullying detection in Bengali is underdeveloped compared with English and some other major languages. Earlier Bangla cyberbullying work had several limitations:

1. **Small datasets** (roughly 2K–44K in prior work)
2. **Single-source bias**
3. **Coarse label granularity** (binary or 5-class)
4. **No strong cross-source / cross-script robustness evaluation**

Our work addresses these issues by building a larger, more diverse benchmark and testing performance beyond a simple random split.

---

## 4. Prior Work Comparison

### 4.1 Ahmed et al. (2021) — Hybrid Neural Network
- Dataset: 44,001 Facebook comments
- Classes: 5
- Method: CNN-LSTM hybrid + SVM ensemble
- Best results: Binary accuracy 87.91%, multiclass accuracy 85.00%
- Limitation: single source, no robustness evaluation

### 4.2 Sihab-Us-Sakib et al. (2024) — XLM-R on CBD
- Dataset: 2,751 labelled texts
- Classes: 5
- Method: XLM-R fine-tuning
- Best result: F1 = 0.83, accuracy = 82.61%
- Limitation: very small dataset, no ensemble, no robustness evaluation

### 4.3 Saifullah et al. (2024) — BullyFilterNeT
- Dataset: 44,001 Facebook comments
- Method: BanglaBERT-based model
- Best result: Accuracy = 88.04%
- Limitation: single source, no robustness evaluation

### 4.4 Hoque & Seddiqui (2025) — Transformer Stacking
- Dataset: 44,001 Facebook comments
- Classes: 5
- Method: stacking of XLM-R, mBERT, and Bangla-BERT
- Best results: Binary F1 = 93.61%, multiclass F1 = 89.23%
- Limitation: single source, no cross-source/cross-script testing, 5-class taxonomy

### 4.5 Main differences of our work

| Difference | Our work |
|---|---|
| Dataset size | 135,575 |
| Number of sources | 4 |
| Script coverage | Bangla + Romanized |
| Final abuse taxonomy | 9 classes |
| Robustness evaluation | Yes |
| Multi-task setup | Yes |
| Ablation study | Yes |

---

## 5. Dataset: BanglaCyberBench

### 5.1 Sources

| Source | Samples | Script | Origin | Raw label style |
|---|---:|---|---|---|
| `banth` | 73,999 | Romanized | Kaggle | One binary harmful column + several 0/1 type columns |
| `bd_shs` | 5,029 | Bangla | Mendeley | One harmful column + one type column |
| `facebook_44001` | 44,001 | Bangla | Mendeley | One main label column (e.g., not bully, sexual, threat, troll, religious) |
| `multilabel_12557` | 12,546 | Bangla | Kaggle | Separate binary columns: bully, sexual, religious, threat, spam |

**Total:** 135,575 samples

### 5.2 Why this benchmark is multi-source
Because it combines **4 different public datasets**, not just one.

### 5.3 Why this benchmark is multi-script
Because it includes:
- **Romanized Bangla:** 73,999 (54.6%)
- **Bangla script:** 61,576 (45.4%)

### 5.4 Binary distribution

| Label | Samples | Percentage |
|---|---:|---:|
| Not Harmful (0) | 75,545 | 55.7% |
| Harmful (1) | 60,030 | 44.3% |

Ratio = **1.26:1**, so the binary task is near-balanced.

---

## 6. How the Datasets Were Merged

The raw datasets had very different schemas. To combine them, we standardized them into a common format.

### 6.1 Unified merged schema

The cleaned merged dataset uses columns such as:

- `text`
- `label_binary`
- `label_type`
- `source`
- `script`
- `original_file`

### 6.2 Why a unified schema was necessary
Without one schema:
- preprocessing would be inconsistent,
- splitting would be messy,
- model training would be harder,
- analysis across sources would be unreliable.

### 6.3 Source-wise raw label mapping

| Dataset | Text column used | Raw binary signal | Raw type signal | Binary mapping | Type mapping into merged benchmark |
|---|---|---|---|---|---|
| `banth` | `Text` | `Label` | separate 0/1 type columns | `label_binary = Label` | active type columns joined; if none active then `none` |
| `bd_shs` | `sentence` | `hate speech` | `type` | `label_binary = hate speech` | `label_type = type` |
| `facebook_44001` | `comment` | derived from `label` | `label` | `not bully -> 0`; other labels -> `1` | `label_type = label` |
| `multilabel_12557` | `comment` | `bully` | separate 0/1 type columns | `label_binary = bully` | active type columns joined; if none active then `none` |

---

## 7. Final 9-Class Abuse Taxonomy

Across the merged data, we found **89 unique raw `label_type` values**. Many were too rare to model properly, so we consolidated them into **9 cleaner classes**.

### 7.1 Why not keep all 89 classes?
Because:
- many classes were extremely small,
- some had fewer than 15 samples,
- training would be unstable,
- the final system would be harder to interpret and compare.

### 7.2 Final 9 classes

1. `none`
2. `abusive`
3. `personal`
4. `sexual`
5. `religious`
6. `threat`
7. `political`
8. `other`
9. `gender`

### 7.3 Mapping table

| Final class | Raw labels / patterns mapped into it |
|---|---|
| `none` | none, not bully |
| `abusive` | Abusive/Violence, troll |
| `personal` | Personal Offense, Body Shaming, Origin, slander, Misc |
| `sexual` | sexual, sexual,religious |
| `religious` | religious, Religious, religion, religion_slander |
| `threat` | threat, callToViolence*, religious,threat, sexual,threat |
| `political` | Political |
| `other` | spam |
| `gender` | gender, Gender, gender_slander |

### 7.4 Compound-label handling

Priority rule:

**threat > sexual > religious > gender > political > abusive > personal > other > none**

Example:
- `sexual,religious` → `sexual`
- `religious,threat` → `threat`
- `sexual,threat` → `threat`

### 7.5 Why this was necessary
Because the final abuse-type task is **multi-class**, not multi-label.  
The benchmark needed **one final class per text**.

### 7.6 Multi-label vs multi-class
- **multi-label** = one text may have several tags
- **multi-class** = one final class only

The final system uses **multi-class abuse-type classification**.

---

## 8. Duplicate Handling

### 8.1 What we found
There were **40,590 raw duplicates** across sources.

### 8.2 What we did
We **did not deduplicate** the final benchmark.

### 8.3 Why we kept duplicates
We kept them intentionally for realism and robustness-study relevance, because real online data and public datasets often overlap.

### 8.4 Honest limitation
A stricter fully deduplicated benchmark could be explored in future work.

---

## 9. Preprocessing

### 9.1 What preprocessing means
Cleaning and standardizing noisy social-media text before training.

### 9.2 Example
Raw text:
`এটা একটা টেস্ট @user123 https://example.com 😂😂😂`

After preprocessing:
- URL is masked
- user mention is masked
- emoji is normalized
- messy characters and spacing are cleaned

### 9.3 Preprocessing steps used

1. **Unicode normalization (NFKC)**  
   Makes text encoding more consistent.

2. **Remove invisible special Bangla-related characters**  
   Removes hidden formatting artifacts.

3. **Mask URLs**  
   Replace with `[URL]`

4. **Mask user mentions**  
   Replace with `[USER]`

5. **Normalize hashtags**  
   `#topic -> [HASHTAG] topic`

6. **Map emojis into text markers**  
   Uses an emoji library if available, otherwise regex fallback.

7. **Reduce repeated characters**  
   Example: `খারাপপপপপ` → normalized shorter form

8. **Normalize whitespace**  
   Cleans extra spaces/tabs/newlines

### 9.4 Final cleaned text field
After preprocessing, the notebook creates:
- `text_clean`

### 9.5 Why preprocessing was light, not aggressive
We used **light normalization suitable for mixed-script social media text**, not heavy rewriting, because abusive language signal can be hidden in:
- emoji,
- repetition,
- hashtags,
- mentions,
- expressive writing style.

### 9.6 Did preprocessing help?
Yes, slightly. In ablation:
- Binary Macro-F1 dropped from **0.9081** to **0.9067** without preprocessing
- Abuse-type Macro-F1 dropped from **0.7424** to **0.7349**

So preprocessing helped modestly.

---

## 10. Splits and Evaluation Settings

### 10.1 Random stratified split
- **80% train**
- **10% validation**
- **10% test**

Final sample counts:
- Train = 108,460
- Validation = 13,557
- Test = 13,558

### 10.2 Why stratified?
To preserve the binary harmful/not-harmful balance across splits.

### 10.3 Source-held-out split
Hold out one full source from training and test only on that source.

This checks:
> Can the model generalize to a completely unseen dataset source?

### 10.4 Script-held-out split
Hold out one script form and test only on that script.

The code looks for script names such as:
- romanized
- banglish
- roman
- mixed

Because the benchmark contains Romanized text, the script-held-out split holds out the Romanized side.

This checks:
> Can the model generalize to a different writing form?

### 10.5 Why both random split and held-out splits were necessary

Because each split answers a different question:

- **Random split:** can the model perform on unseen examples from the same overall benchmark?
- **Source-held-out:** can the model generalize to a completely unseen source?
- **Script-held-out:** can the model generalize to a different script?

**One evaluation is not enough for a benchmark paper.**

---

## 11. Model Methodology in Easy Words

### 11.1 Big picture
For each comment, the final system does **two predictions**:

1. **Binary classification**  
   harmful vs not harmful

2. **Abuse-type classification**  
   one of 9 classes

### 11.2 Input-to-output flow

comment text  
→ transformer reads it  
→ model creates one internal representation  
→ one head predicts binary  
→ one head predicts abuse type  
→ training compares both predictions with true labels  
→ model updates its weights  
→ test-time predictions are used to calculate final metrics

### 11.3 Why this is called multi-task
Because one shared model backbone is trained to do **two tasks at once**.

---

## 12. Transformer Models Used

### 12.1 Main backbones
- **BanglaBERT**
- **MuRIL**
- **XLM-R**

### 12.2 Why these models
- **BanglaBERT**: strongest language-specific Bangla option
- **MuRIL**: useful for multilingual / transliterated settings
- **XLM-R**: strong general multilingual baseline

### 12.3 Backbone details

| Model | Pretrain source | Architecture |
|---|---|---|
| BanglaBERT | BUET CSE NLP | ELECTRA |
| MuRIL | Google | BERT-base |
| XLM-R | Meta AI | RoBERTa-base |

---

## 13. Final Model Architecture

All three backbones use the same classifier architecture.

| Component | Final implemented setup |
|---|---|
| Encoder backbone | BanglaBERT / MuRIL / XLM-R |
| Pooling | 0.5 × CLS + 0.5 × mean pooling |
| Shared head design | Dropout(0.25) → Linear(hidden, 384) → GELU → LayerNorm(384) → Dropout(0.25) → Linear(384, n_classes) |
| Binary head | same design with 2 outputs |
| Abuse-type head | same design with 9 outputs |
| Mixed precision | used |
| token_type_ids | used only for BERT-family models, not XLM-R |

### 13.1 Why pooling was 0.5 × CLS + 0.5 × mean
Safe and honest explanation:

We used equal weighting as a simple balance between:
- **CLS** = global summary of the sentence
- **mean pooling** = information from all tokens

The final log supports the idea that this blend was better than using CLS only. We do **not** claim that 0.5/0.5 is a mathematically proven optimum over every other ratio.

---

## 14. How the Two Tasks Work

### 14.1 Binary classification
The binary head produces **2 scores**:
- score for **not harmful**
- score for **harmful**

The higher score decides the predicted class.

### 14.2 Abuse-type classification
The abuse-type head produces **9 scores**, one for each class:
- none
- abusive
- personal
- sexual
- religious
- threat
- political
- other
- gender

The highest score becomes the predicted abuse type.

### 14.3 How those 9 scores are produced
The model does not use hand-written rules like:
> if a bad word appears, set sexual = 0.8

Instead, it learns weights during training.

Simple flow:
1. transformer reads the comment
2. pooled representation is created
3. first linear layer maps it into a 384-dimensional hidden space
4. GELU + LayerNorm + Dropout reshape/stabilize features
5. final linear layer maps 384 hidden features to 9 class scores

Very simply, one class score is:
> weighted combination of hidden features + bias

Those weights are learned during training.

### 14.4 Why same architecture can still give different F1 by class
Same architecture does **not** mean same learning difficulty.

A class like `gender` had only **90** test samples and F1 **0.56**, while `none` had **7,830** samples and F1 **0.93**. Rare classes are harder because:
- the model sees fewer examples,
- they are easier to confuse with nearby classes,
- small support makes precision/recall less stable.

---

## 15. Losses, Training Tricks, and Why They Were Used

### 15.1 Easy definitions

| Term | Easy meaning |
|---|---|
| Focal loss | pays more attention to hard examples |
| Class weighting | gives more importance to rare classes |
| Mixed precision | trains faster and uses less GPU memory |
| Early stopping | stops training when validation stops improving |
| Multiple seeds | trains several times with different random starts to check stability |

### 15.2 Loss setup

| Task | Loss |
|---|---|
| Binary | FocalLoss(gamma = 1.5) + effective-sample class weights |
| Abuse type | FocalLoss(gamma = 2.5) + same class weights |

### 15.3 Why gamma differs
- **1.5** for binary = moderate focus on hard examples
- **2.5** for abuse type = stronger focus, because the 9-class task is harder

### 15.4 Important honesty note about focal loss
The ablation study showed focal loss was roughly neutral on this near-balanced dataset. So it was a reasonable choice, but not the main reason the system worked.

---

## 16. Training Hyperparameters

| Hyperparameter | Value |
|---|---:|
| max_length | 128 |
| batch_size | 16 |
| gradient_accumulation | 2 |
| effective batch size | 32 |
| epochs | 8 |
| encoder learning rate | 2e-5 |
| head learning rate | 8e-5 |
| layer-wise LR decay | 0.90 |
| label smoothing | 0.03 |
| dropout | 0.25 |
| head hidden dimension | 384 |
| class_weight_beta | 0.999 |
| focal gamma (binary) | 1.5 |
| focal gamma (abuse type) | 2.5 |
| early stopping patience | 3 |
| monitor | 0.7 × binary_F1 + 0.3 × abuse_F1 |
| num_workers | 0 |

### 16.1 Easy explanations of important constants

- **dropout = 0.25**  
  During training, 25% of units are randomly ignored so the classifier does not overfit too tightly.

- **hidden_dim = 384**  
  The size of the hidden classifier layer. Bigger = more capacity, smaller = simpler model. 384 was a practical middle-ground design choice.

- **max_length = 128**  
  Covers most social-media comments while keeping memory manageable.

- **batch_size = 16**  
  Fits the GPU memory.

- **gradient_accumulation = 2**  
  Simulates a larger batch (effective 32) without requiring more VRAM.

- **encoder_lr = 2e-5**  
  Small learning rate so pretrained knowledge is not destroyed too quickly.

- **head_lr = 8e-5**  
  Higher learning rate because the task heads are newly added.

- **lr_decay = 0.90**  
  Lower layers update more slowly than upper layers.

- **label_smoothing = 0.03**  
  Reduces overconfidence during training.

- **patience = 3**  
  Stops training if validation does not improve for 3 checks.

- **monitor = 0.7 × binary_F1 + 0.3 × abuse_F1**  
  Gives more emphasis to the main binary task while still tracking abuse-type quality.

### 16.2 Important ablation note on LR decay
Although LR decay was chosen for standard transfer-learning reasons, the ablation showed that **removing LR decay actually improved performance**. So this is a discussion point, not a final “best” claim.

---

## 17. Baselines

### 17.1 Baseline models
- TF-IDF + Logistic Regression
- TF-IDF + SVM
- TF-IDF + Random Forest
- BiLSTM

### 17.2 Why baselines matter
They provide a fair lower bar before using transformers.

### 17.3 Results

| Model | Macro-F1 | Accuracy | MCC | AUROC |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.8669 | 0.8691 | 0.7342 | 0.9429 |
| TF-IDF + SVM | 0.8877 | 0.8897 | 0.7760 | 0.9558 |
| TF-IDF + Random Forest | 0.9090 | 0.9099 | 0.8183 | 0.9718 |
| BiLSTM | 0.8914 | 0.8926 | 0.7828 | 0.9479 |

**Strongest baseline:** TF-IDF + Random Forest (**0.9090 Macro-F1**)

---

## 18. Transformer Fine-Tuning Results

### 18.1 Seeds used
- 42
- 123
- 456

### 18.2 Total runs
3 models × 3 seeds = **9 runs**

### 18.3 Averaged transformer results

| Model | Binary Macro-F1 | Abuse-Type Macro-F1 | Binary MCC | Abuse-Type MCC |
|---|---|---|---|---|
| BanglaBERT | 0.9071 ± 0.0030 | 0.7407 ± 0.0025 | 0.8146 ± 0.0060 | 0.7529 ± 0.0045 |
| MuRIL | 0.9058 ± 0.0009 | 0.7303 ± 0.0042 | 0.8117 ± 0.0017 | 0.7480 ± 0.0021 |
| XLM-R | 0.8960 ± 0.0032 | 0.7114 ± 0.0041 | 0.7922 ± 0.0061 | 0.7304 ± 0.0058 |

### 18.4 Interpretation
- BanglaBERT performed best on average
- MuRIL was close
- XLM-R was slightly behind
- low std across seeds indicates stable training

---

## 19. Ensemble and Threshold Tuning

### 19.1 How the ensemble was built

| Step | What was done |
|---|---|
| 1 | Fine-tune 3 transformer backbones |
| 2 | Train each with 3 seeds |
| 3 | Save validation and test logits from all 9 runs |
| 4 | Optimize ensemble weights on validation set using Nelder–Mead |
| 5 | Combine logits using weighted averaging |
| 6 | Tune binary threshold |
| 7 | Evaluate final ensemble on test set |

### 19.2 What is a logit?
A logit is the model's **raw score before it turns into a probability**.

Example:
- harmful: 2.3
- not harmful: -1.1

These are logits, not probabilities.

### 19.3 How ensemble weights were chosen
They were **not manually chosen**.

We used the validation-set logits from all 9 models and optimized the weights using **Nelder–Mead**. In simple words, this method keeps trying different weight combinations and keeps the ones that improve validation performance.

So a weight like 0.2817 for `banglabert_seed42` means:
> in the best validation-set mixture, that model deserved about 28.17% of the weighted signal.

Larger weight = model contributed more useful signal  
Smaller weight = model contributed less useful signal

### 19.4 Final optimized weights

| Model-Seed | Weight |
|---|---:|
| banglabert_seed42 | 0.2817 |
| muril_seed42 | 0.1680 |
| xlmr_seed456 | 0.1625 |
| muril_seed123 | 0.1187 |
| banglabert_seed456 | 0.1163 |
| muril_seed456 | 0.0664 |
| banglabert_seed123 | 0.0422 |
| xlmr_seed42 | 0.0336 |
| xlmr_seed123 | 0.0107 |

### 19.5 Threshold tuning
The best binary threshold stayed at **0.50**.  
This makes sense because the binary dataset is near-balanced.

---

## 20. Final Ensemble Results

### 20.1 Binary results

| Metric | Value |
|---|---:|
| Accuracy | 0.9256 |
| Macro-F1 | 0.9247 |
| Weighted-F1 | 0.9256 |
| MCC | 0.8494 |
| AUROC | 0.9731 |
| AUPRC | 0.9658 |

### 20.2 Abuse-type results

| Metric | Value |
|---|---:|
| Macro-F1 | 0.7746 |
| Accuracy | 0.8688 |

### 20.3 Per-class abuse-type report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| none | 0.93 | 0.93 | 0.93 | 7,830 |
| religious | 0.89 | 0.86 | 0.87 | 908 |
| sexual | 0.79 | 0.81 | 0.80 | 1,083 |
| personal | 0.79 | 0.78 | 0.78 | 1,344 |
| other | 0.74 | 0.80 | 0.77 | 79 |
| abusive | 0.76 | 0.75 | 0.76 | 1,600 |
| political | 0.75 | 0.77 | 0.76 | 259 |
| threat | 0.73 | 0.77 | 0.75 | 365 |
| gender | 0.59 | 0.53 | 0.56 | 90 |

### 20.4 Main takeaways
- Ensemble improved binary F1 from best single **0.9095** to **0.9247**
- Ensemble improved abuse-type F1 from best single about **0.74** to **0.7746**
- `gender` was the weakest class because of very small support
- `none` was the strongest class

---

## 21. Why Macro-F1 Was the Main Metric

### Easy explanation
Macro-F1 computes F1 **for each class separately** and then averages them equally.

### Why this matters
It prevents large classes from hiding poor performance on small classes.

That is especially important in the 9-class abuse-type task, where some classes are much smaller than others.

---

## 22. Robustness Evaluation

### 22.1 Binary robustness results

| Split | N | Macro-F1 | Weighted-F1 | Accuracy | MCC | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| random_test (in-domain) | 13,558 | 0.9245 | 0.9254 | 0.9254 | 0.8489 | 0.9738 |
| source_holdout_banth | 73,999 | 0.9777 | 0.9819 | 0.9819 | 0.9554 | 0.9959 |
| source_holdout_bd_shs | 5,029 | 0.9342 | 0.9341 | 0.9342 | 0.8744 | 0.9926 |
| source_holdout_facebook_44001 | 44,001 | 0.9736 | 0.9761 | 0.9761 | 0.9473 | 0.9952 |
| source_holdout_multilabel_12557 | 12,546 | 0.9304 | 0.9358 | 0.9357 | 0.8608 | 0.9799 |
| script_holdout_romanized | 73,999 | 0.9777 | 0.9819 | 0.9819 | 0.9554 | 0.9959 |

### 22.2 Key observations
1. All holdout splits stayed at **F1 ≥ 0.93**
2. `source_holdout_banth` and `script_holdout_romanized` are identical because `banth` is entirely Romanized
3. Some holdout scores are higher than in-domain random test, likely because those sources are cleaner or more separable
4. Hardest split: `multilabel_12557`

### 22.3 Important honesty note
`source_holdout_banth` and `script_holdout_romanized` should **not** be presented as fully independent evidence.

---

## 23. Ablation Study

### 23.1 Results

| Ablation | Binary Macro-F1 | Δ Binary | Abuse-Type Macro-F1 | Δ Abuse |
|---|---:|---:|---:|---:|
| full_multitask | 0.9081 | — | 0.7424 | — |
| binary_only | 0.9138 | +0.0057 | N/A | — |
| no_focal | 0.9109 | +0.0028 | 0.7444 | +0.0020 |
| no_lrdecay | 0.9266 | +0.0185 | 0.7826 | +0.0402 |
| no_preprocessing | 0.9067 | -0.0014 | 0.7349 | -0.0075 |

### 23.2 Main lessons
- **binary_only** helps binary slightly, but loses the 9-class task
- **no_focal** is almost neutral
- **no_lrdecay** is the surprising result: removing LR decay improved both tasks
- **no_preprocessing** hurts slightly

### 23.3 Seminar interpretation
The multi-task setup gives richer output with only a small binary trade-off.

---

## 24. Error Analysis

### 24.1 Overall
- Total FP = 537
- Total FN = 472
- Error rate = 7.4%

### 24.2 Error rate by source
- banth = 5.68%
- bd_shs = 14.45%
- facebook_44001 = 8.26%
- multilabel_12557 = 12.48%

### 24.3 Error rate by script
- Bangla = 9.57%
- Romanized = 5.68%

### 24.4 Main interpretation
- Hardest sources: `bd_shs` and `multilabel_12557`
- Romanized was easier than Bangla in this benchmark

---

## 25. Key Design Decisions

| Decision | Rationale |
|---|---|
| Multi-class (not multi-label) for abuse_type | Compound labels are about 11% of data; multi-class is simpler and more comparable |
| 9 classes (not 89) | 89 classes were too sparse and imbalanced |
| Macro-F1 as primary metric | Gives equal importance to all classes |
| Default threshold 0.50 | Tuning found no benefit from moving it |
| Non-uniform ensemble weights | Validation-set optimization showed some models contributed more than others |
| 3 seeds per model | Enough stability for submission-time experiments |
| CLS + mean pooling | Balanced global summary + token-level detail |
| Focal loss | Reasonable for imbalance, though ablation shows only small impact here |
| Duplicates kept | Realism for robustness study |

---

## 26. Practical Viva / Seminar Explanations

### 26.1 Why is the benchmark multi-source?
Because it merges **4 public datasets** instead of relying on one source.

### 26.2 Why is it multi-script?
Because it includes both **Bangla script** and **Romanized Bangla**.

### 26.3 Why did we merge labels into 9 classes?
Because 89 raw classes were too fragmented and many had too few examples.

### 26.4 Why did we keep duplicates?
For realism and overlap across public sources, though a deduplicated version is a possible future extension.

### 26.5 Why 0.5 × CLS + 0.5 × mean?
Because it balances global sentence summary and all-token information. It was a practical empirical blend, not a proven unique optimum.

### 26.6 Why does `gender` have lower F1?
Because the model had far fewer examples for that class, so it had fewer chances to learn its patterns and more confusion with nearby classes.

### 26.7 What is a logit?
The raw model score before probability conversion.

### 26.8 Why are ensemble weights different?
They were optimized on validation data using Nelder–Mead. Bigger weight means that model helped the ensemble more.

---

## 27. Output Directory Structure

```text
../outputs/
├── models_v2_fix/
│   ├── label_encoders.json
│   ├── transformer_results_all.csv
│   ├── transformer_results_averaged.csv
│   ├── banglabert_seed42/
│   │   ├── best_model.pt
│   │   ├── val_logits.pt
│   │   ├── test_logits.pt
│   │   ├── results.json
│   │   └── label_encoders.json
│   ├── banglabert_seed{123,456}/
│   ├── muril_seed{42,123,456}/
│   └── xlmr_seed{42,123,456}/
├── ensemble/
│   ├── final_config.json
│   ├── ensemble_test_metrics.json
│   ├── test_preds.npy
│   ├── test_probs.npy
│   ├── threshold_tuning.png
│   └── cm_ensemble_test.png
├── robustness/
│   └── robustness_results.csv
├── ablations/
│   └── ablation_results.csv
├── baselines/
│   └── baseline_results.csv
└── paper/
    ├── results_summary.json
    ├── table1_main_results.csv
    ├── table1b_multitask.csv
    ├── table2_ablations.csv
    ├── table3_robustness.csv
    ├── table1.tex
    ├── table2.tex
    ├── table3.tex
    ├── fig_main_results.png
    ├── figures/
    └── tables/
```

---

## 28. Bugs Fixed in v3 Notebooks

| Bug | Affected | Fix |
|---|---|---|
| Python 3.13 install crash | NB05 | tokenizers≥0.19.0, transformers≥4.44.0 |
| num_workers>0 deadlock on Windows Jupyter | NB05, NB09 | num_workers=0 |
| `sexual,religious` wrongly mapped | NB05/06/08/09 | fixed priority: sexual > religious |
| abuse_type F1 collapsed with 89 classes | NB05 | 9-class consolidation + beta clamp |
| Only 1 model/seed ran | NB05 | all 3 models × 3 seeds enabled |
| NB06 label mismatch | NB06 | load shared label_encoders.json |
| NB09 binary labels all = -1 | NB09 | fix encoder lookup |
| NB09 model architecture mismatch | NB09 | align with NB05 |
| NB08 rebuilt label encoders independently | NB08 | load label_encoders.json from NB05 |
| deprecated GradScaler API | NB05, NB09 | use `torch.amp.GradScaler('cuda')` |
| XLM-R received token_type_ids | NB05/08/09 | skip token_type_ids for XLM-R |
| NB06 missing ensemble metrics json | NB06 | add save step |
| NB07 KeyError for `macro_f1` | NB07 | correct column name |

---

## 29. Limitations and Honest Claims

1. The final implemented system is **2-task**, not 3-task.  
   It predicts **binary harmfulness** and **9-class abuse type**. Do not claim the final reported system predicts severity.

2. `source_holdout_banth` and `script_holdout_romanized` overlap strongly and should not be treated as fully independent evidence.

3. Duplicates were intentionally kept. This is defendable, but a deduplicated benchmark remains future work.

4. The 0.5/0.5 pooling blend was a balanced empirical choice, not a proven global optimum.

5. The ablation suggests uniform LR may be better than the reported LR-decay setup, which is an important discussion point.

---

## 30. Critical Viva / Seminar Questions and Straight Answers

### Q1. Why is your benchmark called multi-source?
Because it merges 4 different public datasets instead of using only one.

### Q2. Why is it multi-script?
Because it contains both Bangla script and Romanized Bangla.

### Q3. Why did you convert raw labels to 9 classes?
Because 89 classes were too sparse and unstable for reliable training.

### Q4. Why not keep multi-label abuse type?
Because compound labels were only about 11% of the data, and a multi-class setup was simpler and easier to compare with prior work.

### Q5. Why did you keep duplicates?
For realism and overlap across public sources, but we acknowledge deduplicated benchmarking as future work.

### Q6. Why do you need source-held-out and script-held-out splits?
Because random split alone cannot prove the model generalizes to unseen sources or unseen writing forms.

### Q7. Why are `source_holdout_banth` and `script_holdout_romanized` the same?
Because `banth` is entirely Romanized, so those evaluations use the same data.

### Q8. Why use a multi-task setup?
It gives both yes/no harmfulness and 9-class abuse type. Binary-only improves slightly, but loses richer output.

### Q9. Why is BanglaBERT better?
Because it has language-specific pretraining for Bangla.

### Q10. Why is `gender` weaker?
Because it has very low support, so the model has fewer examples to learn from.

### Q11. Why Macro-F1 instead of only accuracy?
Because Macro-F1 gives equal importance to every class, including small ones.

### Q12. Why 0.5 × CLS + 0.5 × mean?
Because it balances global sentence summary and all-token information. It was a practical empirical blend, not a proven unique optimum.

### Q13. Why are ensemble weights not equal?
Because validation-set optimization found some models more useful than others.

### Q14. What is a logit?
A raw model score before converting to probability.

### Q15. Why was threshold 0.50 best?
Because the binary dataset is near-balanced and tuning found no gain from shifting it.

### Q16. Why did no_lrdecay perform better?
Likely because this large multi-script dataset benefited from more uniform and aggressive fine-tuning than originally expected.

### Q17. Did preprocessing matter?
Yes, but modestly. Removing preprocessing reduced both binary and abuse-type Macro-F1 slightly.

### Q18. What should you never overclaim in seminar?
Do not claim:
- a final severity model,
- fully independent evidence from banth/script holdout,
- exhaustive proof that 0.5/0.5 pooling is optimal,
- that duplicates cannot affect scores.

---

## 31. Final Handoff Checklist

| Item | Status |
|---|---|
| All notebooks (NB04–NB10) | ✅ |
| All model checkpoints | ✅ |
| All logits | ✅ |
| Ensemble config + predictions | ✅ |
| Ablation CSV | ✅ |
| Robustness CSV | ✅ |
| Publication figures | ✅ |
| LaTeX tables | ✅ |
| Consolidated results JSON | ✅ |
| This master experiment log | ✅ |

---

*BanglaCyberBench Experiment Log — Masterclass Updated | April 2026*
