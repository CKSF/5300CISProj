# Simple Baselines (`simple-baseline.py`)

This script runs all three non-transformer baselines for predicting whether an Amazon review received zero helpful votes (class 0) or at least three helpful votes (class 1). It combines the former simple and strong baseline scripts into one entry point.

- **All-zero baseline.** Predict class 0 for every review and assign a score of 0.0.
- **Word-count baseline.** Count whitespace-separated words in the review title and body. Predict class 1 when the count is at least the threshold. We select the threshold that maximizes accuracy on the 2015–2018 Electronics development set, breaking ties in favor of the smallest threshold. We use the same threshold on all five test sets and use the raw word count as the ROC-AUC score.
- **TF-IDF + metadata logistic regression.** Combine sublinearly scaled unigram/bigram TF-IDF features (`max_features=50000`, `min_df=2`, Unicode accent stripping) from the review title and body with standardized metadata: `rating`, binary `verified_purchase`, and `log(1 + word_count)`. Fit the vectorizer, scaler, and classifiers on the Electronics 2015–2018 training split (24,000 reviews). Select the regularization parameter `C` by accuracy on the corresponding development split (3,000 reviews), retaining the first candidate on ties. The score is the probability of class 1, and the predicted label is 1 when that probability is at least 0.5.

In the previously recorded runs, the selected word-count threshold was **43 words**, with development accuracy of **0.7743**. Logistic regression selected **C = 0.5**, with development accuracy **0.7913** and ROC-AUC **0.8659**. These settings are selected again on each run.

The RoBERTa strong baseline has its own script, `strong-baseline.py`, and usage guide: [strong-baseline.md](strong-baseline.md).

## Dependencies

```bash
python -m pip install numpy scipy scikit-learn
```

All three baselines run on CPU.

## Running the baselines

From the repository root, run:

```bash
python milestone2/simple-baseline.py
```

You can also run `python simple-baseline.py` from the `milestone2` directory. Default paths are resolved relative to the script.

The script reads `data/electronics_2015_2018/train.jsonl` and `dev.jsonl`, tunes the word-count threshold and logistic regression, then generates predictions for all five test sets. Test labels are not used to generate predictions.

### Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--data-dir` | Repository's `data/` directory | Root containing the split directories |
| `--output-dir` | `milestone2/predictions/` | Prediction output directory for all three baselines |
| `--max-features` | `50000` | Maximum TF-IDF vocabulary size |
| `--c-grid` | `0.1 0.5 1.0 2.0 5.0` | Logistic regression C candidates, selected by development accuracy |

For example, run all baselines with a smaller vocabulary and a different C grid:

```bash
python milestone2/simple-baseline.py \
  --data-dir data \
  --output-dir milestone2/predictions_rerun \
  --max-features 20000 \
  --c-grid 0.1 0.5 1.0
```

Explicit relative paths are resolved from the current working directory. `--max-features` and `--c-grid` affect only logistic regression.

## Prediction files

The script writes one JSONL prediction file per baseline and test set (15 files total) to `milestone2/predictions/`. Existing files with the same names are replaced. Each baseline uses its own output prefix:

| Baseline | Filename pattern |
|---|---|
| All-zero | `all_zero.<split>.jsonl` |
| Word-count | `word_count.<split>.jsonl` |
| TF-IDF + metadata logistic regression | `tf-idf.<split>.jsonl` |

The five split names are `electronics_2015_2018`, `shift_electronics_2021`, `shift_books_2015_2018`, `shift_home_kitchen_2015_2018`, and `shift_beauty_2015_2018`.

Each prediction contains the zero-based `row_id`, binary `prediction`, and numeric `score`. The review's `id` is copied when present. Logistic regression probabilities are rounded to six decimal places; word-count scores are raw integer counts.

## Evaluating predictions

For example, score the word-count baseline on the 2015–2018 Electronics test set:

```bash
python milestone2/score.py \
  --predictions milestone2/predictions/word_count.electronics_2015_2018.jsonl \
  --gold data/electronics_2015_2018/test.jsonl
```

To score logistic regression on the same test set:

```bash
python milestone2/score.py \
  --predictions milestone2/predictions/tf-idf.electronics_2015_2018.jsonl \
  --gold data/electronics_2015_2018/test.jsonl
```

Use the corresponding prefix and split name to evaluate the other outputs. See [scoring.md](scoring.md) for metric definitions.

## Example prediction

An all-zero prediction for the first review in a test file:

```json
{"row_id": 0, "prediction": 0, "score": 0.0}
```

## Test results

Previously recorded test results are preserved below; each cell lists accuracy / macro-F1 / ROC-AUC.

| Test set | All-zero | Word-count | TF-IDF + metadata LR |
|---|---|---|---|
| Electronics 2015–2018 | 0.5000 / 0.3333 / 0.5000 | 0.7647 / 0.7647 / 0.8435 | 0.7947 / 0.7947 / 0.8681 |
| Electronics 2021 | 0.5000 / 0.3333 / 0.5000 | 0.7404 / 0.7403 / 0.8129 | 0.7472 / 0.7470 / 0.8257 |
| Books 2015–2018 | 0.5000 / 0.3333 / 0.5000 | 0.7638 / 0.7636 / 0.8244 | 0.7718 / 0.7718 / 0.8405 |
| Home & Kitchen 2015–2018 | 0.5000 / 0.3333 / 0.5000 | 0.7492 / 0.7485 / 0.8232 | 0.7592 / 0.7578 / 0.8406 |
| Beauty & Personal Care 2015–2018 | 0.5000 / 0.3333 / 0.5000 | 0.7438 / 0.7429 / 0.8207 | 0.7460 / 0.7443 / 0.8327 |
