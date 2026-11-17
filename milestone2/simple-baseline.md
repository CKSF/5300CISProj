# Simple Baselines

We compare two baselines for predicting whether an Amazon review received zero helpful votes (class 0) or at least three helpful votes (class 1).

- **All-zero baseline.** Predict class 0 for every review and assign a score of 0.0.
- **Word-count baseline.** Count whitespace-separated words in the review title and body. Predict class 1 when the count is at least the threshold. We select the threshold that maximizes accuracy on the 2015–2018 Electronics development set, breaking ties in favor of the smallest threshold. We use the same threshold on all five test sets and use the raw word count as the ROC-AUC score.

The selected threshold was **43 words**, with development accuracy of **0.7743**.

## Running the baselines

From the repository root, run:

```bash
python milestone2/simple-baseline.py
```

The script writes one JSONL prediction file per baseline and test set to `milestone2/predictions/`. For example, to score the word-count baseline on the 2015–2018 Electronics test set, run:

```bash
python milestone2/score.py \
  --predictions milestone2/predictions/word_count.electronics_2015_2018.jsonl \
  --gold data/electronics_2015_2018/test.jsonl
```

## Example prediction

An all-zero prediction for the first review in a test file:

```json
{"row_id": 0, "prediction": 0, "score": 0.0}
```

## Test results

Each cell lists accuracy / macro-F1 / ROC-AUC.

| Test set | All-zero | Word-count |
|---|---|---|
| Electronics 2015–2018 | 0.5000 / 0.3333 / 0.5000 | 0.7647 / 0.7647 / 0.8435 |
| Electronics 2021 | 0.5000 / 0.3333 / 0.5000 | 0.7404 / 0.7403 / 0.8129 |
| Books 2015–2018 | 0.5000 / 0.3333 / 0.5000 | 0.7638 / 0.7636 / 0.8244 |
| Home & Kitchen 2015–2018 | 0.5000 / 0.3333 / 0.5000 | 0.7492 / 0.7485 / 0.8232 |
| Beauty & Personal Care 2015–2018 | 0.5000 / 0.3333 / 0.5000 | 0.7438 / 0.7429 / 0.8207 |
