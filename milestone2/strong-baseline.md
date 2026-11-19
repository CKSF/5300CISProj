# Strong Baseline (`strong-baseline.py`)

## Overview

The strong baseline fine-tunes `roberta-base` to predict whether an Amazon review
received zero helpful votes (class 0) or at least three helpful votes (class 1).
It trains on the Electronics 2015–2018 training split and selects the model
with the highest accuracy on the corresponding development split. The selected
model then generates predictions for all five test sets.

The input is the review title and body, joined as `title </s> text`. If the
title is empty, the script uses only the body. Inputs are truncated or padded
to 256 tokens by default. The score is the model's probability of class 1;
the predicted label is 1 when that probability is at least 0.5.

## Dependencies

Install the required packages in your Python environment:

```bash
python -m pip install numpy scikit-learn torch transformers
```

The first run downloads the pretrained tokenizer and model from Hugging Face
unless they are already cached. The script automatically selects CUDA if
available, otherwise Apple MPS, otherwise CPU. GPU acceleration is recommended
for training; mixed precision is enabled on CUDA.

## Running the baseline

From the repository root, run:

```bash
python milestone2/strong-baseline.py
```

By default, the script reads `data/electronics_2015_2018/train.jsonl` and
`data/electronics_2015_2018/dev.jsonl`, followed by `test.jsonl` from each of
the five split directories listed below. Training and development records
should contain a binary `label`; review inputs use `title` and `text`.
Test labels are not used when generating predictions.

Default data and output paths are resolved relative to the script, so you
can also run `python strong-baseline.py` from the `milestone2` directory.

### Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--data-dir` | Repository's `data/` directory | Directory containing the dataset split directories |
| `--output-dir` | `milestone2/predictions/` | Directory for prediction files |
| `--model-name` | `roberta-base` | Hugging Face model name or local pretrained model directory |
| `--max-length` | `256` | Maximum tokenized input length, including special tokens |
| `--batch-size` | `32` | Training batch size; development and test batches use twice this size |
| `--epochs` | `3` | Number of training epochs; use a positive integer |
| `--lr` | `2e-5` | AdamW learning rate |
| `--seed` | `42` | Random seed for PyTorch and NumPy |

For example, use a smaller batch size if the default exceeds available
device memory, and write results to a separate directory:

```bash
python milestone2/strong-baseline.py \
  --batch-size 8 \
  --epochs 3 \
  --output-dir milestone2/predictions_roberta
```

Explicit relative paths passed to `--data-dir` or `--output-dir` are relative
to your current working directory.

Training uses AdamW with weight decay 0.01 and a linear learning-rate schedule
with warmup over the first 10% of training steps. Each epoch prints average
training loss, development accuracy, macro-F1, and ROC-AUC. The script restores
the epoch with the best development accuracy before generating predictions;
ties retain the earlier epoch. Model weights are kept in memory and are not
saved to disk, so each invocation trains again.

## Prediction files

The default run writes these files to `milestone2/predictions/`, replacing
existing files with the same names:

| Test set | Prediction file |
|---|---|
| Electronics 2015–2018 | `roberta.electronics_2015_2018.jsonl` |
| Electronics 2021 | `roberta.shift_electronics_2021.jsonl` |
| Books 2015–2018 | `roberta.shift_books_2015_2018.jsonl` |
| Home & Kitchen 2015–2018 | `roberta.shift_home_kitchen_2015_2018.jsonl` |
| Beauty & Personal Care 2015–2018 | `roberta.shift_beauty_2015_2018.jsonl` |

Each line corresponds to a review in the same order as its test file.
`row_id` is the zero-based row index, `prediction` is the binary label, and
`score` is the class-1 probability rounded to six decimal places. The review's
`id` is also copied when present. An illustrative prediction is:

```json
{"row_id": 0, "prediction": 1, "score": 0.812345}
```

## Evaluating predictions

From the repository root, score the in-distribution test predictions with:

```bash
python milestone2/score.py \
  --predictions milestone2/predictions/roberta.electronics_2015_2018.jsonl \
  --gold data/electronics_2015_2018/test.jsonl
```

For another test set, replace the split name in both paths. If you used a
custom output directory, update the prediction path accordingly. The scorer
reports accuracy, macro-F1, and ROC-AUC; add `--json` for JSON output.
See [scoring.md](scoring.md) for metric definitions and prediction validation.

## Previously recorded results

The earlier combined strong-baseline guide reported the following development
metrics. The best epoch was epoch 3, with accuracy 0.8037.

| Epoch | Training loss | Dev accuracy | Dev macro-F1 | Dev ROC-AUC |
|---|---|---|---|---|
| 1 | 0.4940 | 0.7977 | 0.7975 | 0.8710 |
| 2 | 0.4379 | 0.8000 | 0.7999 | 0.8746 |
| 3 | 0.3960 | 0.8037 | 0.8037 | 0.8756 |

| Test set | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|
| Electronics 2015–2018 | 0.7997 | 0.7996 | 0.8737 |
| Electronics 2021 | 0.7568 | 0.7568 | 0.8305 |
| Books 2015–2018 | 0.7556 | 0.7541 | 0.8343 |
| Home & Kitchen 2015–2018 | 0.7620 | 0.7616 | 0.8386 |
| Beauty & Personal Care 2015–2018 | 0.7356 | 0.7332 | 0.8202 |

For all-zero, word-count, and TF-IDF + metadata logistic regression usage and
results, see [simple-baseline.md](simple-baseline.md).
