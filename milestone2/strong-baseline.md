# Strong Baselines (`strong-baseline.py` & `roberta-baseline.py`)

## Overview
We provide two complementary strong baselines trained on the in-distribution **Electronics (2015–2018)** training split (`../data/electronics_2015_2018/train.jsonl`, 24,000 reviews) and validated on the development split (`../data/electronics_2015_2018/dev.jsonl`, 3,000 reviews):

1. **TF-IDF + Metadata Logistic Regression (`strong-baseline.py`)**: Combines sublinearly scaled unigram/bigram TF-IDF features (`max_features=50000`, `ngram_range=(1, 2)`, `min_df=2`) with standardized review metadata (`rating`, binary `verified_purchase`, and $\log(1 + \text{word\_count})$). The regularization parameter $C \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$ is tuned on dev, selecting $C = 0.5$ (dev accuracy `0.7913`, ROC-AUC `0.8659`).
2. **Fine-tuned RoBERTa (`roberta-baseline.py`)**: Fine-tunes `roberta-base` with a binary sequence classification head over concatenated review title and body (`title </s> text`, `max_length=256`) using AdamW ($\text{lr}=2\times 10^{-5}$, weight decay $0.01$, linear warmup over 10% of steps, batch size $32$, 3 epochs). Best dev accuracy reaches `0.8037` (ROC-AUC `0.8756`) at Epoch 3.

## Dependencies
```bash
pip install numpy scipy scikit-learn torch transformers
```

## How to Run

### 1. Run TF-IDF + Metadata Logistic Regression
From the `milestone2` directory:
```bash
python strong-baseline.py
```
Outputs `predictions/strong.<split>.jsonl` for all 5 test splits.

### 2. Run Fine-tuned RoBERTa
From the `milestone2` directory (GPU recommended):
```bash
python roberta-baseline.py
```
Outputs `predictions/roberta.<split>.jsonl` for all 5 test splits.

#### Example `roberta-baseline.py` Training Output
```text
Epoch 1/3 | loss=0.4940 | dev_acc=0.7977, dev_f1=0.7975, dev_auc=0.8710
Epoch 2/3 | loss=0.4379 | dev_acc=0.8000, dev_f1=0.7999, dev_auc=0.8746
Epoch 3/3 | loss=0.3960 | dev_acc=0.8037, dev_f1=0.8037, dev_auc=0.8756
Restoring best model (dev accuracy: 0.8037)
electronics_2015_2018: wrote 3000 predictions to roberta.electronics_2015_2018.jsonl
shift_electronics_2021: wrote 5000 predictions to roberta.shift_electronics_2021.jsonl
shift_books_2015_2018: wrote 5000 predictions to roberta.shift_books_2015_2018.jsonl
shift_home_kitchen_2015_2018: wrote 5000 predictions to roberta.shift_home_kitchen_2015_2018.jsonl
shift_beauty_2015_2018: wrote 5000 predictions to roberta.shift_beauty_2015_2018.jsonl
```

## Evaluating Predictions with `score.py`
```bash
# Evaluate TF-IDF + Metadata LR
python score.py --predictions predictions/strong.electronics_2015_2018.jsonl --gold ../data/electronics_2015_2018/test.jsonl

# Evaluate Fine-tuned RoBERTa
python score.py --predictions predictions/roberta.electronics_2015_2018.jsonl --gold ../data/electronics_2015_2018/test.jsonl
```

### Summary of Test Split Results

| Split | TF-IDF + Meta LR (`strong.*`) Acc / F1 / AUC | Fine-tuned RoBERTa (`roberta.*`) Acc / F1 / AUC |
| :--- | :---: | :---: |
| `electronics_2015_2018` | 0.7947 / 0.7947 / 0.8681 | 0.7997 / 0.7996 / 0.8737 |
| `shift_electronics_2021` | 0.7472 / 0.7470 / 0.8257 | 0.7568 / 0.7568 / 0.8305 |
| `shift_books_2015_2018` | 0.7718 / 0.7718 / 0.8405 | 0.7556 / 0.7541 / 0.8343 |
| `shift_home_kitchen_2015_2018` | 0.7592 / 0.7578 / 0.8406 | 0.7620 / 0.7616 / 0.8386 |
| `shift_beauty_2015_2018` | 0.7460 / 0.7443 / 0.8327 | 0.7356 / 0.7332 / 0.8202 |