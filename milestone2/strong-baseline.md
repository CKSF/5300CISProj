# Strong Baseline (`strong-baseline.py`)

## Overview
`strong-baseline.py` implements an $L_2$-regularized Logistic Regression classifier combining TF-IDF n-gram features and standardized metadata. It trains on the in-distribution `electronics_2015_2018` training split (`../data/electronics_2015_2018/train.jsonl`, 24,000 reviews) and tunes the inverse regularization strength $C \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$ on the development split (`../data/electronics_2015_2018/dev.jsonl`, 3,000 reviews), selecting $C = 0.5$ (dev accuracy `0.7913`, ROC-AUC `0.8659`).

### Feature Representation
1. **Textual Features (`TfidfVectorizer`)**: Concatenates review `title` and `text`, extracting unigrams and bigrams (`ngram_range=(1, 2)`, `min_df=2`, `max_features=50000`, `sublinear_tf=True`, `strip_accents="unicode"`).
2. **Metadata Features (`StandardScaler`)**: Extracts numerical star `rating`, binary `verified_purchase` indicator ($0$ or $1$), and log-transformed total word count $\log(1 + \text{word\_count}(\text{title} + \text{text}))$, standardized using training-split statistics.

## Dependencies
```bash
pip install numpy scipy scikit-learn
```

## How to Run
From the `milestone2` directory (where `score.py` is located), run:
```bash
python strong-baseline.py
```

### Example Training & Inference Output
```text
Dev (C=0.1): accuracy=0.7857, macro_f1=0.7856, roc_auc=0.8600
Dev (C=0.5): accuracy=0.7913, macro_f1=0.7913, roc_auc=0.8659
Dev (C=1.0): accuracy=0.7910, macro_f1=0.7910, roc_auc=0.8658
Dev (C=2.0): accuracy=0.7907, macro_f1=0.7907, roc_auc=0.8634
Dev (C=5.0): accuracy=0.7827, macro_f1=0.7827, roc_auc=0.8555
Selected C=0.5 (dev accuracy: 0.7913)
electronics_2015_2018: wrote 3000 predictions to strong.electronics_2015_2018.jsonl
shift_electronics_2021: wrote 5000 predictions to strong.shift_electronics_2021.jsonl
shift_books_2015_2018: wrote 5000 predictions to strong.shift_books_2015_2018.jsonl
shift_home_kitchen_2015_2018: wrote 5000 predictions to strong.shift_home_kitchen_2015_2018.jsonl
shift_beauty_2015_2018: wrote 5000 predictions to strong.shift_beauty_2015_2018.jsonl
```

## Evaluating Predictions with `score.py`
Run `score.py` on any generated file in `predictions/`:
```bash
python score.py --predictions predictions/strong.electronics_2015_2018.jsonl --gold ../data/electronics_2015_2018/test.jsonl
```

### Example Evaluation Output Across Splits
```text
# electronics_2015_2018 (In-Distribution)
examples: 3000
accuracy: 0.7947
macro_f1: 0.7947
roc_auc:  0.8681

# shift_electronics_2021 (Time Shift)
examples: 5000
accuracy: 0.7472
macro_f1: 0.7470
roc_auc:  0.8257

# shift_books_2015_2018 (Category Shift)
examples: 5000
accuracy: 0.7718
macro_f1: 0.7718
roc_auc:  0.8405

# shift_home_kitchen_2015_2018 (Category Shift)
examples: 5000
accuracy: 0.7592
macro_f1: 0.7578
roc_auc:  0.8406

# shift_beauty_2015_2018 (Category Shift)
examples: 5000
accuracy: 0.7460
macro_f1: 0.7443
roc_auc:  0.8327
```