# Scoring

Our task is binary classification. Given a review's title and text, a system
predicts whether the review received at least 3 helpful votes (class 1) or
0 helpful votes (class 0). `score.py` compares a system's predictions with the
gold labels and reports three metrics. **Accuracy is our primary metric**;
macro-F1 and ROC-AUC are reported alongside it.

## Metrics

Let $N$ be the number of test reviews, $y_i \in \{0, 1\}$ the gold label of
review $i$, $\hat{y}_i \in \{0, 1\}$ the predicted label, and $s_i$ the
real-valued score the system assigns to review $i$ (larger means more likely
class 1).

### Accuracy (primary)

The fraction of reviews whose predicted label equals the gold label:

$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]$$

Every test set is sampled to be exactly 50% class 1 and 50% class 0, so a
system that ignores the input scores 0.5. This is why accuracy is a fair
headline number here and why it can be compared across our test sets.

### Macro-averaged F1

For a class $c$, let $TP_c$ be the number of reviews with $y_i = c$ and
$\hat{y}_i = c$, $FP_c$ the number with $y_i \neq c$ and $\hat{y}_i = c$, and
$FN_c$ the number with $y_i = c$ and $\hat{y}_i \neq c$. Then

$$P_c = \frac{TP_c}{TP_c + FP_c}, \qquad
R_c = \frac{TP_c}{TP_c + FN_c}, \qquad
F1_c = \frac{2 P_c R_c}{P_c + R_c} = \frac{2\,TP_c}{2\,TP_c + FP_c + FN_c}$$

Macro-F1 is the unweighted mean over the two classes:

$$\text{Macro-F1} = \frac{F1_0 + F1_1}{2}$$

If $2\,TP_c + FP_c + FN_c = 0$ we set $F1_c = 0$. Unlike accuracy, macro-F1
penalizes a system that predicts mostly one class: predicting class 0 for
every review gives accuracy 0.5 but macro-F1 of only 0.333, because
$F1_1 = 0$.

### ROC-AUC

The area under the ROC curve, which plots the true-positive rate against the
false-positive rate as the decision threshold on $s_i$ varies. It equals the
probability that a randomly chosen class-1 review receives a higher score than
a randomly chosen class-0 review, with ties counted as one half:

$$\text{AUC} = \frac{1}{N_1 N_0} \sum_{i:\,y_i = 1} \; \sum_{j:\,y_j = 0}
\left( \mathbb{1}[s_i > s_j] + \tfrac{1}{2}\,\mathbb{1}[s_i = s_j] \right)$$

where $N_1$ and $N_0$ are the numbers of class-1 and class-0 reviews. We
compute it from ranks (the Mann–Whitney U statistic divided by $N_1 N_0$),
which gives the same value. AUC depends only on the ordering of the scores,
not on the threshold used to produce $\hat{y}_i$, so it shows whether a system
that loses accuracy on a shifted test set still ranks reviews well and has
only a badly placed threshold. A constant score gives AUC 0.5.

## Running the script

`score.py` needs Python 3 and no extra packages.

```
python score.py --predictions predictions.jsonl --gold test.jsonl
```

- `--gold`: a test file from our dataset (see `data.md`), for example
  `data/electronics_2015_2018/test.jsonl`. Each line is a JSON object whose
  `label` field is `true` or `false`.
- `--predictions`: one JSON object per line with these fields:

| Field | Meaning |
|---|---|
| `row_id` | 0-based index of the record in the gold file |
| `prediction` | 0 or 1 (1 = at least 3 helpful votes) |
| `score` | a number, larger = more likely class 1 (used for ROC-AUC only) |
| `id` | optional; the `id` of the gold record, used to check alignment |

Example prediction lines:

```
{"row_id": 0, "prediction": 0, "score": 0.0}
{"row_id": 1, "prediction": 0, "score": 0.0}
```

The script stops with an error message if the number of predictions differs
from the number of gold records, if a `row_id` is missing, repeated or out of
range, if a `prediction` is not 0 or 1, or if an `id` does not match the gold
record. Add `--json` to print the results as a single JSON object.

### Example

Scoring the all-zero baseline on the in-distribution test set:

```
$ python score.py --predictions all_zero.electronics_2015_2018.jsonl \
                  --gold data/electronics_2015_2018/test.jsonl
examples: 3000
accuracy: 0.5000
macro_f1: 0.3333
roc_auc:  0.5000
```

We score every system on five test sets: Electronics 2015–2018
(in-distribution), Electronics 2021 (time shift), and Books, Home & Kitchen,
and Beauty & Personal Care from 2015–2018 (category shift).

## References

- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition
  Letters*, 27(8), 861–874.
- Hanley, J. A., and McNeil, B. J. (1982). The meaning and use of the area
  under a receiver operating characteristic (ROC) curve. *Radiology*, 143(1),
  29–36.
- Sokolova, M., and Lapalme, G. (2009). A systematic analysis of performance
  measures for classification tasks. *Information Processing & Management*,
  45(4), 427–437.
- Wikipedia. Accuracy and precision.
  https://en.wikipedia.org/wiki/Accuracy_and_precision
- Wikipedia. F-score. https://en.wikipedia.org/wiki/F-score
- Wikipedia. Receiver operating characteristic.
  https://en.wikipedia.org/wiki/Receiver_operating_characteristic
