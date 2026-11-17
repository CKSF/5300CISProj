"""Generate TF-IDF + metadata logistic regression baseline predictions for Milestone 2."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from score import read_jsonl, to_label


MILESTONE_DIR = Path(__file__).resolve().parent
DATA_DIR = MILESTONE_DIR.parent / "data"
OUTPUT_DIR = MILESTONE_DIR / "predictions"
MAIN_SPLIT = "electronics_2015_2018"
TEST_SPLITS = (
    MAIN_SPLIT,
    "shift_electronics_2021",
    "shift_books_2015_2018",
    "shift_home_kitchen_2015_2018",
    "shift_beauty_2015_2018",
)


def word_count(review):
    """Count whitespace-separated words in the review title and body."""
    title = review.get("title") or ""
    body = review.get("text") or ""
    return len(title.split()) + len(body.split())


def extract_features_and_labels(reviews, source_name, require_labels=True):
    """Extract combined text, numerical metadata, and binary labels from reviews."""
    texts = []
    metadata = []
    labels = []

    for i, review in enumerate(reviews):
        title = (review.get("title") or "").strip()
        body = (review.get("text") or "").strip()
        texts.append(f"{title} {body}".strip())

        rating = float(review.get("rating") or 0.0)
        verified = 1.0 if review.get("verified_purchase") in (True, 1, "True", "true") else 0.0
        log_words = np.log1p(word_count(review))
        metadata.append([rating, verified, log_words])

        if require_labels and "label" in review:
            labels.append(to_label(review["label"], f"{source_name}, row {i}"))

    meta_array = np.asarray(metadata, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64) if labels else None
    return texts, meta_array, label_array


def write_prediction(output, row_id, review, prediction, score):
    record = {
        "row_id": int(row_id),
        "prediction": int(prediction),
        "score": round(float(score), 6),
    }
    if "id" in review:
        record["id"] = review["id"]
    output.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train and run TF-IDF + metadata logistic regression strong baseline."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Root data directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output predictions directory")
    parser.add_argument("--max-features", type=int, default=50000, help="Max TF-IDF vocabulary size")
    parser.add_argument(
        "--c-grid",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0, 2.0, 5.0],
        help="Regularization strength C candidates to tune on dev",
    )
    args = parser.parse_args()

    train_path = args.data_dir / MAIN_SPLIT / "train.jsonl"
    dev_path = args.data_dir / MAIN_SPLIT / "dev.jsonl"

    train_reviews = read_jsonl(train_path)
    dev_reviews = read_jsonl(dev_path)

    train_texts, train_meta, y_train = extract_features_and_labels(train_reviews, train_path)
    dev_texts, dev_meta, y_dev = extract_features_and_labels(dev_reviews, dev_path)

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    scaler = StandardScaler()

    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_train_meta = csr_matrix(scaler.fit_transform(train_meta))
    X_train = hstack([X_train_tfidf, X_train_meta], format="csr")

    X_dev_tfidf = vectorizer.transform(dev_texts)
    X_dev_meta = csr_matrix(scaler.transform(dev_meta))
    X_dev = hstack([X_dev_tfidf, X_dev_meta], format="csr")

    best_c = None
    best_dev_acc = -1.0
    best_clf = None

    for c_val in args.c_grid:
        clf = LogisticRegression(C=c_val, max_iter=1000, solver="lbfgs", random_state=42)
        clf.fit(X_train, y_train)
        dev_scores = clf.predict_proba(X_dev)[:, 1]
        dev_preds = (dev_scores >= 0.5).astype(int)
        dev_acc = accuracy_score(y_dev, dev_preds)
        dev_f1 = f1_score(y_dev, dev_preds, average="macro")
        dev_auc = roc_auc_score(y_dev, dev_scores)
        print(
            f"Dev (C={c_val}): accuracy={dev_acc:.4f}, macro_f1={dev_f1:.4f}, roc_auc={dev_auc:.4f}"
        )
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_c = c_val
            best_clf = clf

    print(f"Selected C={best_c} (dev accuracy: {best_dev_acc:.4f})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in TEST_SPLITS:
        test_path = args.data_dir / split / "test.jsonl"
        reviews = read_jsonl(test_path)
        test_texts, test_meta, _ = extract_features_and_labels(
            reviews, test_path, require_labels=False
        )

        X_test_tfidf = vectorizer.transform(test_texts)
        X_test_meta = csr_matrix(scaler.transform(test_meta))
        X_test = hstack([X_test_tfidf, X_test_meta], format="csr")

        scores = best_clf.predict_proba(X_test)[:, 1]
        preds = (scores >= 0.5).astype(int)

        out_path = args.output_dir / f"strong.{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as out_file:
            for row_id, (review, pred, score) in enumerate(zip(reviews, preds, scores)):
                write_prediction(out_file, row_id, review, int(pred), float(score))

        print(f"{split}: wrote {len(reviews)} predictions to {out_path.name}")


if __name__ == "__main__":
    main()