"""Generate all-zero and word-count baseline predictions for Milestone 2."""

import json
from pathlib import Path

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


def choose_threshold(dev_path):
    """Maximize dev accuracy, choosing the smallest threshold if tied."""
    reviews = read_jsonl(dev_path)
    if not reviews:
        raise ValueError(f"Development set is empty: {dev_path}")

    examples = [
        (word_count(review), to_label(review["label"], f"{dev_path}, row {i}"))
        for i, review in enumerate(reviews)
    ]

    # With prediction = int(count >= threshold), predictions change only
    # when the threshold passes an observed word count.
    candidates = {0}
    candidates.update(count + 1 for count, _ in examples)

    def correct_predictions(threshold):
        return sum((count >= threshold) == label for count, label in examples)

    threshold = max(sorted(candidates), key=correct_predictions)
    return threshold, correct_predictions(threshold) / len(examples)


def write_prediction(output, row_id, review, prediction, score):
    record = {"row_id": row_id, "prediction": prediction, "score": score}
    if "id" in review:
        record["id"] = review["id"]
    output.write(json.dumps(record) + "\n")


def main():
    threshold, dev_accuracy = choose_threshold(DATA_DIR / MAIN_SPLIT / "dev.jsonl")
    print(f"Word-count threshold: {threshold} (dev accuracy: {dev_accuracy:.4f})")

    OUTPUT_DIR.mkdir(exist_ok=True)
    for split in TEST_SPLITS:
        reviews = read_jsonl(DATA_DIR / split / "test.jsonl")
        zero_path = OUTPUT_DIR / f"all_zero.{split}.jsonl"
        count_path = OUTPUT_DIR / f"word_count.{split}.jsonl"

        with zero_path.open("w", encoding="utf-8") as zero_output, \
             count_path.open("w", encoding="utf-8") as count_output:
            for row_id, review in enumerate(reviews):
                count = word_count(review)
                write_prediction(zero_output, row_id, review, 0, 0.0)
                write_prediction(
                    count_output, row_id, review, int(count >= threshold), count
                )

        print(f"{split}: wrote {len(reviews)} predictions per baseline")


if __name__ == "__main__":
    main()
