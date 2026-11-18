"""Fine-tune roberta-base on Electronics 2015-2018 and generate predictions for all 5 test splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from score import read_jsonl, to_label


MILESTONE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
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


class ReviewDataset(Dataset):
    def __init__(self, reviews, tokenizer, max_length=256, source_name="", has_labels=True):
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels
        self.labels = [
            to_label(r["label"], f"{source_name}, row {i}") if has_labels and "label" in r else 0
            for i, r in enumerate(reviews)
        ]

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        r = self.reviews[idx]
        title = (r.get("title") or "").strip()
        body = (r.get("text") or "").strip()
        text = f"{title} </s> {body}".strip() if title else body

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.has_labels:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def evaluate(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            if "labels" in batch:
                all_labels.extend(batch["labels"].numpy())

    probs_arr = np.asarray(all_probs, dtype=np.float64)
    preds_arr = (probs_arr >= 0.5).astype(int)
    return preds_arr, probs_arr, np.asarray(all_labels) if all_labels else None


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
    parser = argparse.ArgumentParser(description="Fine-tune roberta-base strong baseline.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-name", type=str, default="roberta-base")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)

    train_path = args.data_dir / MAIN_SPLIT / "train.jsonl"
    dev_path = args.data_dir / MAIN_SPLIT / "dev.jsonl"

    train_reviews = read_jsonl(train_path)
    dev_reviews = read_jsonl(dev_path)

    train_ds = ReviewDataset(train_reviews, tokenizer, args.max_length, str(train_path), has_labels=True)
    dev_ds = ReviewDataset(dev_reviews, tokenizer, args.max_length, str(dev_path), has_labels=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size * 2, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_dev_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        dev_preds, dev_scores, y_dev = evaluate(model, dev_loader, device)
        dev_acc = accuracy_score(y_dev, dev_preds)
        dev_f1 = f1_score(y_dev, dev_preds, average="macro")
        dev_auc = roc_auc_score(y_dev, dev_scores)
        print(
            f"Epoch {epoch}/{args.epochs} | loss={total_loss/len(train_loader):.4f} | "
            f"dev_acc={dev_acc:.4f}, dev_f1={dev_f1:.4f}, dev_auc={dev_auc:.4f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Restoring best model (dev accuracy: {best_dev_acc:.4f})")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in TEST_SPLITS:
        test_path = args.data_dir / split / "test.jsonl"
        reviews = read_jsonl(test_path)
        test_ds = ReviewDataset(reviews, tokenizer, args.max_length, str(test_path), has_labels=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)

        preds, scores, _ = evaluate(model, test_loader, device)
        out_path = args.output_dir / f"roberta.{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as out_file:
            for row_id, (review, pred, score) in enumerate(zip(reviews, preds, scores)):
                write_prediction(out_file, row_id, review, int(pred), float(score))
        print(f"{split}: wrote {len(reviews)} predictions to {out_path.name}")


if __name__ == "__main__":
    main()