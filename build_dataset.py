#!/usr/bin/env python3
"""
Build the CIS 5300 review-helpfulness dataset from Amazon Reviews 2023 (raw review files).

Label:  helpful_vote >= 3 -> True,  helpful_vote == 0 -> False,  1-2 votes are dropped.
Sampling: 50/50 True/False inside every (cohort, year, split).
Splits:   main data is split by PRODUCT (parent_asin hash): 80% train / 10% dev / 10% test.
          Shifted test sets only use products that were never in train/dev.
Dedup:    identical (title + text) is kept only once across ALL splits and cohorts.

Usage (full run, needs internet; each category file is one streaming pass):
    python3 build_dataset.py --out data
Use a local copy if you downloaded the files yourself (recommended for the huge Electronics file):
    python3 build_dataset.py --out data --src Electronics=/path/Electronics.jsonl.gz \
        --src Books=/path/Books.jsonl.gz --src Home_and_Kitchen=... --src Beauty_and_Personal_Care=...
Quick smoke test:
    python3 build_dataset.py --out data_smoke --scale 0.02 --max-lines 300000
Pack:
    python3 build_dataset.py --out data --pack     (also writes data.tar.gz)
"""
import argparse, gzip, hashlib, io, json, math, os, random, re, sys, tarfile, time, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

# Streamed straight from HuggingFace (plain .jsonl, nothing is stored on disk). One full pass per category,
# because the files are not sorted by year. Resumes automatically if the connection drops.
BASE_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{}.jsonl"
VOTE_RE = re.compile(r'"helpful_votes?":\s*(\d+)')
TS_RE = re.compile(r'"(?:sort_)?timestamp":\s*(\d+)')
SEED = 530
OVERSAMPLE = 1.2  # collect extra so that deduplication does not leave a bucket short
POS_MIN_VOTES = 3

# name, category, years, quota per split (total reviews, split evenly over years and labels),
# fixed_split: shifted test sets always go to "test"
COHORTS = [
    dict(name="electronics_2015_2018", cat="Electronics", years=[2015, 2016, 2017, 2018],
         quota={"train": 24000, "dev": 3000, "test": 3000}),
    dict(name="electronics_2012_2014", cat="Electronics", years=[2012, 2013, 2014],
         quota={"train": 24000, "dev": 3000}),
    dict(name="electronics_2019_2020", cat="Electronics", years=[2019, 2020],
         quota={"train": 24000, "dev": 3000}),
    dict(name="shift_electronics_2021", cat="Electronics", years=[2021],
         quota={"test": 5000}, fixed_split="test"),
    dict(name="shift_books_2015_2018", cat="Books", years=[2015, 2016, 2017, 2018],
         quota={"test": 5000}, fixed_split="test"),
    dict(name="shift_home_kitchen_2015_2018", cat="Home_and_Kitchen", years=[2015, 2016, 2017, 2018],
         quota={"test": 5000}, fixed_split="test"),
    dict(name="shift_beauty_2015_2018", cat="Beauty_and_Personal_Care", years=[2015, 2016, 2017, 2018],
         quota={"test": 5000}, fixed_split="test"),
]


def product_split(asin):
    """Deterministic 80/10/10 split by product id."""
    h = int(hashlib.md5(asin.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("dev" if h < 90 else "test")


def text_key(title, text):
    norm = " ".join((title + " " + text).lower().split())
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def year_of(rec):
    ts = rec.get("timestamp", rec.get("sort_timestamp"))
    if ts is None:
        return None
    ts = float(ts)
    if ts > 1e11:  # milliseconds
        ts /= 1000.0
    return datetime.fromtimestamp(ts, tz=timezone.utc).year


class ResumableHTTP(io.RawIOBase):
    """Read an HTTP file as a stream; if the connection drops, reconnect with a Range header and continue."""

    def __init__(self, url, retries=30):
        self.url, self.retries, self.pos, self.resp, self.total = url, retries, 0, None, None

    def readable(self):
        return True

    def _open(self):
        headers = {"User-Agent": "cis5300-data-builder"}
        if self.pos:
            headers["Range"] = f"bytes={self.pos}-"
        resp = urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=60)
        if self.pos and resp.status != 206:
            raise IOError(f"server ignored Range (status {resp.status}); cannot resume")
        # total file size, so that a connection closed early is not mistaken for end-of-file
        cr = resp.headers.get("Content-Range")
        cl = resp.headers.get("Content-Length")
        if cr and "/" in cr and cr.rsplit("/", 1)[1].isdigit():
            self.total = int(cr.rsplit("/", 1)[1])
        elif cl and cl.isdigit():
            self.total = self.pos + int(cl)
        self.resp = resp

    def readinto(self, b):
        for attempt in range(self.retries):
            try:
                if self.resp is None:
                    self._open()
                data = self.resp.read(len(b))
                if not data and self.total is not None and self.pos < self.total:
                    raise IOError(f"connection closed early at {self.pos:,}/{self.total:,}")
                b[:len(data)] = data
                self.pos += len(data)
                return len(data)
            except Exception as e:  # dropped connection, timeout, incomplete read
                print(f"  connection problem at byte {self.pos:,} ({e!r}); retry {attempt + 1}", file=sys.stderr)
                self.resp = None
                time.sleep(min(2 ** attempt, 30))
        raise IOError(f"giving up on {self.url} at byte {self.pos:,}")


def open_stream(src):
    if os.path.exists(src):
        raw = open(src, "rb")
    else:
        raw = io.BufferedReader(ResumableHTTP(src), buffer_size=8 * 1024 * 1024)
    if src.endswith(".gz"):
        raw = gzip.GzipFile(fileobj=raw)
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def offer(bucket, item, cap, rng):
    """Reservoir sampling into bucket = [seen_count, items]."""
    bucket[0] += 1
    if len(bucket[1]) < cap:
        bucket[1].append(item)
    else:
        j = rng.randrange(bucket[0])
        if j < cap:
            bucket[1][j] = item


def scan_category(cat, src, cohorts, scale, rng, max_lines):
    """One streaming pass over a category file; returns {bucket_key: [seen, items]}."""
    buckets = defaultdict(lambda: [0, []])
    caps = {}
    for ci, c in cohorts:
        for split, total in c["quota"].items():
            per_label = total * scale / len(c["years"]) / 2
            caps[(ci, split)] = max(1, math.ceil(per_label * OVERSAMPLE))
    n = bad = kept = 0
    wanted_years = {y for _, c in cohorts for y in c["years"]}
    with open_stream(src) as f:
        for line in f:
            n += 1
            if max_lines and n > max_lines:
                break
            if n % 1_000_000 == 0:
                print(f"  [{cat}] {n:,} lines read, {kept:,} candidates kept", file=sys.stderr)
            # cheap regex pre-filter: skip 1-2 vote reviews and unwanted years without parsing the JSON
            m = VOTE_RE.search(line)
            if m and 0 < int(m.group(1)) < POS_MIN_VOTES:
                continue
            m = TS_RE.search(line)
            if m:
                t = float(m.group(1))
                if datetime.fromtimestamp(t / 1000.0 if t > 1e11 else t, tz=timezone.utc).year not in wanted_years:
                    continue
            try:
                r = json.loads(line)
            except ValueError:
                bad += 1
                continue
            votes = int(r.get("helpful_vote", r.get("helpful_votes")) or 0)  # field name differs by file version
            if 0 < votes < POS_MIN_VOTES:
                continue
            label = votes >= POS_MIN_VOTES
            title = (r.get("title") or "").strip()
            text = (r.get("text") or "").strip()
            asin = r.get("parent_asin") or r.get("asin")
            if not text or not asin:
                continue
            year = year_of(r)
            if year is None:
                continue
            psplit = product_split(asin)
            for ci, c in cohorts:
                if year not in c["years"]:
                    continue
                if "fixed_split" in c:
                    split = c["fixed_split"]
                    if cat == "Electronics" and psplit != "test":  # unseen products only
                        continue
                else:
                    split = psplit
                    if split not in c["quota"]:
                        continue
                item = dict(id=f"{cat}:{asin}:{r.get('user_id', '')}:{r.get('timestamp', '')}",
                            category=cat, year=year, product=asin, title=title, text=text,
                            rating=r.get("rating"), verified_purchase=r.get("verified_purchase"),
                            helpful_vote=votes, label=label)
                offer(buckets[(ci, year, split, label)], item, caps[(ci, split)], rng)
                kept += 1
    print(f"  [{cat}] done: {n:,} lines, {bad} unparsable", file=sys.stderr)
    return dict(buckets)


def scan_job(job):
    cat, src, cohorts, scale, max_lines = job
    print(f"Scanning {cat} from {src}", file=sys.stderr)
    return scan_category(cat, src, cohorts, scale, random.Random(f"{SEED}-{cat}"), max_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--src", action="append", default=[], help="Category=path_or_url (repeatable)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all quotas (smoke tests)")
    ap.add_argument("--max-lines", type=int, default=0, help="stop each file after N lines (smoke tests)")
    ap.add_argument("--jobs", type=int, default=4, help="categories scanned in parallel")
    ap.add_argument("--pack", action="store_true", help="also write <out>.tar.gz")
    args = ap.parse_args()

    srcs = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in args.src}
    rng = random.Random(SEED)
    jobs = []
    for cat in dict.fromkeys(c["cat"] for c in COHORTS):
        cohorts = [(i, c) for i, c in enumerate(COHORTS) if c["cat"] == cat]
        jobs.append((cat, srcs.get(cat, BASE_URL.format(cat)), cohorts, args.scale, args.max_lines))
    all_buckets = {}
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as ex:  # one process per category
        for part in ex.map(scan_job, jobs):
            all_buckets.update(part)

    # Assemble: test first, then dev, then train, so duplicates are removed from the train side.
    seen = set()
    result = defaultdict(list)          # (cohort name, split) -> items
    shortfall = []
    for split in ("test", "dev", "train"):
        for ci, c in enumerate(COHORTS):
            if split not in c["quota"]:
                continue
            target = round(c["quota"][split] * args.scale / len(c["years"]) / 2)
            for year in c["years"]:
                for label in (True, False):
                    items = list(all_buckets.get((ci, year, split, label), [0, []])[1])
                    rng.shuffle(items)
                    got = 0
                    for it in items:
                        k = text_key(it["title"], it["text"])
                        if k in seen:
                            continue
                        seen.add(k)
                        result[(c["name"], split)].append(it)
                        got += 1
                        if got >= target:
                            break
                    if got < target:
                        shortfall.append(f"{c['name']}/{split}/{year}/label={label}: {got}/{target}")

    stats = {}
    for (name, split), items in sorted(result.items()):
        rng.shuffle(items)
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        stats[f"{name}/{split}"] = dict(
            n=len(items), true=sum(it["label"] for it in items),
            false=sum(not it["label"] for it in items),
            products=len({it["product"] for it in items}),
            mean_words=round(sum(len(it["text"].split()) for it in items) / max(1, len(items)), 1))

    # product leakage check inside the main cohort and against the 2021 test set
    prod = lambda n, s: {it["product"] for it in result.get((n, s), [])}
    main_n = "electronics_2015_2018"
    leaks = {
        "train&dev": len(prod(main_n, "train") & prod(main_n, "dev")),
        "train&test": len(prod(main_n, "train") & prod(main_n, "test")),
        "dev&test": len(prod(main_n, "dev") & prod(main_n, "test")),
        "train&shift_electronics_2021": len(prod(main_n, "train") & prod("shift_electronics_2021", "test")),
    }
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(dict(stats=stats, product_leakage=leaks, shortfall=shortfall), f, indent=2)
    write_data_md(args.out, stats, leaks, shortfall, result)
    print(json.dumps(dict(stats=stats, product_leakage=leaks, shortfall=shortfall), indent=2))
    if args.pack:
        tar_path = args.out.rstrip("/") + ".tar.gz"
        with tarfile.open(tar_path, "w:gz") as t:
            t.add(args.out, arcname=os.path.basename(args.out.rstrip("/")))
        print("wrote", tar_path)


def write_data_md(out, stats, leaks, shortfall, result):
    ex = next(iter(result.get(("electronics_2015_2018", "train"), [])), None)
    ex_json = json.dumps(ex, ensure_ascii=False, indent=2) if ex else "(no example)"
    rows = "\n".join(f"| {k} | {v['n']} | {v['true']} | {v['false']} | {v['products']} | {v['mean_words']} |"
                     for k, v in sorted(stats.items()))
    md = f"""# data.md

## Source
Amazon Reviews 2023 (Hou et al., 2024), raw review files by category
(https://amazon-reviews-2023.github.io/). Categories used: Electronics, Books, Home_and_Kitchen,
Beauty_and_Personal_Care. Full dataset (if only a sample is uploaded): <DRIVE_LINK>

## Task and label
Input: review title + text. Output: True / False ("many readers found the review helpful").
True = at least {POS_MIN_VOTES} helpful votes; False = 0 helpful votes; reviews with 1-2 votes are discarded.
The data has no total-vote count, so the label is a vote-popularity proxy, not a helpful/total ratio.
Within every (cohort, year, split) True and False are sampled 50/50, so chance accuracy is 50%.

## Format
One JSON object per line (`.jsonl`). Fields: `id`, `category`, `year`, `product` (parent_asin), `title`, `text`,
`rating`, `verified_purchase`, `helpful_vote`, `label` (true/false).

Example (first line of electronics_2015_2018/train.jsonl):
```json
{ex_json}
```

## Layout and sizes
`electronics_2015_2018/{{train,dev,test}}.jsonl` is the main data, split 80/10/10 by product.
`electronics_2012_2014` and `electronics_2019_2020` provide equal-sized training (and dev) sets for the
time-shift experiment. `shift_*` are test-only sets: Electronics 2021 (time shift; products never seen in
train/dev) and Books / Home & Kitchen / Beauty 2015-2018 (category shift).

| split | reviews | True | False | products | mean words |
|---|---|---|---|---|---|
{rows}

## Leakage checks
Product overlap in the main cohort: {json.dumps(leaks)}. Duplicate (title+text) removed across all splits.
{"Buckets below target size: " + "; ".join(shortfall) if shortfall else "All buckets reached their target size."}
"""
    with open(os.path.join(out, "data.md"), "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
