# data.md

## Source
Amazon Reviews 2023 (Hou et al., 2024), raw review files by category
(https://amazon-reviews-2023.github.io/). Categories used: Electronics, Books, Home_and_Kitchen,
Beauty_and_Personal_Care. Full dataset (if only a sample is uploaded): <DRIVE_LINK>

## Task and label
Input: review title + text. Output: True / False ("many readers found the review helpful").
True = at least 3 helpful votes; False = 0 helpful votes; reviews with 1-2 votes are discarded.
The data has no total-vote count, so the label is a vote-popularity proxy, not a helpful/total ratio.
Within every (cohort, year, split) True and False are sampled 50/50, so chance accuracy is 50%.

## Format
One JSON object per line (`.jsonl`). Fields: `id`, `category`, `year`, `product` (parent_asin), `title`, `text`,
`rating`, `verified_purchase`, `helpful_vote`, `label` (true/false).

Example (first line of electronics_2015_2018/train.jsonl):
```json
{
  "id": "Electronics:B00PG6M934:AGYRUHCBV7WH7PW6AGNLE4OY4YXA:1455805176000",
  "category": "Electronics",
  "year": 2016,
  "product": "B00PG6M934",
  "title": "Great Camera",
  "text": "seeing is believing. The features are endless and the picture quality rivals my DSLR",
  "rating": 5.0,
  "verified_purchase": true,
  "helpful_vote": 0,
  "label": false
}
```

## Layout and sizes
`electronics_2015_2018/{train,dev,test}.jsonl` is the main data, split 80/10/10 by product.
`electronics_2012_2014` and `electronics_2019_2020` provide equal-sized training (and dev) sets for the
time-shift experiment. `shift_*` are test-only sets: Electronics 2021 (time shift; products never seen in
train/dev) and Books / Home & Kitchen / Beauty 2015-2018 (category shift).

| split | reviews | True | False | products | mean words |
|---|---|---|---|---|---|
| electronics_2012_2014/dev | 3000 | 1500 | 1500 | 1852 | 118.6 |
| electronics_2012_2014/train | 24000 | 12000 | 12000 | 15048 | 117.3 |
| electronics_2015_2018/dev | 3000 | 1500 | 1500 | 2096 | 82.7 |
| electronics_2015_2018/test | 3000 | 1500 | 1500 | 2042 | 83.0 |
| electronics_2015_2018/train | 24000 | 12000 | 12000 | 16410 | 83.7 |
| electronics_2019_2020/dev | 3000 | 1500 | 1500 | 2141 | 75.5 |
| electronics_2019_2020/train | 24000 | 12000 | 12000 | 16860 | 72.0 |
| shift_beauty_2015_2018/test | 5000 | 2500 | 2500 | 4412 | 54.2 |
| shift_books_2015_2018/test | 5000 | 2500 | 2500 | 4793 | 96.4 |
| shift_electronics_2021/test | 5000 | 2500 | 2500 | 2748 | 69.7 |
| shift_home_kitchen_2015_2018/test | 5000 | 2500 | 2500 | 4681 | 55.8 |

## Leakage checks
Product overlap in the main cohort: {"train&dev": 0, "train&test": 0, "dev&test": 0, "train&shift_electronics_2021": 0}. Duplicate (title+text) removed across all splits.
All buckets reached their target size.
