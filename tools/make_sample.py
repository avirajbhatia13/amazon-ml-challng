"""Cut a small, realistic sample of the training data for local development.

  python tools/make_sample.py                       # ~100k S1 dev-train, ~50k S1 dev-test
  python tools/make_sample.py --train-s1 30000 --test-s1 15000   # smaller / faster

Sampling is by whole states, so every entity keeps its look-alike competitors
(same-state businesses with similar names/addresses). Sampled states are split
into two disjoint sets:
  data/sample/train/  train_source{1,2,3}.tsv + train_ground_truth.tsv   (fit the model here)
  data/sample/test/   test_source{1,2,3}.tsv                             (predict here, like the real test)
  data/sample/test_labels.tsv                                            (held-out answers, for local scoring)
Reads one source file at a time to keep memory low.
"""
import argparse
import random
import sys
import zlib
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from lexicon import state_map  # noqa: E402

RAW = ROOT / "data" / "raw" / "archive" / "dataset" / "train"
OUT = ROOT / "data" / "sample"


def read(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path, separator="\t", quote_style="never")


def with_state(df):
    parts = []
    for country in df["country"].unique().to_list():
        smap = state_map(country)
        part = df.filter(pl.col("country") == country)
        st = (part["business_address"].fill_null("").str.to_lowercase().str.split(",")
              .list.eval(pl.element().str.strip_chars(" .-#()").replace_strict(smap, default=None))
              .list.drop_nulls().list.first())
        parts.append(part.with_columns(state=pl.concat_str([pl.lit(country + ":"), st])))
    return pl.concat(parts)


def pick_states(s1, target, rng, exclude=()):
    """Whole states, per country in proportion to its share of S1."""
    counts = s1.group_by("country", "state").len().drop_nulls("state").sort("state")
    share = s1.group_by("country").len().with_columns(pl.col("len") / s1.height)
    chosen = []
    for country, frac in sorted(share.iter_rows()):
        want = target * frac
        c = counts.filter((pl.col("country") == country) & ~pl.col("state").is_in(list(exclude)))
        size = dict(zip(c["state"], c["len"]))
        states = [s for s in c["state"].to_list() if size[s] <= max(want, 1) * 1.2]
        rng.shuffle(states)
        total = 0
        for s in states:
            if total >= want:
                break
            if total + size[s] <= want * 1.3:
                chosen.append(s)
                total += size[s]
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-s1", type=int, default=100_000)
    ap.add_argument("--test-s1", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    s1 = with_state(read(RAW / "train_source1.tsv"))
    train_states = pick_states(s1, args.train_s1, rng)
    test_states = pick_states(s1, args.test_s1, rng, exclude=set(train_states))
    print("dev-train states:", sorted(train_states))
    print("dev-test states: ", sorted(test_states))

    gt = read(RAW / "train_ground_truth.tsv").with_columns(pl.col("matched_entity_ids").fill_null(""))
    owner = (gt.select(s1="source1_entity_id", m=pl.col("matched_entity_ids").str.split(","))
             .explode("m").filter(pl.col("m") != ""))

    for name, states in (("train", train_states), ("test", test_states)):
        s1_part = s1.filter(pl.col("state").is_in(states))
        ids = s1_part["entity_id"].implode()
        write(s1_part.drop("state"), OUT / name / f"{name}_source1.tsv")
        my_matches = owner.filter(pl.col("s1").is_in(ids))["m"].implode()
        all_matched = owner["m"].implode()
        n2 = n3 = 0
        for k in (2, 3):
            sx = with_state(read(RAW / f"train_source{k}.tsv"))
            h = sx["entity_id"].map_elements(lambda x: zlib.crc32(x.encode()) % 1000, return_dtype=pl.Int64)
            frac = len(s1_part) / len(s1)
            keep = (pl.col("entity_id").is_in(my_matches)                        # true matches
                    | pl.col("state").is_in(states)                                # everything else in these states
                    | (pl.col("state").is_null() & ~pl.col("entity_id").is_in(all_matched)
                       & (h < 1000 * frac)))                                       # stateless unmatched noise
            sx = sx.filter(keep).drop("state")
            write(sx, OUT / name / f"{name}_source{k}.tsv")
            n2, n3 = (sx.height, n3) if k == 2 else (n2, sx.height)
        labels = gt.filter(pl.col("source1_entity_id").is_in(ids))
        write(labels, OUT / name / "train_ground_truth.tsv" if name == "train" else OUT / "test_labels.tsv")
        print(f"{name}: S1={s1_part.height} S2={n2} S3={n3}",
              dict(s1_part.group_by("country").len().iter_rows()))


if __name__ == "__main__":
    main()
