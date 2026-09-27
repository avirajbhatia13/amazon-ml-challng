# Amazon ML Challenge 2026 — Business Entity Resolution

For each Source 1 business, find its records in Source 2 and Source 3 using only the name and address.
Scored by macro F0.5, so false merges cost about twice as much as misses.

## Two ways to run
| | Where | Data | Use for |
|---|---|---|---|
| **Local sample** | your Mac | `data/sample` (~115k + 56k S1, whole states) | developing, debugging, quick experiments |
| **Full run** | Kaggle (30 GB RAM) | full dataset | leaderboard submissions. See [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) |

## Local workflow
```bash
source .venv/bin/activate
python tools/make_sample.py                  # once: cut data/sample from data/raw (~1 min)

# full pipeline on the sample (~12 min; training is the slow part on a Mac)
python code/run_pipeline.py --data data/sample --work work/sample all

# honest score: the sample's test states were never seen in training
python code/run_pipeline.py --data data/sample --work work/sample score --labels data/sample/test_labels.tsv

# look at mistakes: false merges, missed matches, blocking misses
python code/errors.py --data data/sample --work work/sample --n 20 [--country India]

# official format check
python data/raw/archive/validate_submission.py -m work/sample/output/matching_results.tsv \
    -c work/sample/output/candidate_pairs.tsv -t data/sample/test
```
You can run stages one at a time, since each one reuses the previous one's cache:
`learn normalize-train featurize-train train normalize-test featurize-test predict`.
For example, after changing only the model, run just `train predict score`.

## Pipeline (`code/`)
| file | what it does |
|---|---|
| `translit.py` | learns native-script → English words (Hindi, Tamil, Kannada, Gujarati, Bengali names; state names) from training pairs |
| `lexicon.py` | legal forms (Inc/Pvt Ltd/SARL…), per-country address abbreviations, US and Indian states, city aliases |
| `normalize.py` | cleans names (reordering, legal words anywhere, domains, digit-for-letter swaps, mojibake) and addresses (shuffled parts, states, house numbers) |
| `blocking.py` | hash-join keys on rare name/address tokens → candidates, then cheap fuzzy pruning |
| `features.py` | 57 pair features: fuzzy scores, IDF-weighted overlaps, numbers, state, chain frequency, rank against competing candidates |
| `pipeline.py` | stages; grouped CV; threshold and assignment-policy search on out-of-fold macro F0.5 |
| `metric.py` | the challenge metric |
| `errors.py` | error analysis |

## Current results (local sample)
- Blocking recall after pruning: 97.3% (US), 96.2% (India), about 14 candidates per S1 entity
- CV macro F0.5: 0.969. **Held-out states: 0.9655** (precision 0.986, recall 0.935)
