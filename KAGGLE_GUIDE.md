# Running the pipeline on Kaggle

## What Kaggle gives you
Kaggle (owned by Google) offers free **cloud notebooks**: a Jupyter notebook in your browser, running on
a Kaggle computer instead of your Mac. A CPU notebook has:

| | Kaggle CPU notebook | Your Mac |
|---|---|---|
| RAM | **30 GB** | 8 GB (≈3 GB free) |
| CPU cores | 4 | 8 |
| Disk for outputs | 20 GB (`/kaggle/working`) | ~12 GB free |
| Max run time | 12 hours per session | — |
| Cost | free (CPU time has no weekly quota; only GPU time does) | — |

You get files onto Kaggle by uploading them as **Datasets**. A dataset is a folder of files that is
stored on Kaggle and can be attached to any notebook. We use two, and both stay **private**:
1. **the challenge data**: upload once.
2. **our code**: re-upload as a new version whenever the code changes.

The notebook (`notebooks/kaggle_run.ipynb`) then runs `code/run_pipeline.py` on the full data and
writes `matching_results.tsv` + `candidate_pairs.tsv`, which you download and submit.

> **Rules check:** running code on Kaggle is fine. The challenge bans *external data and lookups*,
> not cloud compute. Keep both datasets **Private**: the challenge data is not yours to publish.

---

## One-time setup (≈15 minutes)

### Step 1: Create an account and verify your phone
1. Go to <https://www.kaggle.com> → **Register** (the "Sign in with Google" option is quickest).
2. Click your profile picture (top right) → **Settings**.
3. Under **Phone Verification**, verify your phone number.
   *Why:* notebooks can only use the Internet (needed for `pip install`) after phone verification.

### Step 2: Upload the challenge data as a private dataset
1. Go to <https://www.kaggle.com/datasets> → **+ New Dataset**.
2. Drag in **`data/raw/archive.zip`** from this project (1.1 GB). Kaggle unzips it automatically.
   Wait for the upload bar to finish; don't close the tab.
3. Title: `amazon-ml-2026-data`.
4. Make sure visibility is **Private** → **Create**.
5. Processing takes a few minutes. When it's done, the dataset page shows `dataset/train`, `dataset/test`
   and `validate_submission.py`.

### Step 3: Upload the code as a private dataset
1. In a terminal, in this project folder, run:
   ```bash
   tools/package_code.sh          # creates dist/er-code.zip
   ```
2. <https://www.kaggle.com/datasets> → **+ New Dataset** → drag in `dist/er-code.zip`.
3. Title: `er-code`, **Private** → **Create**.

### Step 4: Create the notebook
1. Go to <https://www.kaggle.com/code> → **+ New Notebook**.
2. **File → Import Notebook** → upload `notebooks/kaggle_run.ipynb` from this project.
3. In the **right-hand panel**:
   - **Session options → Accelerator: None** (CPU; we don't need a GPU).
   - **Session options → Internet: On** (appears only after phone verification).
   - **Input → + Add Input → Your Work → Datasets**: add `amazon-ml-2026-data` and `er-code`.
4. Rename the notebook at the top left (e.g. `er-full-run`).

---

## Running it

You can run a notebook in two ways:

**A. Interactively (recommended for the first run).** Click each cell and press **Shift + Enter**,
or use **Run All**. You watch the output live, but the browser tab must stay open, and the session
stops after about 20 minutes of inactivity. Use this to check that each stage works.

**B. In the background ("commit").** Click **Save Version** (top right) → **Save & Run All (Commit)**
→ **Save**. Kaggle runs the whole notebook on its own, even if you close the laptop. When it finishes
(the notebook's **Versions** list shows ✓), the outputs are kept permanently. Use this for full runs.

### What each cell does
| Cell | What happens | Expected time* |
|---|---|---|
| 1. Install | installs polars / rapidfuzz / unidecode (LightGBM is preinstalled) | 1 min |
| 2. Find folders | locates the attached datasets; prints RAM / CPUs / disk | seconds |
| 3a. learn + normalize-train | learns Hindi/Tamil/…→English words; cleans 12.5M train records | ~11 min |
| 3b. featurize-train | blocking + features; prints **recall per key** and **BLOCKING RECALL** per country | 25–40 min |
| 3c. train | stage 1 (pair model) then stage 2 (each pair judged against its competitors), 4-fold CV each → prints **CV macro F0.5** for both + threshold; fits final models | 75–100 min |
| 3d. test | cleans + featurizes test, writes predictions | 40–55 min |
| 4. Validate | copies the two files to `/kaggle/working/`, runs the official validator | 1 min |
| 5. Clean up | deletes the large cache so the output fits Kaggle's 20 GB limit | seconds |

\*From the first full run (≈2 h in total), adjusted for the heavier second version (≈3–3.5 h).
Use **Save & Run All (Commit)** for a run this long.

### Getting the results
Open the notebook → **Output** tab (or the version's page) → download `matching_results.tsv`
(and `candidate_pairs.tsv` for the final zip). Upload `matching_results.tsv` to the challenge
leaderboard.

### Updating the code
After changing code locally:
1. Run `tools/package_code.sh` again.
2. Open your `er-code` dataset page → **New Version** → upload the new `dist/er-code.zip` → **Create**.
3. In the notebook's right panel, the input shows an update icon; click it to use the latest version
   (or restart the session).

---

## If something goes wrong
| Symptom | Fix |
|---|---|
| `Internet` toggle is missing / `pip install` fails | Verify your phone (Step 1), then Internet: On |
| `StopIteration` in cell 2 | A dataset isn't attached; check **Input** in the right panel |
| Cell dies with "kernel restarted" / out of memory | Lower `--train-fraction` (e.g. 0.25) in cell 3c |
| Session stopped while you were away | Use **Save & Run All (Commit)** instead of interactive |
| Output larger than 20 GB | Make sure cell 5 ran (it deletes the cache) |

Paste the error text back to Claude and it can fix the code.
