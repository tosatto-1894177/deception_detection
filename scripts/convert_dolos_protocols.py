#!/usr/bin/env python3
"""
Convert DOLOS official Training_Protocols + labels into standardized JSON splits.

Outputs (by default):
- data/processed/splits/3fold/fold{1,2,3}/{train.json, val.json, test.json}
- data/processed/splits/default/{train.json, val.json, test.json}   (copied from fold1)
- data/processed/splits/gender/{female,male}/{train.json, val.json, test.json}  (test filtered)
- data/processed/splits/duration/{short,long}/{train.json, val.json, test.json} (test filtered)

USAGE
-----
python convert_dolos_protocols.py \
  --protocols_dir data/raw/dolos/Training_Protocols \
  --labels data/processed/labels/labels.csv \
  --out_dir data/processed/splits \
  --val_frac 0.2 \
  --seed 42

NOTES
-----
- The labels file must include at least: filename, label (0/1), subject_id.
- If your labels file is .xlsx, the script will read the first sheet by default.
- The Training_Protocols CSVs (train_fold*.csv, test_fold*.csv, female.csv, male.csv, short.csv, long.csv)
  are expected to have a 'filename' column; if not, the first column will be treated as filename.
- 'filename' matching is case-insensitive; extensions are normalized.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import GroupShuffleSplit

# ---------- Helpers ----------

def _norm_colnames(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _ensure_filename_col(df: pd.DataFrame) -> pd.Series:
    # Try common column names
    for key in ["filename", "file", "clip", "video", "name", "path"]:
        if key in df.columns:
            return df[key].astype(str)
    # Fallback: use first column
    return df.iloc[:, 0].astype(str)

def _normalize_filename(s: pd.Series) -> pd.Series:
    # Normalize extensions and strip spaces
    s = s.str.strip()
    # If entries contain paths, take basename
    s = s.apply(lambda x: os.path.basename(x))
    # ensure .mp4
    s = s.apply(lambda x: Path(x).with_suffix(".mp4").name)
    return s

def load_labels(labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if labels_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(labels_path, engine="openpyxl" if labels_path.suffix.lower()==".xlsx" else None)
    else:
        df = pd.read_csv(labels_path)
    df = _norm_colnames(df)
    # Map potential column synonyms
    colmap = {}
    # filename
    if "filename" not in df.columns:
        for cand in ["file", "clip", "video", "name", "path"]:
            if cand in df.columns:
                colmap[cand] = "filename"; break
    # label
    if "label" not in df.columns:
        # try veracity/truth fields
        for cand in ["veracity", "truth", "class", "ground_truth", "gt", "islie"]:
            if cand in df.columns:
                colmap[cand] = "label"; break
    # subject_id
    if "subject_id" not in df.columns:
        for cand in ["subject", "person_id", "speaker", "id", "pid", "participant"]:
            if cand in df.columns:
                colmap[cand] = "subject_id"; break
    if colmap:
        df = df.rename(columns=colmap)

    # normalize filename and enforce minimal columns
    df["filename"] = _normalize_filename(_ensure_filename_col(df))
    if "label" not in df.columns:
        raise ValueError("Labels file must contain a 'label' column (0/1 or truth/lie).")
    # normalize label to 0/1
    if df["label"].dtype == object:
        df["label"] = df["label"].str.strip().str.lower().map({"lie":1, "truth":0, "true":0, "false":1, "1":1, "0":0})
    df["label"] = df["label"].astype(int)

    if "subject_id" not in df.columns:
        raise ValueError("Labels file must contain a 'subject_id' column.")
    df["subject_id"] = df["subject_id"].astype(str)

    # Optional: gender/duration if present
    for opt in ["gender", "duration_sec", "episode"]:
        if opt not in df.columns:
            df[opt] = pd.NA

    # Drop duplicates on filename keep first
    df = df.drop_duplicates(subset=["filename"], keep="first").reset_index(drop=True)
    return df[["filename", "label", "subject_id", "gender", "duration_sec", "episode"]]

def load_filename_list(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    df = _norm_colnames(df)
    s = _ensure_filename_col(df)
    return _normalize_filename(s)

def to_json_list(df: pd.DataFrame):
    # df must have filename, label, subject_id
    out = [{"filename": r["filename"],
            "label": int(r["label"]),
            "subject_id": str(r["subject_id"])}
           for _, r in df.iterrows()]
    return out

def save_json_list(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_json_list(df), f, indent=2)
    print(f"Saved: {out_path} ({len(df)})")

def split_train_val_by_subject(train_df: pd.DataFrame, val_frac: float, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    idx_train, idx_val = next(gss.split(train_df, groups=train_df["subject_id"]))
    return train_df.iloc[idx_train].reset_index(drop=True), train_df.iloc[idx_val].reset_index(drop=True)

def intersect_by_filename(df: pd.DataFrame, names: pd.Series) -> pd.DataFrame:
    names = set(names.tolist())
    return df[df["filename"].isin(names)].reset_index(drop=True)

# ---------- Main conversion ----------

def main(args):
    protocols_dir = Path(args.protocols_dir)
    labels_path   = Path(args.labels)
    out_dir       = Path(args.out_dir)
    val_frac      = float(args.val_frac)
    seed          = int(args.seed)

    # Load labels master
    labels = load_labels(labels_path)
    print(f"Loaded labels: {labels.shape}, sample:\n{labels.head(3)}\n")

    # Load protocol lists
    def prot(name): return protocols_dir / name

    required = ["train_fold1.csv","test_fold1.csv","train_fold2.csv","test_fold2.csv","train_fold3.csv","test_fold3.csv"]
    missing = [x for x in required if not prot(x).exists()]
    if missing:
        raise FileNotFoundError(f"Missing protocol files: {missing} in {protocols_dir}")

    # Build 3fold splits
    for k in [1,2,3]:
        tr_names = load_filename_list(prot(f"train_fold{k}.csv"))
        te_names = load_filename_list(prot(f"test_fold{k}.csv"))
        # Join with labels (drop unknowns)
        tr_df = intersect_by_filename(labels, tr_names)
        te_df = intersect_by_filename(labels, te_names)

        if tr_df.empty or te_df.empty:
            raise RuntimeError(f"Empty split for fold{k}. Check filename matching.")

        # Create val from train (subject-disjoint)
        tr_sub, va_sub = split_train_val_by_subject(tr_df, val_frac, seed)

        save_json_list(tr_sub, out_dir / "3fold" / f"fold{k}" / "train.json")
        save_json_list(va_sub, out_dir / "3fold" / f"fold{k}" / "val.json")
        save_json_list(te_df,  out_dir / "3fold" / f"fold{k}" / "test.json")

    # Default = copy from fold1
    import shutil, os
    for split in ["train","val","test"]:
        src = out_dir / "3fold" / "fold1" / f"{split}.json"
        dst = out_dir / "default" / f"{split}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied default/{split}.json from fold1")

    # Gender/duration filtered TESTS (train/val copied from default)
    # We will use provided CSV lists female.csv, male.csv, short.csv, long.csv if they exist.
    filters = {
        "gender/female": prot("female.csv"),
        "gender/male":   prot("male.csv"),
        "duration/short": prot("short.csv"),
        "duration/long":  prot("long.csv"),
    }
    # Load default test
    default_test_path = out_dir / "default" / "test.json"
    default_test = json.loads(Path(default_test_path).read_text(encoding="utf-8"))
    default_test_names = pd.Series([item["filename"] for item in default_test])
    default_test_df = intersect_by_filename(labels, default_test_names)

    for key, fpath in filters.items():
        if not fpath.exists():
            print(f"[WARN] Filter file not found: {fpath} — skipping {key}")
            continue
        names = load_filename_list(fpath)
        te_filt = intersect_by_filename(default_test_df, names)
        if te_filt.empty:
            print(f"[WARN] Filtered test for '{key}' is empty. Check filenames.")
        # Save train/val identical to default (common practice) and filtered test
        base = out_dir / key
        # copy train/val
        for split in ["train","val"]:
            src = out_dir / "default" / f"{split}.json"
            dst = base / f"{split}.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        # write test
        save_json_list(te_filt, base / "test.json")

    print("\nAll done. Splits are under:", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols_dir", type=str, required=True,
                    help="Path to DOLOS Training_Protocols directory (with fold/gender/duration CSVs).")
    ap.add_argument("--labels", type=str, required=True,
                    help="Path to consolidated labels file: CSV/XLSX with at least filename,label,subject_id.")
    ap.add_argument("--out_dir", type=str, default="data/processed/splits",
                    help="Output directory for JSON splits.")
    ap.add_argument("--val_frac", type=float, default=0.2,
                    help="Validation fraction carved from train (subject-disjoint).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for GroupShuffleSplit.")
    args = ap.parse_args()
    main(args)
