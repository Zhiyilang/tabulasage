#!/usr/bin/env python3
"""
TabulaSage — an offline, open-source-AI data profiler.

Give it a CSV and it will:
  1. Profile every column (types, missingness, distribution).
  2. Detect anomalous rows with an IsolationForest (scikit-learn).
  3. Optionally train an interpretable model if you name a --target column.
  4. Turn all of that into plain-English insights you can paste into a report.

Everything runs locally on your CPU. No network, no API keys, no data
ever leaves your machine. The AI at the core (scikit-learn's IsolationForest
and tree-based models) is fully open source and auditable.

Usage:
    python tabulasage.py data.csv
    python tabulasage.py data.csv --target churned
    python tabulasage.py data.csv --json summary.json
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter

# --- open-source AI core (scikit-learn) -----------------------------------
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import __version__ as sklearn_version


def to_float(x):
    try:
        if x is None:
            return None
        x = x.strip()
        if x == "":
            return None
        return float(x)
    except ValueError:
        return None


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []
    return cols, rows


def infer_column(col, rows):
    values = [r.get(col) for r in rows]
    present = [v for v in values if v is not None and v.strip() != ""]
    missing = len(values) - len(present)
    numeric_flags = [to_float(v) for v in present]
    numeric_count = sum(1 for v in numeric_flags if v is not None)
    is_numeric = present and numeric_count >= 0.8 * len(present)
    if is_numeric:
        nums = [v for v in numeric_flags if v is not None]
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        mean = sum(nums_sorted) / n
        var = sum((x - mean) ** 2 for x in nums_sorted) / n
        std = math.sqrt(var)
        median = (nums_sorted[n // 2] if n % 2 else
                  (nums_sorted[n // 2 - 1] + nums_sorted[n // 2]) / 2)
        return {"name": col, "type": "numeric", "count": len(present),
                "missing": missing, "mean": mean, "std": std,
                "min": nums_sorted[0], "median": median, "max": nums_sorted[-1]}
    else:
        counter = Counter(present)
        top = counter.most_common(1)[0] if counter else ("", 0)
        return {"name": col, "type": "categorical", "count": len(present),
                "missing": missing, "unique": len(counter),
                "top_value": top[0], "top_freq": top[1]}


def build_matrix(cols, rows, profiles, exclude=frozenset()):
    """Return a numeric feature matrix + the feature names used.
    `exclude` holds column names to drop (e.g. the --target, to avoid leakage)."""
    cols = [c for c in cols if c not in exclude]
    num_cols = [p["name"] for p in profiles if p["type"] == "numeric" and p["name"] not in exclude]
    cat_cols = [p["name"] for p in profiles if p["type"] == "categorical" and p["name"] not in exclude]
    X, feat_names = [], list(num_cols)
    # simple numeric imputation (column median) — keeps everything offline & tiny
    medians = {c: next(p["median"] for p in profiles if p["name"] == c) for c in num_cols}
    # top categories for one-hot (keep it small & interpretable)
    top_cats = {}
    for c in cat_cols:
        counter = Counter(r.get(c) for r in rows if r.get(c) not in (None, ""))
        top_cats[c] = [v for v, _ in counter.most_common(5)]
        feat_names.extend(f"{c}={v}" for v in top_cats[c])
    for r in rows:
        row = []
        for c in num_cols:
            v = to_float(r.get(c))
            row.append(medians[c] if v is None else v)
        for c in cat_cols:
            val = r.get(c)
            row.extend([1.0 if val == t else 0.0 for t in top_cats[c]])
        X.append(row)
    return X, feat_names


def detect_anomalies(X):
    # standardize so mixed units (age vs income) don't distort the forest
    Xs = StandardScaler().fit_transform(X)
    model = IsolationForest(n_estimators=200, random_state=42)
    model.fit(Xs)
    scores = model.score_samples(Xs)        # lower = more anomalous
    # parameter-free threshold: flag points > 2 std below the mean anomaly score
    mean_s, std_s = scores.mean(), scores.std()
    threshold = mean_s - 2 * std_s
    anomalous_idx = [i for i in range(len(X)) if scores[i] < threshold]
    order = sorted(range(len(X)), key=lambda i: scores[i])
    return anomalous_idx, scores, order


def train_interpretable(X, y, feat_names):
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
        clf.fit(Xtr, ytr)
        acc = clf.score(Xte, yte)
        importances = sorted(zip(feat_names, clf.feature_importances_),
                             key=lambda t: t[1], reverse=True)
        importances = [(n, v) for n, v in importances if v > 0][:5]
        return {"accuracy": acc, "top_features": importances,
                "classes": [str(c) for c in clf.classes_]}
    except Exception as e:
        return {"error": str(e)}


def make_insights(profiles, n_rows, anomalous, model_info, dup_count):
    ins = []
    hi_missing = sorted([p for p in profiles if p["missing"] > 0],
                        key=lambda p: p["missing"], reverse=True)
    if hi_missing:
        worst = hi_missing[0]
        pct = 100 * worst["missing"] / n_rows
        ins.append(f"'{worst['name']}' has {worst['missing']} missing "
                   f"values ({pct:.1f}%) — the largest gap in the dataset.")
    if dup_count:
        ins.append(f"{dup_count} duplicate row(s) detected — worth deduplicating "
                   f"before analysis.")
    if anomalous:
        ins.append(f"IsolationForest flagged {len(anomalous)} row(s) "
                   f"({100*len(anomalous)/n_rows:.1f}%) as anomalous — "
                   f"likely data-entry errors or rare cases worth a human look.")
    else:
        ins.append("No strong outliers detected by the anomaly model.")
    if model_info and "accuracy" in model_info:
        top = model_info["top_features"][0] if model_info["top_features"] else None
        if top:
            ins.append(f"Predicting the target: a shallow decision tree reaches "
                       f"{model_info['accuracy']*100:.0f}% test accuracy; the "
                       f"strongest driver is '{top[0]}' (importance {top[1]:.2f}).")
    cat = [p for p in profiles if p["type"] == "categorical" and p.get("unique", 0) > 50]
    if cat:
        ins.append(f"Column '{cat[0]['name']}' looks like a free-text/id column "
                   f"({cat[0]['unique']} distinct values) — probably not useful as a category.")
    return ins


def fmt_num(v):
    if v is None:
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.3g}"


def main():
    ap = argparse.ArgumentParser(description="Offline open-source-AI CSV profiler")
    ap.add_argument("csv", help="path to a CSV file")
    ap.add_argument("--target", help="optional label column to model")
    ap.add_argument("--json", dest="json_out", help="write a JSON summary here")
    args = ap.parse_args()

    cols, rows = load_csv(args.csv)
    if not rows:
        print("No data rows found."); sys.exit(1)
    n = len(rows)
    profiles = [infer_column(c, rows) for c in cols]

    # duplicate rows (raw)
    seen = Counter(tuple(r.get(c) for c in cols) for r in rows)
    dup_count = sum(v - 1 for v in seen.values() if v > 1)

    X, feat_names = build_matrix(cols, rows, profiles,
                                 exclude={args.target} if args.target else frozenset())
    anomalous, scores, order = detect_anomalies(X)

    model_info = None
    if args.target and args.target in cols:
        y = [r.get(args.target) for r in rows]
        if len(set(y)) > 1:
            model_info = train_interpretable(X, y, feat_names)

    insights = make_insights(profiles, n, anomalous, model_info, dup_count)

    print("=" * 64)
    print(f" TabulaSage — offline AI profile  |  scikit-learn v{sklearn_version}")
    print(f" {args.csv}  |  {n} rows x {len(cols)} columns")
    print("=" * 64)
    print("\nCOLUMN PROFILE")
    for p in profiles:
        if p["type"] == "numeric":
            print(f"  • {p['name']:<20} numeric   mean={fmt_num(p['mean'])} "
                  f"std={fmt_num(p['std'])} min={fmt_num(p['min'])} "
                  f"median={fmt_num(p['median'])} max={fmt_num(p['max'])} "
                  f"missing={p['missing']}")
        else:
            print(f"  • {p['name']:<20} category  unique={p['unique']} "
                  f"top='{p['top_value']}'({p['top_freq']}) missing={p['missing']}")

    print(f"\nANOMALY DETECTION (IsolationForest)")
    print(f"  flagged {len(anomalous)} of {n} rows as anomalous")
    for i in order[:3]:
        tag = "ANOMALY" if i in anomalous else "normal "
        print(f"    [{tag}] row {i}: " +
              ", ".join(f"{c}={rows[i].get(c)}" for c in cols[:4]))

    if model_info:
        print(f"\nINTERPRETABLE MODEL (target = {args.target})")
        if "error" in model_info:
            print(f"  could not train: {model_info['error']}")
        else:
            print(f"  test accuracy: {model_info['accuracy']*100:.1f}%  "
                  f"(classes: {model_info['classes']})")
            print("  top drivers:")
            for name, imp in model_info["top_features"]:
                print(f"    - {name:<24} {imp:.3f}")

    print("\nPLAIN-ENGLISH INSIGHTS")
    for s in insights:
        print(f"  → {s}")
    print()

    if args.json_out:
        summary = {
            "file": args.csv, "rows": n, "columns": len(cols),
            "profiles": profiles, "anomalies": anomalous,
            "n_anomalies": len(anomalous), "duplicates": dup_count,
            "model": model_info, "insights": insights,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        print(f"JSON summary written to {args.json_out}")


if __name__ == "__main__":
    main()
