"""
test_model.py
=============
Try out the saved sustainability-relevance model. Keep this file in the same folder
as relevance_model.py and the model_out folder.

Three ways to use it:

  1) Interactive (type headlines, press Enter on an empty line to quit):
         python test_model.py

  2) A few headlines straight from the command line:
         python test_model.py --text "Delhi AQI turns severe" "Miami Heat beat Celtics"

  3) A CSV file (needs a title column). Writes <name>_predictions.csv.
     If the CSV also has a label column (1/0 or relevant/not_relevant) it prints
     accuracy, precision, recall, F1, ROC-AUC, a confusion matrix and the worst mistakes,
     so you can use it on validation_ground_truth.csv:
         python test_model.py --csv new_headlines.csv
         python test_model.py --csv data_validation/validation_ground_truth.csv

Options:
  --model-dir model_out     folder created by train_model.py
  --model best|logreg|nn    which saved model to use (default: the best one)
  --threshold 0.5           cut-off for "relevant" (default: the tuned value in the model card)
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from relevance_model import ID2LABEL, RelevanceClassifier


def to_binary(series):
    names = {"relevant": 1, "not_relevant": 0, "not relevant": 0, "positive": 1, "negative": 0,
             "true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
    s = series.astype(str).str.strip().str.lower().map(names)
    return s.fillna(pd.to_numeric(series, errors="coerce").map({1.0: 1, 0.0: 0}))


def show(results):
    for r in results:
        print(f"  {r['score']:.3f}  {r['label_name']:<13} {r['title']}")


def run_interactive(clf, threshold):
    print("Type a headline and press Enter (empty line to quit).")
    while True:
        try:
            text = input("headline> ").strip()
        except EOFError:
            break
        if not text:
            break
        show(clf.predict([text], threshold))


def run_csv(clf, args, threshold):
    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    if args.text_col not in df.columns:
        raise SystemExit(
            f"Column '{args.text_col}' not found. Columns are: {list(df.columns)}")
    texts = df[args.text_col].astype(str).tolist()
    scores = clf.predict_proba(texts)
    df["score"] = scores.round(4)
    df["pred_label"] = (scores >= threshold).astype(int)
    df["pred_name"] = df["pred_label"].map(ID2LABEL)
    out = Path(args.out) if args.out else Path(args.csv).with_name(
        Path(args.csv).stem + "_predictions.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Scored {len(df)} headlines: {int(df['pred_label'].sum())} relevant, "
          f"{int((df['pred_label'] == 0).sum())} not relevant")
    print(f"Saved: {out.resolve()}")

    if args.label_col not in df.columns:
        return
    y = to_binary(df[args.label_col])
    keep = y.notna()
    if keep.sum() == 0:
        print(
            f"(column '{args.label_col}' has no usable 1/0 labels, skipping metrics)")
        return
    y, pred, sc = y[keep].astype(
        int), df.loc[keep, "pred_label"], scores[keep.values]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f"\nMetrics on {len(y)} labelled rows (threshold {threshold:.3f}):")
    print(f"  accuracy  {accuracy_score(y, pred):.3f}")
    print(f"  precision {precision_score(y, pred, zero_division=0):.3f}")
    print(f"  recall    {recall_score(y, pred, zero_division=0):.3f}")
    print(f"  F1        {f1_score(y, pred, zero_division=0):.3f}")
    if y.nunique() == 2:
        print(f"  ROC-AUC   {roc_auc_score(y, sc):.3f}")
    print("\n  Confusion matrix (rows = actual, columns = predicted)")
    print("                   not_relevant  relevant")
    print(f"  not_relevant     {tn:>12}  {fp:>8}")
    print(f"  relevant         {fn:>12}  {tp:>8}")

    wrong = df.loc[keep].assign(actual=y)
    wrong = wrong[wrong["actual"] != wrong["pred_label"]]
    fps = wrong[wrong["actual"] == 0].sort_values(
        "score", ascending=False).head(10)
    fns = wrong[wrong["actual"] == 1].sort_values("score").head(10)
    if len(fps):
        print("\n  Most confident FALSE POSITIVES (predicted relevant, actually not):")
        for _, r in fps.iterrows():
            print(f"    {r['score']:.3f}  {r[args.text_col]}")
    if len(fns):
        print("\n  Most confident MISSES (predicted not relevant, actually relevant):")
        for _, r in fns.iterrows():
            print(f"    {r['score']:.3f}  {r[args.text_col]}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="model_out")
    ap.add_argument("--model", default="best",
                    choices=["best", "logreg", "nn"])
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--text", nargs="+", help="one or more headlines to score")
    ap.add_argument("--csv", help="CSV file of headlines to score")
    ap.add_argument("--text-col", default="title")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--out", help="where to save CSV predictions")
    args = ap.parse_args()

    clf = RelevanceClassifier.load(
        args.model_dir, None if args.model == "best" else args.model)
    threshold = clf.threshold if args.threshold is None else args.threshold
    print(f"Model: {clf.name}   threshold: {threshold:.3f}   "
          f"(relevant if score >= threshold)\n")

    if args.csv:
        run_csv(clf, args, threshold)
    elif args.text:
        show(clf.predict(args.text, threshold))
    else:
        run_interactive(clf, threshold)


if __name__ == "__main__":
    main()
