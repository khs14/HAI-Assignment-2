"""
evaluate_study.py
=================
Run the saved model(s) over the study dataset (validation_ground_truth.csv), then report
how well they do and where they go wrong. Keep this file in the same folder as
relevance_model.py, train_model.py and the model_out folder.

    python evaluate_study.py --data validation_ground_truth.csv

Needs the columns `title` and `label` (1 = relevant, 0 = not relevant). If the file also has
`tier`, `region` and `keyword` (or `base_query`) columns, the error breakdowns use them.

What it writes to --out (default: study_eval/):
  study_dataset_submission.csv   input title, model score + label, ground truth, correct?
                                 (this is the "study dataset file" for the assignment)
  metrics.csv / metrics.json     accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion counts
  error_breakdown.csv            error rate by tier, region, keyword and model confidence
  errors_to_tag.csv              every mistake with a suggested pattern (auto_pattern) and an empty
                                 manual_pattern column that you can fill in by hand
  error_distribution.csv         the error distribution table: pattern -> number and % of all errors
  charts: confusion matrices, ROC, precision-recall, score distribution, accuracy by tier,
          error distribution

Typical workflow for the error analysis:
  1. Run the script once.
  2. Open errors_to_tag.csv, read the mistakes, and write your own short pattern name in
     manual_pattern (blank = keep the suggested one).
  3. Run the script again. Your tags are kept and error_distribution.csv is rebuilt from them.

Options: --model-dir, --model (best|logreg|nn), --threshold, --out, --prevalence
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score, roc_auc_score

import train_model as tm
from relevance_model import ID2LABEL, RelevanceClassifier

# --------------------------------------------------------------------------- #
# Suggested error patterns (a starting point: edit them by hand in errors_to_tag.csv)
# --------------------------------------------------------------------------- #
ENV_RE = re.compile(
    r"climate|carbon|emission|renewable|solar|wind|pollut|plastic|recycl|waste|biodivers|deforest|"
    r"wildlife|conserv|coral|sustainab|green|energy|clean|net[- ]zero|hydrogen|electric|aqi|"
    r"air quality|water|drought|flood|heat ?wave|wildfire|glacier|sea level|ocean|forest|species|"
    r"ecosystem|fossil|coal|epa|environment", re.I)
INDIA_RE = re.compile(
    r"\b(delhi|india|indian|mumbai|chennai|kerala|punjab|ganga|ganges|namami|cpcb|ngt|green tribunal|"
    r"stubble|surya ghar|western ghats|project tiger|isro|bengaluru|bangalore|kolkata|hyderabad|"
    r"uttar pradesh|haryana|gujarat|maharashtra)\b", re.I)


def auto_pattern(title, actual, tier):
    if actual == 0:  # false positive: model said relevant, truth is not
        if tier == "hard":
            return "Ambiguous word used in a non-environmental sense"
        if tier == "adjacent":
            return "Adjacent topic (energy markets, business, social or governance)"
        if tier == "easy":
            return "General news misclassified"
        return ("Environment-related word used in another sense" if ENV_RE.search(title)
                else "No obvious cause")
    # false negative: model said not relevant, truth is relevant
    if ENV_RE.search(title):
        return "Environmental keyword present but scored low"
    if INDIA_RE.search(title):
        return "India-specific place or term without a common environmental word"
    return "No obvious environmental keyword (implicit topic)"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def bootstrap_ci(y, pred, n=1000, seed=0):
    """95% bootstrap confidence intervals for accuracy, precision, recall and F1."""
    rng = np.random.RandomState(seed)
    out = {k: [] for k in ("accuracy", "precision", "recall", "f1")}
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        yy, pp = y[i], pred[i]
        tp = int(((yy == 1) & (pp == 1)).sum())
        fp = int(((yy == 0) & (pp == 1)).sum())
        fn = int(((yy == 1) & (pp == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out["accuracy"].append((yy == pp).mean())
        out["precision"].append(prec)
        out["recall"].append(rec)
        out["f1"].append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def bootstrap_auc_ci(y, score, n=1000, seed=0):
    """95% bootstrap confidence intervals for ROC-AUC and PR-AUC (resampling the trials)."""
    rng = np.random.RandomState(seed)
    roc, pr = [], []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():  # needs both classes
            continue
        roc.append(roc_auc_score(y[i], score[i]))
        pr.append(average_precision_score(y[i], score[i]))
    return {"roc_auc": (float(np.percentile(roc, 2.5)), float(np.percentile(roc, 97.5))),
            "pr_auc": (float(np.percentile(pr, 2.5)), float(np.percentile(pr, 97.5)))}


def precision_at_prevalence(recall, specificity, prevalence):
    """Expected precision if only `prevalence` of a real feed is relevant."""
    tp = recall * prevalence
    fp = (1 - specificity) * (1 - prevalence)
    return tp / (tp + fp) if tp + fp else 0.0


def read_csv_any(path, **kw):
    """Read a CSV saved as UTF-8 (with or without BOM) or as Excel's default Windows encoding."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", **kw)


def breakdown(df, dim, col):
    g = df.groupby(col)["error"].agg(n="size", errors="sum").reset_index()
    g.insert(0, "dimension", dim)
    g = g.rename(columns={col: "group"})
    g["error_rate_pct"] = (100 * g["errors"] / g["n"]).round(1)
    g["share_of_all_errors_pct"] = (100 * g["errors"] / max(int(df["error"].sum()), 1)).round(1)
    return g


def plot_error_distribution(dist, path):
    d = dist.sort_values("errors")
    fig, ax = plt.subplots(figsize=(9, 0.9 + 0.55 * len(d)))
    colors = [tm.ORANGE if t == "False positive" else tm.BLUE for t in d["error_type"]]
    y = np.arange(len(d))
    ax.barh(y, d["pct_of_all_errors"], color=colors, height=0.6, edgecolor=tm.SURFACE)
    for yi, (pct, n) in enumerate(zip(d["pct_of_all_errors"], d["errors"])):
        ax.text(pct + 0.8, yi, f"{pct:.1f}%  (n={n})", va="center", fontsize=9, color=tm.INK)
    ax.set_yticks(y, [textwrap_label(p) for p in d["pattern"]])
    ax.set_xlim(0, max(d["pct_of_all_errors"].max() * 1.25, 10))
    ax.set_xlabel("Share of all model errors (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Error distribution by pattern", loc="left")
    ax.legend(handles=[Line2D([], [], color=tm.ORANGE, lw=8, label="False positive (predicted relevant)"),
                       Line2D([], [], color=tm.BLUE, lw=8, label="False negative (predicted not relevant)")],
              loc="lower right")
    tm.save_fig(fig, path)


def textwrap_label(text, width=46):
    import textwrap
    return "\n".join(textwrap.wrap(text, width))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="study dataset CSV, e.g. validation_ground_truth.csv")
    ap.add_argument("--model-dir", default=str(here / "model_out"))
    ap.add_argument("--model", default="best", choices=["best", "logreg", "nn"],
                    help="model used for the submission file and the error analysis")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the saved (tuned) threshold for the error analysis")
    ap.add_argument("--out", default="study_eval")
    ap.add_argument("--text-col", default="title")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--prevalence", type=float, default=0.10,
                    help="share of relevant headlines in a real news feed, for the precision estimate")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tm.set_style()

    # ---- data ----------------------------------------------------------------
    raw = read_csv_any(args.data)
    for col in (args.text_col, args.label_col):
        if col not in raw.columns:
            raise SystemExit(f"Column '{col}' not found. Columns are: {list(raw.columns)}")
    df = pd.DataFrame({
        "id": raw["id"].astype(str) if "id" in raw.columns else [f"S{i + 1:03d}" for i in range(len(raw))],
        "title": raw[args.text_col].astype(str).str.strip(),
        "truth": tm.to_binary(raw[args.label_col]),
    })
    for name, cands in (("tier", ["tier"]), ("region", ["region"]),
                        ("keyword", ["keyword", "base_query", "query"])):
        col = next((c for c in cands if c in raw.columns), None)
        if col:
            df[name] = raw[col].astype(str)
    bad = df["truth"].isna() | df["title"].isin(["", "nan"])
    if bad.any():
        print(f"Dropping {int(bad.sum())} rows with an empty title or an unusable label")
    df = df[~bad].reset_index(drop=True)
    df["truth"] = df["truth"].astype(int)
    y = df["truth"].values
    print(f"Study dataset: {len(df)} headlines, {int(y.sum())} relevant, {int((y == 0).sum())} not relevant")
    if len(df) < 200:
        print("Warning: the assignment asks for at least 200 trials")

    # ---- model scores ---------------------------------------------------------
    card = json.loads((Path(args.model_dir) / "model_card.json").read_text(encoding="utf-8"))
    names = list(card["models"])
    chosen = card["best_model"] if args.model == "best" else args.model
    if chosen not in names:
        raise SystemExit(f"Model '{chosen}' is not in {args.model_dir}. Available: {names}")
    scores, thr, precision_target = {}, {}, card.get("precision_target", 0.95)
    for n in names:
        clf = RelevanceClassifier.load(args.model_dir, n)
        scores[n] = clf.predict_proba(df["title"].tolist())
        thr[n] = clf.threshold if args.threshold is None else args.threshold
    print(f"Models: {', '.join(names)}.  Used for the submission and error analysis: {chosen} "
          f"(threshold {thr[chosen]:.3f})\n")

    # ---- metrics for every model ----------------------------------------------
    default, tuned, aucs, aps, rows = {}, {}, {}, {}, []
    for n in names:
        default[n] = tm.metrics_at(y, scores[n], 0.5)
        tuned[n] = tm.metrics_at(y, scores[n], thr[n])
        aucs[n] = float(roc_auc_score(y, scores[n]))
        aps[n] = float(average_precision_score(y, scores[n]))
        for label, m in (("0.50 (default)", default[n]), (f"{thr[n]:.2f} (saved)", tuned[n])):
            rows.append({"model": tm.MODEL_NAMES[n], "threshold": label, "accuracy": m["accuracy"],
                         "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                         "roc_auc": aucs[n], "pr_auc": aps[n],
                         "TN": m["tn"], "FP": m["fp"], "FN": m["fn"], "TP": m["tp"]})
    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False)
    print("=== Study-set results ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    m = tuned[chosen]
    ci = bootstrap_ci(y, (scores[chosen] >= thr[chosen]).astype(int))
    ci_auc = bootstrap_auc_ci(y, scores[chosen])
    ci.update(ci_auc)
    p_prev = precision_at_prevalence(m["recall"], m["specificity"], args.prevalence)
    print(f"\n{tm.MODEL_NAMES[chosen]} at threshold {thr[chosen]:.3f} (95% bootstrap intervals):")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"  {k:<9} {m[k]:.3f}   [{ci[k][0]:.3f}, {ci[k][1]:.3f}]")
    print(f"  ROC-AUC   {aucs[chosen]:.3f}   [{ci['roc_auc'][0]:.3f}, {ci['roc_auc'][1]:.3f}]")
    print(f"  PR-AUC    {aps[chosen]:.3f}   [{ci['pr_auc'][0]:.3f}, {ci['pr_auc'][1]:.3f}]")
    print(f"  Precision target was {precision_target:.0%}; "
          f"{'reached' if m['precision'] >= precision_target else 'NOT reached'} on this set.")
    print(f"  Estimated precision if only {args.prevalence:.0%} of a real feed is relevant: {p_prev:.3f}")
    print("\n  Confusion matrix (rows = actual, columns = predicted)")
    print("                   not_relevant  relevant")
    print(f"  not_relevant     {m['tn']:>12}  {m['fp']:>8}")
    print(f"  relevant         {m['fn']:>12}  {m['tp']:>8}")
    (out / "metrics.json").write_text(json.dumps({
        "model": chosen, "threshold": thr[chosen], "n": len(df), "metrics": m,
        "confidence_intervals_95": ci, "roc_auc": aucs[chosen], "pr_auc": aps[chosen],
        "estimated_precision_at_prevalence": {"prevalence": args.prevalence, "precision": p_prev},
        "all_models": table.to_dict(orient="records")}, indent=2), encoding="utf-8")

    # ---- per-headline results ---------------------------------------------------
    df["model_score"] = scores[chosen].round(4)
    df["model_label"] = (scores[chosen] >= thr[chosen]).astype(int)
    df["correct"] = (df["model_label"] == df["truth"]).astype(int)
    df["error"] = 1 - df["correct"]
    df["error_type"] = np.where(df["error"] == 0, "",
                                np.where(df["model_label"] == 1, "False positive", "False negative"))
    df["model_confident"] = np.where((df["model_score"] <= 0.2) | (df["model_score"] >= 0.8),
                                     "confident (score <=0.2 or >=0.8)", "uncertain (0.2-0.8)")
    sub = pd.DataFrame({
        "id": df["id"], "title": df["title"],
        "ground_truth": df["truth"], "ground_truth_name": df["truth"].map(ID2LABEL),
        "model_score": df["model_score"], "model_label": df["model_label"],
        "model_label_name": df["model_label"].map(ID2LABEL), "correct": df["correct"]})
    for extra in ("tier", "region", "keyword"):
        if extra in df.columns:
            sub[extra] = df[extra]
    for n in names:
        sub[f"score_{n}"] = scores[n].round(4)
    sub.to_csv(out / "study_dataset_submission.csv", index=False, encoding="utf-8-sig")

    # ---- error breakdowns ---------------------------------------------------------
    n_err = int(df["error"].sum())
    print(f"\n=== Errors: {n_err} of {len(df)} ({100 * n_err / len(df):.1f}%) ===")
    print(f"  false positives (predicted relevant, actually not): {int((df['error_type'] == 'False positive').sum())}")
    print(f"  false negatives (predicted not relevant, actually relevant): {int((df['error_type'] == 'False negative').sum())}")
    parts = []
    for dim in ("tier", "region", "keyword", "model_confident"):
        if dim in df.columns:
            parts.append(breakdown(df, dim, dim))
    bd = pd.concat(parts, ignore_index=True)
    kw = bd[bd["dimension"] == "keyword"]
    keep_kw = kw[kw["errors"] > 0].sort_values(["errors", "error_rate_pct"], ascending=False).head(20)
    bd = pd.concat([bd[bd["dimension"] != "keyword"], keep_kw], ignore_index=True)
    bd.to_csv(out / "error_breakdown.csv", index=False)
    for dim in ("tier", "region", "model_confident"):
        part = bd[bd["dimension"] == dim]
        if len(part):
            print(f"\n  By {dim}:")
            print(part[["group", "n", "errors", "error_rate_pct", "share_of_all_errors_pct"]]
                  .to_string(index=False))
    if len(keep_kw):
        print("\n  Keywords with the most errors (top 10):")
        print(keep_kw.head(10)[["group", "n", "errors", "error_rate_pct"]].to_string(index=False))
        if kw["n"].median() < 5:
            print(f"  (each keyword has only about {int(kw['n'].median())} headlines, so keyword-level "
                  f"error rates are noisy; look for repeated themes rather than single keywords)")
    near = int(((df["error"] == 1) & ((df["model_score"] - thr[chosen]).abs() <= 0.15)).sum())
    if n_err:
        print(f"\n  {near} of {n_err} errors ({100 * near / n_err:.0f}%) scored within 0.15 of the threshold, "
              f"so they are borderline cases (some may be ambiguous labels).")

    # ---- errors to tag + error distribution table ----------------------------------
    err = df[df["error"] == 1].copy()
    err["auto_pattern"] = [auto_pattern(t, a, tier) for t, a, tier in
                           zip(err["title"], err["truth"], err["tier"] if "tier" in err else [None] * len(err))]
    err["near_threshold"] = ((err["model_score"] - thr[chosen]).abs() <= 0.15).map({True: "yes", False: ""})
    err["manual_pattern"] = ""
    tag_path = out / "errors_to_tag.csv"
    kept = 0
    if tag_path.exists():  # carry over patterns you already typed
        old = read_csv_any(tag_path, dtype=str).fillna("")
        prev = dict(zip(old["id"], old["manual_pattern"]))
        err["manual_pattern"] = err["id"].map(prev).fillna("")
        kept = int((err["manual_pattern"] != "").sum())
    err["pattern"] = np.where(err["manual_pattern"].str.strip() != "", err["manual_pattern"].str.strip(),
                              err["auto_pattern"])
    cols = ["id", "title", "error_type", "truth", "model_score", "near_threshold"]
    cols += [c for c in ("tier", "region", "keyword") if c in err.columns]
    err.sort_values(["error_type", "model_score"], ascending=[True, False])[
        cols + ["auto_pattern", "manual_pattern"]].rename(columns={"truth": "actual_label"}).to_csv(
        tag_path, index=False, encoding="utf-8-sig")

    if n_err:
        dist = (err.groupby(["error_type", "pattern"]).size().reset_index(name="errors")
                .sort_values("errors", ascending=False))
        dist["pct_of_all_errors"] = (100 * dist["errors"] / n_err).round(1)
        dist["pct_of_error_type"] = (100 * dist["errors"] /
                                     dist.groupby("error_type")["errors"].transform("sum")).round(1)
        dist.to_csv(out / "error_distribution.csv", index=False)
        print("\n=== Error distribution table ===")
        print(dist.to_string(index=False))
        n_auto = int((err["manual_pattern"].str.strip() == "").sum())
        print(f"\n  {n_auto} of {n_err} errors still use the suggested (automatic) pattern; "
              f"{kept} use your manual tags.")
        print(f"  To refine: edit manual_pattern in {tag_path} and run this script again.")
        plot_error_distribution(dist, out / "error_distribution.png")

    # ---- charts -------------------------------------------------------------------
    for n in names:
        tm.plot_confusions(n, default[n], tuned[n], precision_target, out / f"confusion_matrix_{n}.png")
    tm.plot_roc(y, scores, tuned, aucs, out / "roc_curve.png")
    tm.plot_pr(y, scores, tuned, aps, precision_target, out / "precision_recall_curve.png")
    tm.plot_score_distribution(y, scores[chosen], tuned[chosen], chosen, out / "score_distribution.png")
    if "tier" in df.columns:
        tm.plot_tiers(df[["tier"]].assign(correct=df["correct"] == 1), chosen, out / "accuracy_by_tier.png")

    print(f"\nEverything is in: {out.resolve()}")
    print("Submission file for the assignment: study_dataset_submission.csv")


if __name__ == "__main__":
    main()