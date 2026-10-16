"""
train_model.py
==============
Train, evaluate and save a "sustainability relevant / not relevant" headline classifier.

Only the `title` and `label` columns are needed (label: 1 = relevant, 0 = not relevant).
If the CSV also has a keyword column (`base_query` or `query`) it is used to split the
data BY KEYWORD, so the test set contains keywords the model never saw (an honest,
stricter estimate). A `tier` column, if present, adds an accuracy-by-tier chart.

Usage (local or Colab; needs numpy, scipy, pandas, scikit-learn, matplotlib, joblib):
    python train_model.py --data small_combined.csv --out model_out

Outputs in --out:
    logreg/ , nn/            saved models
    model_card.json          best model, thresholds, metrics, data summary
    eval/                    metrics.json/csv, test_predictions.csv and the charts
                             (confusion matrix, ROC, precision-recall, training curve,
                              score distribution, accuracy by tier)
Load later with:  RelevanceClassifier.load("model_out")   (see relevance_model.py)
"""
import argparse
import json
import platform
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score, f1_score,
                             log_loss, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import relevance_model as rm

MODEL_NAMES = {"logreg": "TF-IDF + logistic regression", "nn": "From-scratch neural net"}

# chart palette (validated categorical slots 1-2, chart chrome)
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
MODEL_COLORS = {"logreg": BLUE, "nn": ORANGE}
CLASS_COLORS = {1: BLUE, 0: ORANGE}
BLUE_RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#256abf", "#0d366b"])


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def to_binary(series: pd.Series) -> pd.Series:
    names = {"relevant": 1, "not_relevant": 0, "not relevant": 0, "positive": 1,
             "negative": 0, "true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
    s = series.astype(str).str.strip().str.lower().map(names)
    num = pd.to_numeric(series, errors="coerce").map({1.0: 1, 0.0: 0})
    return s.fillna(num)


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(t).lower()).strip()


def load_data(args):
    raw = pd.read_csv(args.data, encoding="utf-8-sig")
    for col in (args.text_col, args.label_col):
        if col not in raw.columns:
            raise SystemExit(f"Column '{col}' not found. Columns are: {list(raw.columns)}")

    df = pd.DataFrame({"title": raw[args.text_col].astype(str).str.strip(),
                       "label": to_binary(raw[args.label_col])})
    group_col = args.group_col
    if group_col == "auto":
        group_col = next((c for c in ("base_query", "query", "keyword") if c in raw.columns), None)
    elif group_col == "none":
        group_col = None
    if group_col:
        df["group"] = raw[group_col].astype(str)
    if "tier" in raw.columns:
        df["tier"] = raw["tier"].astype(str)

    n0 = len(df)
    df = df[df["title"].ne("") & df["title"].ne("nan") & df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    df["norm"] = df["title"].map(norm_title)
    clash = df.groupby("norm")["label"].transform("nunique") > 1  # same title, both labels
    df = df[~clash].drop_duplicates("norm").drop(columns="norm").reset_index(drop=True)
    print(f"Loaded {n0} rows -> {len(df)} after dropping empty/unlabelled, "
          f"duplicate titles and titles with conflicting labels")
    print(f"  relevant (1): {int(df['label'].sum())}   not relevant (0): {int((df['label'] == 0).sum())}")
    if len(df) < 200 or df["label"].nunique() < 2:
        raise SystemExit("Need at least 200 rows covering both labels.")
    return df, group_col


def grouped_split(df, test_size, seed):
    """Group-aware split that also keeps the class balance close to the full data."""
    y = df["label"].values
    best = None
    for s in range(seed, seed + 300):
        tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=s)
                      .split(df, y, df["group"].values))
        score = abs(y[te].mean() - y.mean()) + abs(len(te) / len(df) - test_size)
        if best is None or score < best[0]:
            best = (score, tr, te)
        if score < 0.02:
            break
    return best[1], best[2]


def split_data(df, args):
    use_groups = ("group" in df.columns and args.split == "group" and df["group"].nunique() >= 10)
    if args.split == "group" and not use_groups:
        print("  (no usable keyword column, falling back to a stratified random split)")
    if use_groups:
        rest, test_idx = grouped_split(df, args.test_size, args.seed)
        sub = df.iloc[rest].reset_index(drop=True)
        tr_i, va_i = grouped_split(sub, args.val_size / (1 - args.test_size), args.seed + 1)
        train, val, test = sub.iloc[tr_i], sub.iloc[va_i], df.iloc[test_idx]
        how = "by keyword (test keywords are unseen in training)"
    else:
        rest, test = train_test_split(df, test_size=args.test_size, stratify=df["label"],
                                      random_state=args.seed)
        train, val = train_test_split(rest, test_size=args.val_size / (1 - args.test_size),
                                      stratify=rest["label"], random_state=args.seed)
        how = "stratified random (optimistic: similar headlines can land in both sets)"
    train, val, test = (x.reset_index(drop=True) for x in (train, val, test))
    print(f"Split {how}:")
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"  {name:5s} {len(part):5d} rows, {part['label'].mean():.1%} relevant")
    return train, val, test, how


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def pick_threshold(y, scores, precision_target):
    """Lowest threshold whose validation precision >= target (max recall under that)."""
    prec, rec, thr = precision_recall_curve(y, scores)
    ok = np.where(prec[:-1] >= precision_target)[0]
    if len(ok):
        best_recall = rec[:-1][ok].max()
        cand = ok[rec[:-1][ok] >= best_recall - 1e-12]
        return float(thr[cand].max()), True
    f05 = (1.25 * prec[:-1] * rec[:-1]) / np.maximum(0.25 * prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f05))]), False


def metrics_at(y, scores, thr):
    pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(thr), "accuracy": float((tp + tn) / len(y)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "f0_5": float(fbeta_score(y, pred, beta=0.5, zero_division=0)),
            "specificity": float(tn / max(tn + fp, 1)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.family": "sans-serif", "legend.frameon": False,
        "lines.linewidth": 2.0, "legend.fontsize": 9,
    })


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_confusion(ax, m, title):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(cm, cmap=BLUE_RAMP, vmin=0, vmax=max(cm.max(), 1))
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / max(cm[i].sum(), 1)
            dark = cm[i, j] > 0.5 * cm.max()
            ax.text(j, i, f"{cm[i, j]}\n{pct:.1%} of row", ha="center", va="center", fontsize=11,
                    color="white" if dark else INK)
    ax.set_xticks([0, 1], ["Not relevant", "Relevant"])
    ax.set_yticks([0, 1], ["Not relevant", "Relevant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, loc="left", fontsize=10.5)


def plot_confusions(name, m_default, m_tuned, target, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    draw_confusion(axes[0], m_default, f"Threshold {m_default['threshold']:.2f} (default)")
    draw_confusion(axes[1], m_tuned,
                   f"Threshold {m_tuned['threshold']:.2f} (tuned)\nchosen on validation for precision ≥ {target:.0%}")
    fig.suptitle(f"Confusion matrix on the test set: {MODEL_NAMES[name]}", x=0.02, ha="left",
                 fontweight="bold")
    save_fig(fig, path)


def plot_roc(test_y, scores, tuned, aucs, path):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1.2)
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(test_y, s)
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], label=f"{MODEL_NAMES[name]} (AUC {aucs[name]:.3f})")
        m = tuned[name]
        ax.plot(1 - m["specificity"], m["recall"], "o", ms=9, color=MODEL_COLORS[name],
                mec=SURFACE, mew=2)
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [
        Line2D([], [], marker="o", ls="", color=INK2, mec=SURFACE, ms=8, label="Tuned threshold")],
        loc="lower right")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve (test set)", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    save_fig(fig, path)


def plot_pr(test_y, scores, tuned, aps, target, path):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.axhline(target, ls="--", color=MUTED, lw=1.2)
    ax.text(0.02, target + 0.008, f"precision target {target:.0%}", color=INK2, fontsize=9)
    ax.axhline(test_y.mean(), ls=":", color=MUTED, lw=1.2)
    ax.text(0.02, test_y.mean() + 0.008, "no-skill baseline", color=INK2, fontsize=9)
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(test_y, s)
        ax.plot(r, p, color=MODEL_COLORS[name], label=f"{MODEL_NAMES[name]} (AP {aps[name]:.3f})")
        m = tuned[name]
        ax.plot(m["recall"], m["precision"], "o", ms=9, color=MODEL_COLORS[name], mec=SURFACE, mew=2)
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [
        Line2D([], [], marker="o", ls="", color=INK2, mec=SURFACE, ms=8, label="Tuned threshold")],
        loc="lower left")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (test set)", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    save_fig(fig, path)


def plot_training(hist, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ep = hist["epoch"]
    axes[0].plot(ep, hist["train_loss"], color=BLUE, label="Training")
    axes[0].plot(ep, hist["val_loss"], color=ORANGE, label="Validation")
    axes[0].axvline(hist["best_epoch"], color=MUTED, ls="--", lw=1.2)
    axes[0].text(hist["best_epoch"], axes[0].get_ylim()[1], " best epoch", color=INK2, fontsize=9, va="top")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Log loss")
    axes[0].set_title("Loss per epoch", loc="left")
    axes[0].legend()
    axes[1].plot(ep, hist["val_auc"], color=ORANGE, label="Validation AUC")
    axes[1].axvline(hist["best_epoch"], color=MUTED, ls="--", lw=1.2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC AUC")
    axes[1].set_title("Validation AUC per epoch", loc="left")
    axes[1].legend(loc="lower right")
    for a in axes:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    save_fig(fig, path)


def plot_score_distribution(test_y, scores, m_tuned, name, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bins = np.linspace(0, 1, 26)
    for cls, label in ((1, "Actually relevant"), (0, "Actually not relevant")):
        ax.hist(scores[test_y == cls], bins=bins, color=CLASS_COLORS[cls], alpha=0.85,
                edgecolor=SURFACE, linewidth=1, label=label)
    top = ax.get_ylim()[1]
    ax.axvline(0.5, color=MUTED, ls="--", lw=1.2)
    ax.text(0.5, top * 0.98, " 0.50", color=INK2, fontsize=9, va="top")
    ax.axvline(m_tuned["threshold"], color=INK, lw=1.6)
    ax.text(m_tuned["threshold"], top * 0.90, f" tuned {m_tuned['threshold']:.2f}", color=INK, fontsize=9, va="top")
    ax.set_xlabel("Predicted probability of 'relevant'")
    ax.set_ylabel("Headlines")
    ax.set_title(f"Score distribution on the test set: {MODEL_NAMES[name]}", loc="left")
    ax.legend(loc="upper right")
    save_fig(fig, path)


def plot_tiers(tier_df, name, path):
    order = [t for t in ("positive", "easy", "adjacent", "hard") if t in set(tier_df["tier"])]
    order += [t for t in tier_df["tier"].unique() if t not in order]
    rows = [(t, (tier_df.loc[tier_df["tier"] == t, "correct"]).mean(), int((tier_df["tier"] == t).sum()))
            for t in order]
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.7 * len(rows)))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=BLUE, height=0.55, edgecolor=SURFACE)
    for yi, (t, v, n) in zip(y, rows):
        ax.text(v + 0.01, yi, f"{v:.1%}", va="center", color=INK, fontsize=10)
    ax.set_yticks(y, [f"{t}  (n={n})" for t, _, n in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of headlines classified correctly")
    ax.set_title(f"Accuracy by headline type: {MODEL_NAMES[name]}", loc="left")
    save_fig(fig, path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
DEMO = [
    "Delhi AQI turns severe as stubble burning rises",
    "Solar and wind overtake coal in the power mix for the first time",
    "Company unveils science-based net zero targets and scope 3 plan",
    "Miami Heat edge Celtics in overtime thriller",
    "Sensex ends higher as RBI holds repo rate steady",
    "Political climate stays tense ahead of the election",
    "Union workers vote to strike over pay and safety",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV with at least title and label columns")
    ap.add_argument("--out", default="model_out")
    ap.add_argument("--text-col", default="title")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--group-col", default="auto", help="keyword column for grouped split; 'none' to disable")
    ap.add_argument("--split", choices=["group", "random"], default="group")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--precision-target", type=float, default=0.95,
                    help="threshold is tuned on validation to reach this precision")
    ap.add_argument("--models", default="logreg,nn", help="comma list from: logreg, nn")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--word-dropout", type=float, default=0.1)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    eval_dir = out / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    names = [m.strip() for m in args.models.split(",") if m.strip()]

    df, group_col = load_data(args)
    train, val, test, split_desc = split_data(df, args)
    Xtr, ytr = train["title"].tolist(), train["label"].values
    Xva, yva = val["title"].tolist(), val["label"].values
    Xte, yte = test["title"].tolist(), test["label"].values

    scorers, hyper, hist = {}, {}, None

    if "logreg" in names:
        print("\nTraining TF-IDF + logistic regression ...")
        best = None
        for C in (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64):
            pipe = rm.build_logreg(C).fit(Xtr, ytr)
            ll = log_loss(yva, np.clip(pipe.predict_proba(Xva)[:, 1], 1e-6, 1 - 1e-6))
            print(f"    C={C:<5} validation log loss {ll:.4f}")
            if best is None or ll < best[0]:
                best = (ll, C, pipe)
        scorers["logreg"] = rm.SklearnScorer(best[2])
        hyper["logreg"] = {"C": best[1]}
        print(f"  chose C={best[1]}")

    if "nn" in names:
        print("\nTraining the from-scratch neural net (NumPy) ...")
        scorer, hist = rm.NNScorer.train(
            Xtr, ytr, Xva, yva, emb_dim=args.emb_dim, hidden=args.hidden, seed=args.seed,
            epochs=args.epochs, lr=args.lr, dropout=args.dropout,
            word_dropout=args.word_dropout, l2=args.l2, patience=args.patience)
        scorers["nn"] = scorer
        hyper["nn"] = {"emb_dim": args.emb_dim, "hidden": args.hidden, "lr": args.lr,
                       "dropout": args.dropout, "word_dropout": args.word_dropout, "l2": args.l2,
                       "vocab_size": len(scorer.vocab.features), "best_epoch": hist["best_epoch"]}

    # ---- scores, thresholds, metrics -------------------------------------
    val_scores = {n: s.predict_proba(Xva) for n, s in scorers.items()}
    test_scores = {n: s.predict_proba(Xte) for n, s in scorers.items()}
    card_models, tuned, default, aucs, aps, rows = {}, {}, {}, {}, {}, []
    for n in scorers:
        thr, reached = pick_threshold(yva, val_scores[n], args.precision_target)
        if not reached:
            print(f"  note: {n} could not reach precision {args.precision_target:.0%} on validation; "
                  f"used the best F0.5 threshold instead")
        default[n] = metrics_at(yte, test_scores[n], 0.5)
        tuned[n] = metrics_at(yte, test_scores[n], thr)
        aucs[n] = float(roc_auc_score(yte, test_scores[n]))
        aps[n] = float(average_precision_score(yte, test_scores[n]))
        card_models[n] = {
            "kind": n, "name": MODEL_NAMES[n], "threshold": thr,
            "precision_target_reached_on_validation": bool(reached),
            "hyperparameters": hyper[n],
            "validation": {"roc_auc": float(roc_auc_score(yva, val_scores[n])),
                           "pr_auc": float(average_precision_score(yva, val_scores[n]))},
            "test": {"roc_auc": aucs[n], "pr_auc": aps[n],
                     "at_0.5": default[n], "at_tuned_threshold": tuned[n]},
        }
        for label, m in (("default 0.5", default[n]), (f"tuned {thr:.2f}", tuned[n])):
            rows.append({"model": MODEL_NAMES[n], "threshold": label,
                         "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
                         "f1": m["f1"], "roc_auc": aucs[n], "pr_auc": aps[n],
                         "TN": m["tn"], "FP": m["fp"], "FN": m["fn"], "TP": m["tp"]})

    best_name = max(scorers, key=lambda n: card_models[n]["validation"]["pr_auc"])
    table = pd.DataFrame(rows)
    print("\n=== Test-set results ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nBest model (highest validation PR-AUC): {MODEL_NAMES[best_name]}")

    # ---- save models, card, tables ---------------------------------------
    for n, s in scorers.items():
        s.save(out / n)
    card = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "task": "binary classification: is a news headline relevant to environmental sustainability",
        "labels": rm.ID2LABEL, "best_model": best_name, "precision_target": args.precision_target,
        "pretrained_weights_used": False, "split": split_desc,
        "data": {"rows": len(df), "train": len(train), "val": len(val), "test": len(test),
                 "share_relevant": float(df["label"].mean()), "group_column": group_col},
        "models": card_models,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "pandas": pd.__version__},
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    table.to_csv(eval_dir / "metrics.csv", index=False)
    (eval_dir / "metrics.json").write_text(json.dumps(card["models"], indent=2), encoding="utf-8")

    preds = pd.DataFrame({"title": Xte, "true_label": yte})
    for n in scorers:
        preds[f"score_{n}"] = test_scores[n].round(4)
    preds["pred_best"] = (test_scores[best_name] >= tuned[best_name]["threshold"]).astype(int)
    for extra in ("tier", "group"):
        if extra in test.columns:
            preds[extra] = test[extra].values
    preds.to_csv(eval_dir / "test_predictions.csv", index=False)

    # ---- charts ------------------------------------------------------------
    for n in scorers:
        plot_confusions(n, default[n], tuned[n], args.precision_target, eval_dir / f"confusion_matrix_{n}.png")
    plot_roc(yte, test_scores, tuned, aucs, eval_dir / "roc_curve.png")
    plot_pr(yte, test_scores, tuned, aps, args.precision_target, eval_dir / "precision_recall_curve.png")
    if hist is not None:
        plot_training(hist, eval_dir / "training_curve_nn.png")
    plot_score_distribution(yte, test_scores[best_name], tuned[best_name], best_name,
                            eval_dir / "score_distribution.png")
    if "tier" in test.columns:
        tdf = pd.DataFrame({"tier": test["tier"].values,
                            "correct": preds["pred_best"].values == yte})
        plot_tiers(tdf, best_name, eval_dir / "accuracy_by_tier.png")
        print("\nAccuracy by tier (best model, tuned threshold):")
        print(tdf.groupby("tier")["correct"].agg(["mean", "size"]).rename(
            columns={"mean": "accuracy", "size": "n"}).to_string(float_format=lambda v: f"{v:.3f}"))

    # ---- reload check + demo ----------------------------------------------
    clf = rm.RelevanceClassifier.load(out)
    same = np.allclose(clf.predict_proba(Xte), test_scores[best_name], atol=1e-9)
    print(f"\nReload check: {'passed' if same else 'FAILED'} (saved model reproduces the test scores)")
    print(f"\nDemo with '{clf.name}' (threshold {clf.threshold:.2f}):")
    for r in clf.predict(DEMO):
        print(f"  {r['score']:.3f}  {r['label_name']:<13} {r['title']}")
    print(f"\nEverything is in: {out.resolve()}")


if __name__ == "__main__":
    main()