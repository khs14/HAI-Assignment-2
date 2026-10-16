"""
relevance_model.py
==================
From-scratch text classifier: is a news headline RELEVANT to environmental
sustainability (label 1) or NOT RELEVANT (label 0)?

No pretrained weights are used anywhere. This one file holds everything needed at
inference time, so a web portal can simply do:

    from relevance_model import RelevanceClassifier
    clf = RelevanceClassifier.load("model_out")
    clf.predict(["Delhi AQI turns severe as stubble burning rises"])

Two models live here:
  * "logreg": TF-IDF (word + character n-grams) -> logistic regression (scikit-learn)
  * "nn"    : fastText-style neural net written in NumPy with hand-coded
              backpropagation and Adam. Mean of learned word/bigram embeddings ->
              ReLU hidden layer -> sigmoid. Random initialisation, trained only on
              your headlines. Inference needs only numpy + scipy.

Command line quick test:
    python relevance_model.py model_out "headline one" "headline two"
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ID2LABEL = {0: "not_relevant", 1: "relevant"}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


# --------------------------------------------------------------------------- #
# Text processing
# --------------------------------------------------------------------------- #
def tokenize(text) -> list[str]:
    """Lower-case word tokenizer (curly apostrophes normalised)."""
    return _TOKEN_RE.findall(str(text).replace("’", "'").lower())


def ngram_features(text) -> set[str]:
    """Distinct unigrams and bigrams of a headline."""
    toks = tokenize(text)
    feats = set(toks)
    feats.update(f"{a} {b}" for a, b in zip(toks, toks[1:]))
    return feats


class NGramVocab:
    """Vocabulary of word unigrams + bigrams (document frequency >= min_count)."""

    def __init__(self, min_count: int = 2, max_features: int = 50000):
        self.min_count = min_count
        self.max_features = max_features
        self.features: list[str] = []
        self.index: dict[str, int] = {}

    def fit(self, texts):
        df = Counter()
        for t in texts:
            df.update(ngram_features(t))
        kept = [(f, c) for f, c in df.items() if c >= self.min_count]
        kept.sort(key=lambda fc: (-fc[1], fc[0]))
        self.set_features([f for f, _ in kept[: self.max_features]])
        return self

    def set_features(self, features):
        self.features = list(features)
        self.index = {f: i for i, f in enumerate(self.features)}
        return self

    def transform(self, texts) -> sp.csr_matrix:
        rows, cols = [], []
        n = 0
        for r, t in enumerate(texts):
            n += 1
            for f in ngram_features(t):
                j = self.index.get(f)
                if j is not None:
                    rows.append(r)
                    cols.append(j)
        data = np.ones(len(rows), dtype=np.float64)
        return sp.csr_matrix((data, (rows, cols)), shape=(n, len(self.features)))


def _row_mean(X: sp.csr_matrix) -> sp.csr_matrix:
    """Scale each row to sum to 1 (empty rows stay empty)."""
    rs = np.asarray(X.sum(axis=1)).ravel()
    inv = np.where(rs > 0, 1.0 / np.maximum(rs, 1e-12), 0.0)
    return (sp.diags(inv) @ X).tocsr()


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


# --------------------------------------------------------------------------- #
# Model 1: from-scratch neural network (NumPy)
# --------------------------------------------------------------------------- #
class FastTextNN:
    """
    mean of n-gram embeddings  ->  ReLU hidden layer  ->  sigmoid output

        H = mean_rows(X) @ E          (n_features x emb_dim embedding table)
        R = relu(H @ W1 + b1)         (hidden layer, dropout while training)
        p = sigmoid(R @ w2 + b2)
    """

    def __init__(self, n_features: int, emb_dim: int = 64, hidden: int = 32, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.p = {
            "E": rng.normal(0.0, 0.1, (n_features, emb_dim)),
            "W1": rng.normal(0.0, np.sqrt(2.0 / emb_dim), (emb_dim, hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0.0, np.sqrt(1.0 / hidden), hidden),
            "b2": np.zeros(1),
        }

    # ---- forward / backward ------------------------------------------------
    def _forward(self, Xm, dropout: float = 0.0, rng=None):
        p = self.p
        H = Xm @ p["E"]
        A = H @ p["W1"] + p["b1"]
        R = np.maximum(A, 0.0)
        M = None
        if dropout > 0.0 and rng is not None:
            M = (rng.rand(*R.shape) >= dropout) / (1.0 - dropout)
            R = R * M
        z = R @ p["w2"] + p["b2"][0]
        return z, (H, A, R, M)

    def _loss_and_grads(self, Xm, y, dropout=0.0, rng=None, l2=0.0):
        z, (H, A, R, M) = self._forward(Xm, dropout, rng)
        n = len(y)
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z))  # binary cross-entropy
        d = (_sigmoid(z) - y) / n                              # dLoss/dlogit
        g = {}
        g["w2"] = R.T @ d
        g["b2"] = np.array([d.sum()])
        dR = np.outer(d, self.p["w2"])
        if M is not None:
            dR = dR * M
        dA = dR * (A > 0)
        g["W1"] = H.T @ dA
        g["b1"] = dA.sum(axis=0)
        g["E"] = Xm.T @ (dA @ self.p["W1"].T)
        if l2:
            for k in ("E", "W1", "w2"):
                g[k] = g[k] + l2 * self.p[k]
        return loss, g

    # ---- inference ---------------------------------------------------------
    def decision_function(self, X, batch_size: int = 4096) -> np.ndarray:
        out = []
        for i in range(0, X.shape[0], batch_size):
            z, _ = self._forward(_row_mean(X[i:i + batch_size]))
            out.append(z)
        return np.concatenate(out) if out else np.zeros(0)

    def predict_proba(self, X) -> np.ndarray:
        return _sigmoid(self.decision_function(X))

    # ---- training ----------------------------------------------------------
    def fit(self, Xtr, ytr, Xva, yva, epochs=60, batch_size=128, lr=0.01, dropout=0.3,
            word_dropout=0.1, l2=1e-5, patience=6, seed=0, verbose=True):
        """Mini-batch Adam. Keeps the weights with the lowest validation loss."""
        from sklearn.metrics import roc_auc_score  # training-time only

        rng = np.random.RandomState(seed)
        ytr = np.asarray(ytr, dtype=np.float64)
        yva = np.asarray(yva, dtype=np.float64)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(x) for k, x in self.p.items()}
        beta1, beta2, t = 0.9, 0.999, 0
        best_loss, best_params, best_epoch, bad = np.inf, None, 0, 0
        hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_auc": []}

        for epoch in range(1, epochs + 1):
            order = rng.permutation(Xtr.shape[0])
            running = 0.0
            for s in range(0, len(order), batch_size):
                idx = order[s:s + batch_size]
                Xb = Xtr[idx]  # fancy indexing returns a copy
                if word_dropout > 0:  # randomly hide words -> robustness to unseen words
                    Xb.data = Xb.data * (rng.rand(Xb.data.size) >= word_dropout)
                loss, g = self._loss_and_grads(_row_mean(Xb), ytr[idx], dropout, rng, l2)
                t += 1
                for k in self.p:
                    m[k] = beta1 * m[k] + (1 - beta1) * g[k]
                    v[k] = beta2 * v[k] + (1 - beta2) * g[k] ** 2
                    mhat = m[k] / (1 - beta1 ** t)
                    vhat = v[k] / (1 - beta2 ** t)
                    self.p[k] = self.p[k] - lr * mhat / (np.sqrt(vhat) + 1e-8)
                running += loss * len(idx)

            zva = self.decision_function(Xva)
            va_loss = float(np.mean(np.logaddexp(0.0, zva) - yva * zva))
            try:
                va_auc = float(roc_auc_score(yva, zva))
            except ValueError:
                va_auc = float("nan")
            hist["epoch"].append(epoch)
            hist["train_loss"].append(running / len(order))
            hist["val_loss"].append(va_loss)
            hist["val_auc"].append(va_auc)
            if verbose and (epoch == 1 or epoch % 5 == 0):
                print(f"    epoch {epoch:3d}  train loss {running / len(order):.4f}  "
                      f"val loss {va_loss:.4f}  val AUC {va_auc:.4f}")

            if va_loss < best_loss - 1e-5:
                best_loss, best_epoch, bad = va_loss, epoch, 0
                best_params = {k: x.copy() for k, x in self.p.items()}
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"    early stop at epoch {epoch} (best epoch {best_epoch})")
                    break
        if best_params is not None:
            self.p = best_params
        hist["best_epoch"] = best_epoch
        return hist


class NNScorer:
    """Vocabulary + FastTextNN with a text-in, probability-out interface."""

    kind = "nn"

    def __init__(self, vocab: NGramVocab, net: FastTextNN):
        self.vocab, self.net = vocab, net

    @classmethod
    def train(cls, train_texts, y_train, val_texts, y_val, emb_dim=64, hidden=32,
              min_count=2, max_features=50000, seed=0, **fit_kwargs):
        vocab = NGramVocab(min_count, max_features).fit(train_texts)
        net = FastTextNN(len(vocab.features), emb_dim, hidden, seed)
        hist = net.fit(vocab.transform(train_texts), y_train,
                       vocab.transform(val_texts), y_val, seed=seed, **fit_kwargs)
        return cls(vocab, net), hist

    def predict_proba(self, texts) -> np.ndarray:
        return self.net.predict_proba(self.vocab.transform(list(texts)))

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "weights.npz", **self.net.p)
        (d / "vocab.json").write_text(json.dumps(self.vocab.features), encoding="utf-8")
        (d / "config.json").write_text(json.dumps({
            "kind": "nn", "emb_dim": int(self.net.p["E"].shape[1]),
            "hidden": int(self.net.p["W1"].shape[1]),
            "n_features": len(self.vocab.features),
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        vocab = NGramVocab().set_features(json.loads((d / "vocab.json").read_text(encoding="utf-8")))
        w = np.load(d / "weights.npz")
        net = FastTextNN(1, 1, 1)
        net.p = {k: w[k] for k in ("E", "W1", "b1", "w2", "b2")}
        return cls(vocab, net)


# --------------------------------------------------------------------------- #
# Model 2: TF-IDF + logistic regression (scikit-learn)
# --------------------------------------------------------------------------- #
def build_logreg(C: float = 4.0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    features = FeatureUnion([
        ("word", TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                 ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 sublinear_tf=True)),
    ])
    return Pipeline([("features", features),
                     ("clf", LogisticRegression(C=C, max_iter=3000, solver="liblinear"))])


class SklearnScorer:
    """Wraps a fitted scikit-learn pipeline with a text-in, probability-out interface."""

    kind = "logreg"

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict_proba(self, texts) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))[:, 1]

    def save(self, directory):
        import joblib
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, d / "model.joblib")

    @classmethod
    def load(cls, directory):
        import joblib
        return cls(joblib.load(Path(directory) / "model.joblib"))


SCORERS = {"logreg": SklearnScorer, "nn": NNScorer}


# --------------------------------------------------------------------------- #
# Inference wrapper (what a web portal would use)
# --------------------------------------------------------------------------- #
class RelevanceClassifier:
    """Loads a saved model folder and predicts relevance for headlines."""

    def __init__(self, scorer, threshold: float, name: str, card: dict):
        self.scorer, self.threshold, self.name, self.card = scorer, float(threshold), name, card

    @classmethod
    def load(cls, model_dir, model: str | None = None):
        d = Path(model_dir)
        card = json.loads((d / "model_card.json").read_text(encoding="utf-8"))
        name = model or card["best_model"]
        scorer = SCORERS[card["models"][name]["kind"]].load(d / name)
        return cls(scorer, card["models"][name]["threshold"], name, card)

    def predict_proba(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.scorer.predict_proba([str(t) for t in texts])

    def predict(self, texts, threshold: float | None = None) -> list[dict]:
        if isinstance(texts, str):
            texts = [texts]
        thr = self.threshold if threshold is None else float(threshold)
        scores = self.predict_proba(texts)
        out = []
        for t, s in zip(texts, scores):
            label = int(s >= thr)
            out.append({"title": t, "score": round(float(s), 4), "label": label,
                        "label_name": ID2LABEL[label]})
        return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('usage: python relevance_model.py <model_dir> "headline 1" ["headline 2" ...]')
        sys.exit(1)
    clf = RelevanceClassifier.load(sys.argv[1])
    print(f"model: {clf.name}  threshold: {clf.threshold:.3f}")
    for r in clf.predict(sys.argv[2:]):
        print(f"{r['score']:.3f}  {r['label_name']:<13} {r['title']}")