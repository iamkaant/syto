"""A small MLP committee, in numpy.

A faithful port of the AutoParty model (Keiser lab; the user's fork lives at
../AutoParty), with PyTorch removed. The original is

    Linear(in, n) -> ReLU -> Dropout   x hidden_layers
    Linear(n, n_options) -> Sigmoid (ordinal) or Softmax (classes)
    BCE loss, Adam(lr, weight_decay), a committee of `committee_size`
    trained on bootstrap or disjoint splits, predict() returning
    (mean, variance across the committee)

Why numpy rather than torch
---------------------------
At the scale of a hit-picking party - hundreds to a few thousand poses, a
couple of thousand features - a hand-written forward and backward pass
trains a committee in seconds. Dropping torch is what lets this run inside
ChimeraX's Python and, later, inside Pyodide in a browser, which is the
whole premise of the shared core.

Bugs in the original that are not reproduced here
-------------------------------------------------
* `np.mean(epoch_losses, val_losses)` in the validation branch, where the
  second argument is silently taken as `axis`;
* `uncertainty_method` offering 'dropout' and 'distance', neither of which
  is implemented - only 'ensemble' is, so only that is offered here.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MLP", "Committee", "ordinal_targets", "decode_ordinal"]


# --------------------------------------------------------------- labels

def ordinal_targets(ranks, n_options):
    """Thermometer encoding: rank k becomes k+1 ones then zeros.

    Grades are ordered, not categorical - "maybe" sits between "bad" and
    "good" - and a thermometer target is what lets the model learn that,
    rather than treating the three as unrelated classes.
    """
    ranks = np.asarray(ranks, dtype=int)
    out = np.zeros((len(ranks), n_options), dtype=np.float64)
    for i, r in enumerate(ranks):
        out[i, :max(0, min(int(r) + 1, n_options))] = 1.0
    return out


def decode_ordinal(probs, threshold=0.5):
    """Thermometer outputs back to a rank.

    Counts units above the threshold. A non-monotone row (0.9, 0.3, 0.8) is
    resolved at the FIRST crossing rather than by counting, because counting
    would quietly reward an inconsistent prediction.
    """
    probs = np.asarray(probs, dtype=np.float64)
    out = np.zeros(len(probs), dtype=int)
    for i, row in enumerate(probs):
        rank = 0
        for v in row:
            if v >= threshold:
                rank += 1
            else:
                break
        out[i] = max(0, rank - 1)
    return out


# ----------------------------------------------------------------- MLP

class MLP:
    """Dense network with ReLU hidden layers and a sigmoid output."""

    def __init__(self, n_in, n_out, n_neurons=256, hidden_layers=2,
                 dropout=0.2, lr=1e-3, weight_decay=1e-2, seed=0):
        self.rng = np.random.default_rng(seed)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        sizes = [n_in] + [n_neurons] * hidden_layers + [n_out]
        self.W, self.b = [], []
        for a, c in zip(sizes[:-1], sizes[1:]):
            # He initialisation, appropriate for ReLU
            self.W.append(self.rng.normal(0.0, np.sqrt(2.0 / a), size=(a, c)))
            self.b.append(np.zeros(c))
        self._reset_adam()
        self.n_in, self.n_out = n_in, n_out

    def _reset_adam(self):
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    # ---------------------------------------------------------- forward

    def forward(self, X, train=False):
        """Returns (output, cache) where cache is needed by backward()."""
        a = np.asarray(X, dtype=np.float64)
        acts, masks, pre = [a], [], []
        n_layers = len(self.W)
        for k in range(n_layers):
            z = a @ self.W[k] + self.b[k]
            pre.append(z)
            if k < n_layers - 1:
                a = np.maximum(z, 0.0)
                if train and self.dropout > 0:
                    # inverted dropout: scale at train time so inference is
                    # a plain forward pass
                    keep = 1.0 - self.dropout
                    m = (self.rng.random(a.shape) < keep) / keep
                    a = a * m
                    masks.append(m)
                else:
                    masks.append(None)
            else:
                a = _sigmoid(z)
            acts.append(a)
        return a, {"acts": acts, "masks": masks, "pre": pre}

    def predict(self, X):
        return self.forward(X, train=False)[0]

    # --------------------------------------------------------- backward

    def backward(self, cache, Y):
        """Gradients of the mean BCE loss with respect to W and b."""
        acts, masks = cache["acts"], cache["masks"]
        Y = np.asarray(Y, dtype=np.float64)
        n = len(Y)
        n_layers = len(self.W)

        # d(mean BCE)/d(pre-activation) for a sigmoid output is (p - y),
        # divided by the number of ELEMENTS, because loss() averages over
        # every output unit as PyTorch's BCELoss does by default. Dividing
        # by the batch size alone makes the gradient n_out times too large,
        # which a numerical check catches immediately and training does not.
        delta = (acts[-1] - Y) / (n * self.n_out)
        gW = [None] * n_layers
        gb = [None] * n_layers
        for k in range(n_layers - 1, -1, -1):
            gW[k] = acts[k].T @ delta
            gb[k] = delta.sum(axis=0)
            if k > 0:
                delta = delta @ self.W[k].T
                if masks[k - 1] is not None:
                    delta = delta * masks[k - 1]
                delta = delta * (cache["pre"][k - 1] > 0)
        return gW, gb

    def step(self, gW, gb, beta1=0.9, beta2=0.999, eps=1e-8):
        """One Adam update, with decoupled weight decay (AdamW)."""
        self.t += 1
        for k in range(len(self.W)):
            self.mW[k] = beta1 * self.mW[k] + (1 - beta1) * gW[k]
            self.vW[k] = beta2 * self.vW[k] + (1 - beta2) * (gW[k] ** 2)
            mhat = self.mW[k] / (1 - beta1 ** self.t)
            vhat = self.vW[k] / (1 - beta2 ** self.t)
            self.W[k] -= self.lr * (mhat / (np.sqrt(vhat) + eps)
                                    + self.weight_decay * self.W[k])

            self.mb[k] = beta1 * self.mb[k] + (1 - beta1) * gb[k]
            self.vb[k] = beta2 * self.vb[k] + (1 - beta2) * (gb[k] ** 2)
            mhat = self.mb[k] / (1 - beta1 ** self.t)
            vhat = self.vb[k] / (1 - beta2 ** self.t)
            self.b[k] -= self.lr * mhat / (np.sqrt(vhat) + eps)

    # ---------------------------------------------------------- fitting

    def loss(self, P, Y):
        P = np.clip(np.asarray(P, dtype=np.float64), 1e-9, 1 - 1e-9)
        Y = np.asarray(Y, dtype=np.float64)
        return float(-np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P)))

    def train_epoch(self, X, Y, batch_size=32):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        order = self.rng.permutation(len(X))
        total, seen = 0.0, 0
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            if len(idx) == 0:
                continue
            P, cache = self.forward(X[idx], train=True)
            gW, gb = self.backward(cache, Y[idx])
            self.step(gW, gb)
            total += self.loss(P, Y[idx]) * len(idx)
            seen += len(idx)
        return total / max(seen, 1)

    # --------------------------------------------------------- persistence

    def state(self):
        return {"W": [w.tolist() for w in self.W],
                "b": [b.tolist() for b in self.b],
                "n_in": self.n_in, "n_out": self.n_out,
                "dropout": self.dropout, "lr": self.lr,
                "weight_decay": self.weight_decay}

    @classmethod
    def from_state(cls, st):
        m = cls(st["n_in"], st["n_out"], n_neurons=1, hidden_layers=0,
                dropout=st.get("dropout", 0.0), lr=st.get("lr", 1e-3),
                weight_decay=st.get("weight_decay", 0.0))
        m.W = [np.asarray(w, dtype=np.float64) for w in st["W"]]
        m.b = [np.asarray(b, dtype=np.float64) for b in st["b"]]
        m._reset_adam()
        return m


def _sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


# ------------------------------------------------------------ committee

class Committee:
    """Several MLPs on different samples; their spread is the uncertainty."""

    def __init__(self, n_in, n_out, size=3, data_split="bootstrap", seed=0,
                 **mlp_kw):
        self.size = int(size)
        self.data_split = data_split
        self.rng = np.random.default_rng(seed)
        self.members = [MLP(n_in, n_out, seed=seed + 1000 * k, **mlp_kw)
                        for k in range(self.size)]
        self.history = []

    def _splits(self, n):
        if self.data_split == "disjoint":
            order = self.rng.permutation(n)
            return [order[k::self.size] for k in range(self.size)]
        return [self.rng.integers(0, n, size=n) for _ in range(self.size)]

    def fit(self, X, Y, epochs=100, batch_size=32, val_fraction=0.1,
            patience=20, verbose=False):
        """Train every member, keeping each one's best validation weights."""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        n = len(X)
        if n < 4:
            raise ValueError("need at least 4 graded poses to train")

        use_val = n >= 20 and val_fraction > 0
        self.history = []
        for m, idx in zip(self.members, self._splits(n)):
            xi, yi = X[idx], Y[idx]
            if use_val:
                cut = max(1, int(len(xi) * val_fraction))
                perm = self.rng.permutation(len(xi))
                vi, ti = perm[:cut], perm[cut:]
                xv, yv, xt, yt = xi[vi], yi[vi], xi[ti], yi[ti]
            else:
                xt, yt, xv, yv = xi, yi, None, None

            best, best_state, waited = np.inf, None, 0
            curve = []
            for ep in range(epochs):
                tr = m.train_epoch(xt, yt, batch_size)
                if xv is None:
                    curve.append((tr, None))
                    continue
                vl = m.loss(m.predict(xv), yv)
                curve.append((tr, vl))
                if vl < best - 1e-6:
                    best, best_state, waited = vl, m.state(), 0
                else:
                    waited += 1
                    if waited >= patience:
                        break
            if best_state is not None:
                restored = MLP.from_state(best_state)
                m.W, m.b = restored.W, restored.b
            self.history.append(curve)
        return self

    def predict(self, X):
        """(mean probabilities, variance across the committee)."""
        preds = np.stack([m.predict(X) for m in self.members])
        return preds.mean(axis=0), preds.var(axis=0)

    def state(self):
        return {"size": self.size, "data_split": self.data_split,
                "members": [m.state() for m in self.members]}

    @classmethod
    def from_state(cls, st):
        first = st["members"][0]
        c = cls(first["n_in"], first["n_out"], size=len(st["members"]),
                data_split=st.get("data_split", "bootstrap"))
        c.members = [MLP.from_state(s) for s in st["members"]]
        return c
