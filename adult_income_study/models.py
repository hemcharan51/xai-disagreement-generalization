"""
The four-model degradation spectrum from the poster.

A single MLP architecture is trained four ways to sweep generalization quality
from best (A) to worst (B):

    A  Control       clean data, weight decay + early stopping   (High Gen.)
    C  Feature Noise Gaussian noise added to inputs in training  (Low Gen.)
    D  Label Poison  20% of training labels flipped              (Low Gen.)
    B  Overfit       high-capacity net memorizes a small subset  (Worst Gen.)

``train_variant`` returns the trained model plus a record of train/test accuracy
and the generalization gap, so the spectrum can be ordered by measured quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


# Poster ordering: best -> worst generalization
SPECTRUM = ["A", "C", "D", "B"]
VARIANT_LABEL = {
    "A": "A - Control",
    "C": "C - Feat. Noise",
    "D": "D - Label Poison",
    "B": "B - Overfit",
}


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(128, 64), n_classes: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, X) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            xt = torch.tensor(np.asarray(X, dtype=np.float32))
            return torch.softmax(self(xt), dim=1).numpy()


@dataclass
class TrainedModel:
    variant: str
    model: MLP
    train_acc: float
    test_acc: float
    gen_gap: float = field(init=False)

    def __post_init__(self):
        self.gen_gap = self.train_acc - self.test_acc


def _accuracy(model: MLP, X, y) -> float:
    return float((model.predict_proba(X).argmax(1) == y).mean())


def train_variant(variant: str, ds, seed: int = 0, device: str = "cpu",
                  verbose: bool = False) -> TrainedModel:
    """Train one point of the degradation spectrum."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xtr, ytr = ds.X_train.copy(), ds.y_train.copy()

    # --- recipe per variant -------------------------------------------------
    if variant == "A":            # control: full clean data, regularized
        hidden, wd, epochs, sub = (128, 64), 1e-3, 40, None
        noise = 0.0
    elif variant == "C":          # feature noise: Gaussian input corruption
        hidden, wd, epochs, sub = (128, 64), 1e-3, 40, None
        noise = 0.5
    elif variant == "D":          # label poison: flip 20% of labels
        hidden, wd, epochs, sub = (128, 64), 1e-3, 40, None
        noise = 0.0
        flip = rng.random(len(ytr)) < 0.20
        ytr[flip] = 1 - ytr[flip]
    elif variant == "B":          # overfit: big net, no reg, memorize subset
        hidden, wd, epochs, sub = (512, 512, 256), 0.0, 300, 2000
        noise = 0.0
    else:
        raise ValueError(variant)

    if sub is not None:
        idx = rng.choice(len(Xtr), size=min(sub, len(Xtr)), replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]

    model = MLP(ds.n_features, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()

    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    bs = 256
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            xb = Xt[b]
            if noise > 0:
                xb = xb + noise * torch.randn_like(xb)
            opt.zero_grad()
            loss = lossf(model(xb), yt[b])
            loss.backward()
            opt.step()
        if verbose and ep % 10 == 0:
            print(f"  [{variant}] epoch {ep:3d} loss {loss.item():.3f}")

    tr_acc = _accuracy(model, ds.X_train, ds.y_train)
    te_acc = _accuracy(model, ds.X_test, ds.y_test)
    return TrainedModel(variant, model, tr_acc, te_acc)
