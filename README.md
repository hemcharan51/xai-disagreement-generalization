# The Disagreement Problem in Explainable ML — code

Recreated research code for the project **"The Disagreement Problem: How Model
Generalization Impacts XAI Consensus"**
(Kowel P Laloo · Chitiveli Hemcharan Varma · advisor Prof. Manisha Padala,
Dept. of CSE, IIT Gandhinagar).

The project builds on, and is layered *on top of*, the paper it reproduces:

> Krishna, Han, Gu, Wu, Jabbari, Lakkaraju (2024).
> *The Disagreement Problem in Explainable Machine Learning: A Practitioner's
> Perspective.* Transactions on Machine Learning Research.

Two artifacts seed this repo and live alongside it:
[`2202.01602v6-4.pdf`](./) (the paper) and
[`xai_disagreement_poster (1).pdf`](./xai_disagreement_poster%20(1).pdf) (our poster).

---

## What's here

```
xai_disagreement/        Shared core library (the paper's framework)
  metrics.py             6 paper metrics + weighted rank agr. + our RA/SA/SRA + entropy
  explainers.py          LIME, KernelSHAP, VanillaGrad, Grad*Input, IG, SmoothGrad
  alignment.py           project every attribution onto one shared index space
  aggregate.py           average metrics over method pairs and instances
  findings.py            the 3 poster findings as reusable analysis + plots

paper_reproduction/      Krishna et al. (2024) reproduction on COMPAS   [build FIRST]
adult_income_study/      OUR study, original: UCI Adult Income + 4 MLPs  (the poster)
mnist_cnn_study/         OUR study, extension: MNIST + a basic CNN

docs/method_notes.md     Formal definitions of every metric
```

The order mirrors the request: the **paper implementation comes first**
(`xai_disagreement/` + `paper_reproduction/`), then **our main research**
(`adult_income_study/` reproduces the poster, `mnist_cnn_study/` extends it to
images).

---

## Setup

The repo is self-contained — LIME, KernelSHAP and the gradient methods are
implemented from scratch, so no `lime`/`shap`/`captum` is required.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> A `.venv/` built with `numpy<2` already exists in this folder; the global
> Anaconda environment on this machine has a broken numpy 1.x/2.x ABI mix and is
> **not** used.

---

## Run

```bash
# 1. Paper reproduction (COMPAS, 6 methods, 6 metrics -> Fig-1 heatmaps)
python -m paper_reproduction.run_disagreement

# 2. Our poster study (Adult Income, 4 MLP variants, 3 findings)
python -m adult_income_study.run_all              # add --quick for a fast smoke run

# 3. Our extension (MNIST + CNN, 3 findings)
python -m mnist_cnn_study.run_all                 # add --quick for a fast smoke run
```

Each writes PNG figures + a `results.json` to its own `figures/` folder.

---

## The study in one paragraph

Prior work treats XAI disagreement as a mathematical artifact of the *methods*.
Our hypothesis: it is also a symptom of **model quality** — poorly generalizing
models produce unreliable gradient landscapes that amplify or mask inter-method
divergence. We train one architecture four ways across a generalization
spectrum — **A** Control · **C** Feature-Noise · **D** Label-Poison · **B**
Overfit — apply four XAI methods (LIME, KernelSHAP, IG, SmoothGrad) all
projected onto the same feature indices, and measure pairwise Rank Agreement
(RA), Sign Agreement (SA) and Signed Rank Agreement (SRA, primary).

Three findings the code reproduces:

1. **Overfitting destroys consensus** — mean pairwise SRA declines as
   generalization degrades (A → C → D → B).
2. **The Uncertainty Paradox** — predictive entropy *positively* correlates with
   SRA: degraded, uncertain models give flat, near-zero attributions, so methods
   trivially "agree on nothing." High agreement ≠ informative.
3. **Family-structured disagreement** — gradient methods (IG, SmoothGrad)
   cluster; perturbation LIME stays isolated; KernelSHAP shifts allegiance under
   degradation. Intra-family consensus is evidence of *shared bias*, not
   correctness.

See [`docs/method_notes.md`](docs/method_notes.md) for exact metric definitions
and the one adaptation we make to the SRA formula.

---

## Reproducibility caveat

This is a faithful re-implementation of the *framework and methodology*, not a
bit-for-bit replay of the original lost code. Models, seeds and the from-scratch
LIME/KernelSHAP samplers differ from the originals, so absolute metric values
will not match the poster digit-for-digit (e.g. our SRA sits lower than the
poster's ≈ 0.73). The machinery, figures and qualitative findings are the
deliverable; everything is computed honestly from freshly trained models.
