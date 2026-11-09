# Order Matters, Randomness Too

Code for my semester paper *Order Matters, Randomness Too: Data Sampling Strategies for Finite-Sum Minimization* (Seminar for Statistics, ETH Zurich, 2026), supervised by Prof. Yuansi Chen and Dr. Peter Whalley. The paper is [`order-matters-randomness-too.pdf`](order-matters-randomness-too.pdf). It covers the setup, hyperparameters, and results in detail.

The paper compares four stochastic first-order methods that differ only in how they go through the data: SGD (sampling with replacement), random reshuffling (RR, a new permutation every epoch), shuffle-once (SO, one permutation reused every epoch), and incremental gradient (IG, a fixed order). It reproduces the ℓ²-regularized logistic regression experiments of [Mishchenko et al. (NeurIPS 2020)](https://arxiv.org/abs/2006.05988), and then extends the comparison to small neural networks.

## Repo layout

- `experiments_*.py`: one script per figure in the paper. Each one trains all four methods, caches the results, and saves the plot to `plots/`
  - logistic regression: `realsim`, `w8a`, `rcv1`
  - neural networks: `bikesharing`, `concrete`, `mnist`. The `_linear` variants are the affine (no hidden layer) baselines, and the others use the two-hidden-layer ReLU network
- `optimizers.py`: mini-batch SGD, RR, SO, and IG
- `objectives.py`: the loss functions and models
- `runner.py`: the shared multi-seed training loop, with result caching and the approximation of the optimum
- `plotting.py`: the convergence plots (geometric mean across seeds with a ±1 std band)
- `plots/`: the generated figures
- `further experiments/`: extra datasets that aren't in the paper (synthetic, California housing, Fashion-MNIST, …)

## Running

Run any experiment script from the repo root, e.g.

```bash
python experiments_w8a_LogisticRegression.py
```

Dependencies: `torch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

## Data

`data/` and `results/` are not tracked in this repo (see `.gitignore`). The scripts create them locally and write cached results into `results/`, so a second run loads those results instead of training again.

Most datasets download automatically (RCV1 via `sklearn.datasets.fetch_rcv1`, Bike Sharing, and Concrete via `fetch_openml`, MNIST via `torchvision`). **real-sim and w8a do not.** Get them from the [LIBSVM datasets page](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) or from [Mishchenko et al.'s code release](https://github.com/konstmish/random_reshuffling) and put them in:

```
data/raw/real-sim.bz2
data/raw/w8a.txt
```
