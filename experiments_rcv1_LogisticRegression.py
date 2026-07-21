# FINAL VERSION (for paper submission) of the RCV1 logistic regression experiment

from objectives import *
from optimizers import *
from plotting import *
from runner import run_experiment, approximate_optimum, subsample_metrics
from paths import RESULTS_DIR, PLOTS_DIR, DATA_DIR

import math
import numpy as np
import torch
from scipy.sparse.linalg import svds
import sklearn.datasets

# ============================================================
# Config
# ============================================================

n_seeds    = 20
n_epochs   = 30
batch_size = 512

# ============================================================
# Load RCV1 dataset
# ============================================================

print("Loading RCV1 from sklearn ...")
A_sp, b_multilabel = sklearn.datasets.fetch_rcv1(subset="all", return_X_y=True,
                                                   data_home=str(DATA_DIR / "sklearn"))
A_sp = A_sp.astype(np.float32)

freq       = np.asarray(b_multilabel.sum(axis=0)).squeeze()
main_class = int(np.argmax(freq))
b_binary   = (np.asarray(b_multilabel[:, main_class].toarray()).ravel() == 1).astype(np.float32)
b_np       = 2.0 * b_binary - 1.0   # transform {0,1} to {-1,+1}
b          = torch.tensor(b_np)

n, d = A_sp.shape
print(f"Dataset: n={n}, d={d}, nnz={A_sp.nnz}, density={A_sp.nnz/(n*d):.4%}")
print(f"Binary labels from class {main_class}  "
      f"(pos={int(b_binary.sum())}, neg={n - int(b_binary.sum())})")

# ============================================================
# Derive hyperparameters from data
# ============================================================

print("Computing spectral smoothness via scipy.sparse.linalg.svds(A, k=1) ...")
sigma_max   = svds(A_sp, k=1, return_singular_vectors=False, random_state=0)[0]
L_spectral_no_reg  = float(sigma_max) ** 2 / (4 * n)

row_norms_sq = np.asarray(A_sp.power(2).sum(axis=1)).ravel()
L_max_no_reg = float(row_norms_sq.max()) / 4

reg = L_spectral_no_reg / (n ** 0.5)

L_spectral_with_reg = L_spectral_no_reg + reg
L_max_with_reg = L_max_no_reg + reg

bs      = batch_size
L_batch = n / (n - 1) * (1 - 1/bs) * L_spectral_with_reg + (n/bs - 1) / (n - 1) * L_max_with_reg

lr0    = 1.0 / reg
lr_max = 1.0 / L_batch
lr_gd  = 1.0 / L_spectral_with_reg

decay_rr_ig_so = reg / 3
decay_sgd      = reg / 2

print(f"L_spectral_no_reg={L_spectral_no_reg:.6f}, L_max_no_reg={L_max_no_reg:.4f}, L_batch={L_batch:.6f}")
print(f"reg={reg:.2e}, lr0={lr0:.2e}, lr_max={lr_max:.4f}, lr_gd={lr_gd:.6f}")

# ============================================================
# Problem and initial point
# ============================================================

problem = LogisticRegression(A_sp, b, reg=reg)
x0      = torch.zeros(d)

# ============================================================
# Approximate optimum (cached)
# ============================================================

OPT_CACHE = RESULTS_DIR / f"rcv1_logreg_optimum_reg{reg:.4e}.pt"

if OPT_CACHE.exists():
    print("Loading cached optimum ...")
    x_opt, f_opt = torch.load(OPT_CACHE)
else:
    print(f"Approximating optimum with GD (5000 steps) ...")
    OPT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    x_opt, f_opt = approximate_optimum(problem, steps=5000, lr=lr_gd, x0=x0)
    torch.save((x_opt, f_opt), OPT_CACHE)

print(f"Optimal loss: {f_opt:.6f}")

# ============================================================
# Derived iteration counts
# ============================================================

n_iters_per_epoch = math.ceil(n / batch_size)
num_iterations    = n_epochs * n_iters_per_epoch
store_every       = n_iters_per_epoch // 20    # storing density: stores every store_every-th iterate (>= 4 snapshots/epoch (provided n_iters_per_epoch >= 4))
plot_every        = 3 * store_every                # display density: plots every plot_every-th iterate (use only integer multiples of store_every)
it_start_decay    = num_iterations // 40

run_tag = f"bs{batch_size}_seeds{n_seeds}_epochs{n_epochs}_lrmax{lr_max:.2e}"
print(f"n_iters_per_epoch={n_iters_per_epoch}, total_iters={num_iterations}")
print(f"it_start_decay={it_start_decay}  (~{it_start_decay // n_iters_per_epoch} epochs warm-up)")

config = {
    "dataset": "rcv1_binary",
    "n": n,
    "d": d,
    "main_class": main_class,
    "L_spectral_no_reg": L_spectral_no_reg,
    "L_max_no_reg": L_max_no_reg,
    "L_batch": L_batch,
    "reg": reg,
    "f_opt": f_opt,
    "opt_gd_steps": 5000,
    "lr0": lr0,
    "lr_max": lr_max,
    "lr_gd": lr_gd,
    "decay_rr_ig_so": decay_rr_ig_so,
    "decay_sgd": decay_sgd,
    "it_start_decay": it_start_decay,
    "batch_size": batch_size,
    "n_seeds": n_seeds,
    "n_epochs": n_epochs,
    "n_iters_per_epoch": n_iters_per_epoch,
    "num_iterations": num_iterations,
    "store_every": store_every,
    "plot_every": plot_every,
    "grad_normalization": "epoch_uniform",
}

CACHE = RESULTS_DIR / f"rcv1_logreg_{run_tag}.pt"

# ============================================================
# Run
# ============================================================

results = run_experiment(
    build_optimizers=lambda p, x, seed: {
        "SGD": SGD(objective=p, x0=x, lr=lr0, lr_max=lr_max, batch_size=batch_size, seed=seed,
                   lr_decay_coef=decay_sgd,      it_start_decay=it_start_decay),
        "RR":  RR( objective=p, x0=x, lr=lr0, lr_max=lr_max, batch_size=batch_size, seed=seed,
                   lr_decay_coef=decay_rr_ig_so, it_start_decay=it_start_decay),
        "IG":  IG( objective=p, x0=x, lr=lr0, lr_max=lr_max, batch_size=batch_size, seed=seed,
                   lr_decay_coef=decay_rr_ig_so, it_start_decay=it_start_decay),
        "SO":  SO( objective=p, x0=x, lr=lr0, lr_max=lr_max, batch_size=batch_size, seed=seed,
                   lr_decay_coef=decay_rr_ig_so, it_start_decay=it_start_decay),
    },
    record_metrics=lambda x, p: {
        "opt_gap":  p.loss(x).item() - f_opt,
        "iter_gap": torch.norm(x - x_opt).item(),
    },
    problem=problem,
    x0=x0,
    cache_path=CACHE,
    config=config,
    n_seeds=n_seeds,
    num_iterations=num_iterations,
    plot_every=plot_every,
    store_every=store_every,
)

# ============================================================
# Plot
# ============================================================

# results are stored at store_every density, thin to plot_every
# density for display (pure slicing, no recomputation)
results = subsample_metrics(results, stride=plot_every // store_every)

# plot_every / n_iters_per_epoch converts stored-point index -> epoch units
epoch_scale = plot_every / n_iters_per_epoch

print("\nPlotting results ...")
iter_gap_sq = {name: [[v ** 2 for v in seed] for seed in seeds]
               for name, seeds in results["iter_gap"].items()}
plot_losses(
    results["opt_gap"],
    ylabel=r"$f(x_k) - f^*$",
    epoch_scale=epoch_scale, xlabel="Epoch",
    save_path=str(PLOTS_DIR / f"rcv1_logreg_optgap_{config_to_tag(config)}.pdf"),
)
plot_losses(
    iter_gap_sq,
    ylabel=r"$\|x_k - x^*\|_2^2$",
    epoch_scale=epoch_scale, xlabel="Epoch",
    save_path=str(PLOTS_DIR / f"rcv1_logreg_itergap_{config_to_tag(config)}.pdf"),
)
