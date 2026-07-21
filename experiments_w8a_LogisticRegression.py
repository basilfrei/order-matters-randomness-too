# FINAL VERSION (for paper submission) of the w8a logistic regression experiment

from objectives import *
from optimizers import *
from plotting import *
from runner import run_experiment, approximate_optimum, subsample_metrics
from paths import RESULTS_DIR, PLOTS_DIR, DATA_DIR

import math
import torch
from sklearn.datasets import load_svmlight_file

# ============================================================
# Config
# ============================================================

n_seeds    = 20
n_epochs   = 250
batch_size = 512

# ============================================================
# Load w8a dataset
# ============================================================

DATA_PATH = DATA_DIR / "raw" / "w8a.txt"

print(f"Loading w8a from {DATA_PATH} ...")
A_sp, b_np = load_svmlight_file(str(DATA_PATH))
A = torch.tensor(A_sp.toarray(), dtype=torch.float64)
b = torch.tensor(b_np, dtype=torch.float64)

n, d = A.shape
print(f"Dataset: n={n}, d={d}")

# ============================================================
# Derive hyperparameters from data
# ============================================================

print("Computing spectral smoothness ...")
cov = (A.double().T @ A.double()) / n
L_spectral_no_reg = torch.linalg.eigvalsh(cov).max().item() / 4

row_norms_sq = (A ** 2).sum(dim=1)
L_max_no_reg = row_norms_sq.max().item() / 4

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

problem = LogisticRegression(A, b, reg=reg)
x0      = torch.zeros(d, dtype=torch.float64)

# ============================================================
# Approximate optimum (cached)
# ============================================================

OPT_CACHE = RESULTS_DIR / f"w8a_logreg_optimum_reg{reg:.4e}.pt"

if OPT_CACHE.exists():
    print("Loading cached optimum ...")
    x_opt, f_opt = torch.load(OPT_CACHE)
else:
    print("Approximating optimal value with GD (5000 steps) ...")
    OPT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    x_opt, f_opt = approximate_optimum(problem, steps=5000, lr=lr_gd, x0=x0)
    torch.save((x_opt, f_opt), OPT_CACHE)

print(f"Optimal loss: {f_opt:.6f}")

# ============================================================
# Derived iteration counts
# ============================================================

n_iters_per_epoch = math.ceil(n / batch_size)
num_iterations    = n_epochs * n_iters_per_epoch
store_every       = n_iters_per_epoch // 20
plot_every        = 19 * store_every
it_start_decay    = num_iterations // 40

run_tag = f"bs{batch_size}_seeds{n_seeds}_epochs{n_epochs}_lrmax{lr_max:.2e}"
print(f"n_iters_per_epoch={n_iters_per_epoch}, total_iters={num_iterations}")
print(f"it_start_decay={it_start_decay}  (~{it_start_decay // n_iters_per_epoch} epochs warm-up)")

config = {
    "dataset": "w8a",
    "n": n,
    "d": d,
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

CACHE = RESULTS_DIR / f"w8a_logreg_{run_tag}.pt"

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
    save_path=str(PLOTS_DIR / f"w8a_logreg_optgap_{config_to_tag(config)}.pdf"),
)
plot_losses(
    iter_gap_sq,
    ylabel=r"$\|x_k - x^*\|_2^2$",
    epoch_scale=epoch_scale, xlabel="Epoch",
    save_path=str(PLOTS_DIR / f"w8a_logreg_itergap_{config_to_tag(config)}.pdf"),
)
