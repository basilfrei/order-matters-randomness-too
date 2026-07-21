# CONVEX BASELINE (linear model, no hidden layers) variant of the Concrete Compressive Strength simple NN experiment

from objectives import *
from optimizers import *
from plotting import *
from runner import run_experiment, approximate_optimum, subsample_metrics, load_final_iterates
from paths import RESULTS_DIR, PLOTS_DIR, DATA_DIR

import os
import math
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

DEVICE = os.environ.get("NN_DEVICE", "cpu")   # "cpu" (default) or "mps" for the M1 GPU

# ============================================================
# Config
# ============================================================

n_seeds    = 5
n_epochs   = 1000
batch_size = 16

# small batch on a subsampled set + many epochs: the RR/SGD separation lives in
# the noise floors near a stationary point, which needs enough gradient noise
# (small bs relative to n) and enough epochs to actually reach the floor.

subsample_n = None             # None = full dataset
hidden_dim  = []              # no hidden layers: d -> 1 (convex baseline)
num_layers  = len(hidden_dim)
reg         = 1e-4
lr          = 1e-3            # constant stepsize; L=4.6, mu=0.06, 1/L_batch~0.11, and the
                              # RR/SGD floor ratio ~ lr*(n/bs)*L/4 = 74*lr needs lr << 1.3e-2

# ============================================================
# Load Concrete Compressive Strength dataset (UCI, via OpenML)
# ============================================================

print("Loading Concrete Compressive Strength (OpenML) ...")
data = fetch_openml(name="Concrete_Data", version=1, as_frame=True, parser="auto",
                    data_home=str(DATA_DIR / "sklearn"))

# OpenML defines no default target for this dataset: the frame holds all 9
# columns, so split off the strength column manually
target_col = "Concrete compressive strength(MPa. megapascals)"
df    = data.frame
y_np  = df[target_col].to_numpy().astype("float64")
X_df  = df.drop(columns=[target_col])

# all 8 features numeric: standardize features and target
A_np = StandardScaler().fit_transform(X_df)
A_np = A_np.clip(-5, 5)
b_np = StandardScaler().fit_transform(y_np.reshape(-1, 1)).ravel()

A = torch.tensor(A_np, dtype=torch.float32)
b = torch.tensor(b_np, dtype=torch.float32)

if subsample_n is not None:
    # fixed-seed subsample, independent of the per-seed init/optimizer RNG
    sub_idx = torch.randperm(A.shape[0], generator=torch.Generator().manual_seed(0))[:subsample_n]
    A = A[sub_idx]
    b = b[sub_idx]

n, d = A.shape
print(f"Dataset: n={n}, d={d}  (all numeric)")

# ============================================================
# Problem and initial point (fresh random init per seed)
# ============================================================

# non-convex objective: a fresh random init per seed gives visible confidence bands
def build_problem(seed):
    torch.manual_seed(seed)
    p = SimpleRegressionNN(A, b, num_hidden_layers=num_layers, hidden_dim=hidden_dim, reg=reg, device=DEVICE)
    return p, p.get_params().clone()

n_params = build_problem(0)[0].get_params().numel()
print(f"Network params: {n_params}  (d={d}, hidden={hidden_dim}, layers={num_layers})")

# ============================================================
# Derived iteration counts
# ============================================================

n_iters_per_epoch = math.ceil(n / batch_size)
num_iterations    = n_epochs * n_iters_per_epoch
store_every       = n_iters_per_epoch // 10      # storing density: stores every store_every-th iterate (>= 10 snapshots/epoch (provided n_iters_per_epoch >= 10))
plot_every        = 30 * store_every             # display density: plots every plot_every-th iterate (use only integer multiples of store_every)

hidden_tag = "x".join(map(str, hidden_dim))
sub_tag    = f"_sub{subsample_n}" if subsample_n is not None else ""
run_tag    = (f"h{hidden_tag}_l{num_layers}{sub_tag}_bs{batch_size}_reg{reg:.0e}"
              f"_lr{lr:.0e}_seeds{n_seeds}_epochs{n_epochs}")
print(f"n_iters_per_epoch={n_iters_per_epoch}, total_iters={num_iterations}")

config = {
    "dataset": "concrete",
    "n": n,
    "d": d,
    "subsample_n": subsample_n,
    "hidden_dim": hidden_dim,
    "num_layers": num_layers,
    "reg": reg,
    "lr": lr,
    "batch_size": batch_size,
    "n_seeds": n_seeds,
    "n_epochs": n_epochs,
    "n_iters_per_epoch": n_iters_per_epoch,
    "num_iterations": num_iterations,
    "store_every": store_every,
    "plot_every": plot_every,
    "grad_normalization": "epoch_uniform",
}

CACHE = RESULTS_DIR / f"concrete_nn_{run_tag}.pt"

# ============================================================
# Run
# ============================================================

results = run_experiment(
    build_optimizers=lambda p, x, seed: {
        "SGD": SGD(objective=p, x0=x, lr=lr, batch_size=batch_size, seed=seed),
        "RR":  RR( objective=p, x0=x, lr=lr, batch_size=batch_size, seed=seed),
        "IG":  IG( objective=p, x0=x, lr=lr, batch_size=batch_size, seed=seed),
        "SO":  SO( objective=p, x0=x, lr=lr, batch_size=batch_size, seed=seed),
    },
    record_metrics=lambda x, p: {
        "loss": p.loss(x).item(),
    },
    build_problem=build_problem,
    cache_path=CACHE,
    config=config,
    n_seeds=n_seeds,
    num_iterations=num_iterations,
    plot_every=plot_every,
    store_every=store_every,
    save_trajectory=False,   # raw iterate snapshots would be huge; final iterates are always cached
)

# ============================================================
# Plot
# ============================================================

# results are stored at store_every density, thin to plot_every
# density for display (pure slicing, no recomputation)
results = subsample_metrics(results, stride=plot_every // store_every)

# plot_every / n_iters_per_epoch converts stored-point index -> epoch units
epoch_scale = plot_every / n_iters_per_epoch

# ============================================================
# Approximate optimum (cached) and gap plot
# ============================================================

# non-convex, so no true f*: polish the best final iterate across all methods/
# seeds with full-batch GD, placing f~* strictly below every curve in the basin
finals = load_final_iterates(CACHE)
p_eval = build_problem(0)[0]   # evaluation copy: same data/objective, init irrelevant

best_x, best_f = None, float("inf")
for name, seeds in finals.items():
    for x_fin in seeds:
        f_val = p_eval.loss(x_fin).item()
        if f_val < best_f:
            best_x, best_f = x_fin, f_val

FSTAR_CACHE = RESULTS_DIR / f"concrete_nn_fstar_{run_tag}.pt"

if FSTAR_CACHE.exists():
    print("Loading cached optimum ...")
    x_star, f_star = torch.load(FSTAR_CACHE)
else:
    # polish with a GD-sized stepsize (~1/L = 0.22): at the minibatch lr the
    # contraction over 2000 steps, exp(-mu*lr*steps) ~ 0.89, would barely move
    print(f"Polishing best final iterate (loss={best_f:.6f}) with GD (2000 steps) ...")
    x_star, f_star = approximate_optimum(p_eval, steps=2000, lr=2e-1, x0=best_x)
    torch.save((x_star, f_star), FSTAR_CACHE)

print(f"Optimal loss: {f_star:.6f}  (best unpolished final: {best_f:.6f})")

print("\nPlotting results ...")
opt_gap = {name: [[v - f_star for v in seed] for seed in seeds]
           for name, seeds in results["loss"].items()}
plot_losses(
    opt_gap,
    ylabel=r"$f(x_k) - \tilde{f}^*$",
    epoch_scale=epoch_scale,
    xlabel="Epoch",
    save_path=str(PLOTS_DIR / f"concrete_nn_linear_gap_{config_to_tag(config)}.pdf"),
)
