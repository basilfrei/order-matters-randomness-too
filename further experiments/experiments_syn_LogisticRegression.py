from objectives import *
from optimizers import *
from plotting import *
from runner import run_experiment, approximate_optimum, subsample_metrics
from paths import RESULTS_DIR, PLOTS_DIR

import math
import torch

# ============================================================
# Config
# ============================================================

n_seeds        = 10
num_iterations = 20000
store_every    = 2                               # d=20: cheap to store every iterate
plot_every     = max(1, num_iterations // 100)   # display density; free to change later, no rerun needed
lr             = 1e-2
batch_size     = 10
reg            = 1e-3

run_tag = f"reg{reg:.0e}_lr{lr:.0e}_bs{batch_size}_seeds{n_seeds}_iters{num_iterations}"

config = {
    "dataset": "synthetic_logistic_regression",
    "n": 1000, "d": 20, "data_seed": 42,
    "reg": reg, "lr": lr, "batch_size": batch_size,
    "n_seeds": n_seeds, "num_iterations": num_iterations,
    "store_every": store_every, "plot_every": plot_every,
    "grad_normalization": "epoch_uniform",
}

CACHE = RESULTS_DIR / f"logreg_{run_tag}.pt"

# ============================================================
# Setup
# ============================================================

print("Setting up synthetic logistic regression problem...")
A_syn, b_syn, _ = create_synthetic_logistic_regression(n=1000, d=20, seed=42)
problem = LogisticRegression(A_syn, b_syn, reg=reg)
x0      = torch.zeros(problem.d)

print("Approximating optimal value...")
x_opt, f_opt = approximate_optimum(problem, steps=50000, lr=lr, x0=x0)
print(f"Optimal loss: {f_opt:.6f}")

n_iters_per_epoch = math.ceil(problem.n / batch_size)
epoch_scale       = plot_every / n_iters_per_epoch

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
        "opt_gap":  p.loss(x).item() - f_opt,
        "iter_gap": torch.norm(x - x_opt).item(),
    },
    problem=problem, x0=x0,
    cache_path=CACHE, config=config,
    n_seeds=n_seeds, num_iterations=num_iterations, plot_every=plot_every,
    store_every=store_every,
)

# ============================================================
# Plot
# ============================================================

# results are computed at store_every density; thin to plot_every for
# display — pure slicing, no recomputation.
results = subsample_metrics(results, stride=plot_every // store_every)

print("\nPlotting results...")
iter_gap_sq = {name: [[v ** 2 for v in seed] for seed in seeds]
               for name, seeds in results["iter_gap"].items()}
plot_losses(
    results["opt_gap"],
    ylabel=r"$f(x_k) - f^*$",
    epoch_scale=epoch_scale, xlabel="Epoch",
    save_path=str(PLOTS_DIR / f"logreg_optgap_{config_to_tag(config)}.pdf"),
)
plot_losses(
    iter_gap_sq,
    ylabel=r"$\|x_k - x^*\|_2^2$",
    epoch_scale=epoch_scale, xlabel="Epoch",
    save_path=str(PLOTS_DIR / f"logreg_itergap_{config_to_tag(config)}.pdf"),
)
