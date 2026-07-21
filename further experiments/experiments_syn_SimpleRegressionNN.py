from objectives import *
from optimizers import *
from plotting import *
from runner import run_experiment, subsample_metrics
from paths import RESULTS_DIR, PLOTS_DIR

import math
import torch

# ============================================================
# Config
# ============================================================

n_seeds        = 10
num_iterations = 50000
store_every    = 1                               # small network: cheap to store every iterate
plot_every     = max(1, num_iterations // 200)   # display density; free to change later, no rerun needed
lr             = 1e-2
batch_size     = 5
reg            = 1e-3
num_layers     = 2
hidden_dim     = 20

run_tag = f"l{num_layers}_reg{reg:.0e}_lr{lr:.0e}_bs{batch_size}_seeds{n_seeds}_iters{num_iterations}"

config = {
    "dataset": "synthetic_simple_NN",
    "n": 1000, "d": 20, "d_out": 1, "num_layers": num_layers, "hidden_dim": hidden_dim,
    "data_seed": 42, "reg": reg, "lr": lr, "batch_size": batch_size,
    "n_seeds": n_seeds, "num_iterations": num_iterations,
    "store_every": store_every, "plot_every": plot_every,
    "grad_normalization": "epoch_uniform",
}

CACHE = RESULTS_DIR / f"simpleNN_{run_tag}.pt"

# ============================================================
# Setup
# ============================================================

print("Setting up synthetic simple NN problem...")
A_syn, b_syn = create_synthetic_simple_NN(n=1000, d=20, d_out=1, num_layers=num_layers, hidden_dim=hidden_dim, seed=42)
problem      = SimpleRegressionNN(A_syn, b_syn, num_hidden_layers=num_layers, hidden_dim=hidden_dim, reg=reg)
x0           = problem.get_params().clone()

# NN is non-convex — no global optimum to compare against, so we plot the
# raw training loss f(x_k) (not an optimality or iterate gap).
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
    record_metrics=lambda x, p: {"loss": p.loss(x).item()},
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
plot_losses(
    results["loss"],
    ylabel=r"$f(x_k)$",
    epoch_scale=epoch_scale, xlabel="Epoch",
    save_path=str(PLOTS_DIR / f"simpleNN_loss_{config_to_tag(config)}.pdf"),
)
