import torch
from pathlib import Path

CACHE_STRICT_KEYS = [
    "n_seeds",
    "num_iterations",
    "store_every",
    "batch_size",
    "n_epochs",
    "n_iters_per_epoch",
]

# Every experiment file builds exactly these four optimizers; used only to
# print a size estimate before running, not to affect the actual computation.
_N_OPTIMIZERS_GUESS = 4


def run_experiment(
    build_optimizers,
    record_metrics,
    cache_path,
    config,
    n_seeds,
    num_iterations,
    plot_every,
    store_every=None,
    problem=None,
    x0=None,
    build_problem=None,
    print_every=None,
    save_trajectory=True,
):
    """
    Shared multi-seed training loop with smart cache validation.

    The optimizer loop only stores cheap iterate snapshots (``store_every``
    apart) — no metric evaluation happens inside the loop. The expensive
    ``record_metrics`` pass (a full-dataset loss / iterate-gap evaluation)
    runs once per seed *after* the loop, over the stored snapshots. This
    mirrors random_reshuffling-master's Trace design (store x cheaply during
    the run, compute losses afterward): picking a different display density
    later never requires rerunning the optimizer — see
    ``metrics_from_trajectory`` / ``load_trajectory``.

    build_optimizers : (problem, x0, seed) -> dict[name, Optimizer]
    record_metrics   : (x, problem) -> dict[metric_name, float]
    build_problem    : (seed) -> (problem, x0)  — used instead of problem/x0
                       when each seed needs its own init (e.g. California NN)
    store_every      : iteration gap between stored iterate snapshots.
                       Defaults to plot_every (today's density) if omitted.
    save_trajectory  : persist the raw snapshots in the cache so a later run
                       can recompute metrics (e.g. a new metric, or a coarser
                       stride) without re-optimizing. Set False for problems
                       where even the snapshot cache would be too large.

    Returns dict: {metric_name: {opt_name: [[seed0_vals], [seed1_vals], ...]}}
    """
    cache_path = Path(cache_path)
    if store_every is None:
        store_every = plot_every
    if print_every is None:
        print_every = max(1, num_iterations // 10)

    # --- cache check ---
    if cache_path.exists():
        saved   = torch.load(cache_path)
        stored  = saved.get("config", {})
        mismatches = {
            k: (stored.get(k), config.get(k))
            for k in CACHE_STRICT_KEYS
            if k in config and stored.get(k) != config.get(k)
        }
        if not mismatches:
            print(f"Loading cached results from {cache_path} ...")
            for k, v in stored.items():
                print(f"  {k} = {v}")
            return {k: v for k, v in saved.items() if k not in ("config", "trajectories", "final_iterates")}
        print(f"Config changed, recomputing (mismatches: {mismatches})")

    # --- estimate trajectory footprint before paying for it ---
    n_snapshots_per_seed = num_iterations // store_every + 1
    probe_x = x0 if x0 is not None else build_problem(0)[1]
    dim = probe_x.numel()
    est_mb = n_snapshots_per_seed * dim * 4 * _N_OPTIMIZERS_GUESS * n_seeds / 1e6
    print(f"Snapshot plan: {n_snapshots_per_seed} points/seed x d={dim} x "
          f"{_N_OPTIMIZERS_GUESS} optimizers x {n_seeds} seeds ~= {est_mb:.1f} MB "
          f"({'saved to cache' if save_trajectory else 'kept in memory only, not cached'})")

    # --- compute ---
    results        = {}  # metric -> {opt_name: [[seed_vals], ...]}
    trajectories   = {}  # opt_name -> [[seed0_snapshots], [seed1_snapshots], ...]
    final_iterates = {}  # opt_name -> [seed0_final_x, seed1_final_x, ...]

    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}")

        if build_problem is not None:
            p, x = build_problem(seed)
        else:
            p, x = problem, x0

        opts      = build_optimizers(p, x, seed)
        seed_traj = {name: [] for name in opts}

        for k in range(num_iterations):
            for name, opt in opts.items():
                # record BEFORE stepping so snapshot j sits exactly at iteration
                # j*store_every (j=0 is x0), then append the final iterate after
                # the loop — previously every point was one step late and neither
                # x0 nor the true final iterate was ever stored
                if k % store_every == 0:
                    seed_traj[name].append(opt.x.detach().clone())
                opt.step()
            if k % print_every == 0:
                print(f"  iter {k}/{num_iterations}")

        for name, opt in opts.items():
            seed_traj[name].append(opt.x.detach().clone())

        for name in opts:
            seed_metrics = {}
            for x_snap in seed_traj[name]:
                for metric, val in record_metrics(x_snap, p).items():
                    seed_metrics.setdefault(metric, []).append(val)
            for metric, vals in seed_metrics.items():
                results.setdefault(metric, {}).setdefault(name, []).append(vals)
            # final iterates are always cached (tiny) — they let a later step
            # (e.g. a GD polish for an approximate f*) start from the best end
            # point even when save_trajectory=False
            final_iterates.setdefault(name, []).append(seed_traj[name][-1])
            if save_trajectory:
                # stack into one contiguous (n_points, d) tensor instead of
                # keeping thousands of individual small tensor objects --
                # pickling many tiny tensors has heavy per-object overhead
                # (a 20-dim, store_every=1 synthetic run once produced a
                # 155MB cache from ~400K small tensors, taking 8+ seconds to
                # merely torch.load on every cache-check); a single stacked
                # tensor is both far smaller and far faster to (de)serialize.
                trajectories.setdefault(name, []).append(torch.stack(seed_traj[name]))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    to_save = {"config": config, **results, "final_iterates": final_iterates}
    if save_trajectory:
        to_save["trajectories"] = trajectories
    torch.save(to_save, cache_path)
    print(f"Saved results to {cache_path}")

    return results


def load_trajectory(cache_path):
    """Load the raw iterate snapshots saved by run_experiment(save_trajectory=True)."""
    saved = torch.load(Path(cache_path))
    return saved.get("trajectories")


def load_final_iterates(cache_path):
    """
    Load the per-optimizer final iterates from a run_experiment cache:
    {opt_name: [seed0_final_x, seed1_final_x, ...]}. Always saved (unlike
    trajectories), but absent from caches created before this was added.
    """
    saved = torch.load(Path(cache_path))
    return saved.get("final_iterates")


def subsample_metrics(results, stride):
    """
    Thin an already-computed {metric: {opt_name: [[seed_vals], ...]}} dict by
    keeping every ``stride``-th recorded point — pure slicing, no
    recomputation, so this is free at plotting time. Use to view a run at a
    coarser display density than it was stored/computed at.
    """
    if stride <= 1:
        return results
    return {
        metric: {name: [seed[::stride] for seed in seeds] for name, seeds in by_name.items()}
        for metric, by_name in results.items()
    }


def metrics_from_trajectory(trajectories, record_metrics, problem, stride=1):
    """
    Recompute {metric: {opt_name: [[seed_vals], ...]}} from stored snapshots
    (as returned by ``load_trajectory``) without rerunning the optimizer.
    ``stride`` is in units of the original ``store_every`` the snapshots were
    recorded at (stride=1 reuses every stored point; stride=k keeps every
    k-th one). Also useful to evaluate a metric that wasn't recorded in the
    original run.
    """
    results = {}
    for name, seeds in trajectories.items():
        for seed_snaps in seeds:
            seed_metrics = {}
            for x_snap in seed_snaps[::stride]:
                for metric, val in record_metrics(x_snap, problem).items():
                    seed_metrics.setdefault(metric, []).append(val)
            for metric, vals in seed_metrics.items():
                results.setdefault(metric, {}).setdefault(name, []).append(vals)
    return results


def approximate_optimum(problem, steps, lr, x0):
    """
    GD to approximate f^*
    """
    x = x0.clone()
    for k in range(steps):
        grad = problem.gradient(x)
        with torch.no_grad():
            x -= lr * grad
        if k % 500 == 0:
            print(f"  GD step {k}/{steps}  loss={problem.loss(x).item():.6f}")
    return x, problem.loss(x).item()
