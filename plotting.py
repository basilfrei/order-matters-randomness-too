import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.ticker import MultipleLocator

matplotlib.rcParams["pdf.fonttype"] = 42   # embed fonts properly in PDF
matplotlib.rcParams["ps.fonttype"]  = 42

COLORS = {
    "SGD":  "#0072B2",  # blue
    "SO":   "#D55E00",  # vermillion
    "IG":   "#009E73",  # green
    "RR":   "#CC79A7",  # magenta/pink
}
LINESTYLES = {
    "SGD": "-",
    "SO":  "-",
    "IG":  "-",
    "RR":  "-",
}

_fallback_colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# Draw order of the curves, front to back: SO > RR = SGD > IG.
# Unknown names fall back to 2.0 (matplotlib's default line zorder).
Z_ORDER = {"SGD": 2.4, "RR": 2.4, "SO": 2.6, "IG": 2.0}

# Legend entries always appear in this order; unknown names go last, in draw order.
LEGEND_ORDER = ["SGD", "RR", "SO", "IG"]

# Keys baked into plot filenames, in this order; configs lacking a key skip it.
_TAG_KEYS = [
    ("n", "n"),
    ("d", "d"),
    ("hidden_dim", "h"),
    ("reg", "reg"),
    ("batch_size", "bs"),
    ("lr", "lr"),
    ("lr0", "lr0"),
    ("lr_max", "lrmax"),
    ("decay_rr_ig_so", "dso"),
    ("decay_sgd", "dsgd"),
    ("n_seeds", "seeds"),
]


def _tag_value(v):
    if v is None:
        return "none"
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, (list, tuple)):
        return "x".join(_tag_value(item) for item in v)
    if isinstance(v, str):
        return "".join(c for c in v if c.isalnum())
    return str(v)


def config_to_tag(config):
    """Filename-safe tag with the core hyperparameters of a config dict:
    n, d, hidden widths, reg, batch size, step sizes, decay coefficients,
    seeds. Missing keys and empty values (e.g. hidden_dim=[]) are skipped."""
    return "_".join(abbrev + _tag_value(config[key])
                    for key, abbrev in _TAG_KEYS
                    if key in config and _tag_value(config[key]) != "")


def _draw_panel(ax, loss_dict, ylabel, epoch_scale, xlabel, xtick_step=None):
    for name, losses in loss_dict.items():
        z = Z_ORDER.get(name, 2.0)
        color = COLORS.get(name) or next(_fallback_colors)
        linestyle = LINESTYLES.get(name, "-")

        if losses and isinstance(losses[0], (list, np.ndarray)):
            # ---- multi-seed ----
            y = np.array(losses, dtype=float)           # (n_seeds, n_points)
            iters = np.arange(y.shape[1]) * epoch_scale

            # log-space statistics so the band is symmetric on the log axis
            y_safe   = np.maximum(y, 1e-15)
            log_y    = np.log(y_safe)
            mean_log = np.mean(log_y, axis=0)
            std_log  = np.std(log_y,  axis=0)
            mean  = np.exp(mean_log)
            upper = np.exp(mean_log + std_log)
            lower = np.exp(mean_log - std_log)

            line, = ax.plot(iters, mean, label=name, color=color, linestyle=linestyle, linewidth=2, zorder=z)
            # band sits just behind its own line but in front of the next method
            ax.fill_between(iters, lower, upper, alpha=0.25, color=line.get_color(),
                            zorder=z - 0.05)
        else:
            # ---- single seed ----
            iters = np.arange(len(losses)) * epoch_scale
            ax.plot(iters, losses, label=name, color=color, linestyle=linestyle, linewidth=2, zorder=z)

    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)
    if xtick_step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(xtick_step))
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    rank = {name: i for i, name in enumerate(LEGEND_ORDER)}
    pairs = sorted(zip(handles, labels), key=lambda hl: rank.get(hl[1], len(rank)))
    if pairs:
        handles, labels = zip(*pairs)
    ax.legend(handles, labels, loc="upper right", fontsize=11, framealpha=0.9, edgecolor="gray")


def _finish_figure(save_path):
    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_losses(loss_dict, ylabel, epoch_scale=1, xlabel="Iteration", save_path=None,
                xtick_step=None):
    """
    loss_dict values:
        [float, ...]             single seed, plots a plain line
        [[float, ...], ...]      multi-seed (outer index = seed),
                                 plots mean ± 1 std in log-space as a shaded band

    epoch_scale : x-axis units per plotted point (shifts point index to true units, e.g. epochs)
    save_path   : if given, saves PDF/PNG to that path (directory is created if needed)
    xtick_step  : if given, x-axis ticks at multiples of this value
    """

    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_panel(ax, loss_dict, ylabel, epoch_scale, xlabel, xtick_step)
    _finish_figure(save_path)
