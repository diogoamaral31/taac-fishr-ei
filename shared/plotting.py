"""Estilo comum dos gráficos."""
import matplotlib.pyplot as plt

COLORS = {
    "erm": "#7f7f7f",
    "fishr": "#1f77b4",
    "fishr_random_envs": "#d62728",
    "eiil_irm": "#ff7f0e",
    "eiil_fishr": "#2ca02c",
}


def set_style():
    plt.rcParams.update({
        "figure.figsize": (7, 4), "figure.dpi": 120, "savefig.dpi": 200,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
        "legend.frameon": False,
    })


def save(fig, name):
    """Guarda PNG e PDF em results/figures/."""
    from pathlib import Path
    out = Path(__file__).resolve().parents[1] / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
