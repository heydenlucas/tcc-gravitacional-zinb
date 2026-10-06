import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

CM = 1/2.54
TEXT_W = 16.0 * CM          # largura util da pagina (21 - 2 x 2,5 cm)
FONT = "Liberation Sans"   # metricamente identica a Arial

def setup():
    plt.rcParams.update({
        "font.family": FONT, "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.linewidth": 1.5, "axes.edgecolor": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "text.color": "black",
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8, "ytick.minor.width": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "axes.facecolor": "white", "figure.facecolor": "white",
        "legend.frameon": False, "savefig.dpi": 300, "savefig.facecolor": "white",
        "axes.titlepad": 2, "mathtext.fontset": "custom",
        "mathtext.rm": FONT, "mathtext.it": FONT + ":italic", "mathtext.bf": FONT + ":bold",
        "axes.unicode_minus": True,
    })

def fmt_br(x, dec=0):
    """Numero no padrao brasileiro: ponto como milhar, virgula como decimal."""
    s = f"{x:,.{dec}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    s = s.replace("-", "−")
    return s

def br_axis(ax, axis="y", dec=0):
    f = FuncFormatter(lambda v, p: fmt_br(v, dec))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)

def panel(ax, letter, x=-0.02, y=1.02):
    """Identificacao do painel: letra maiuscula, canto superior esquerdo, sem pontuacao."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="right" if x < 0 else "left")

def save(fig, name):
    import os
    pasta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")
    os.makedirs(pasta, exist_ok=True)
    fig.savefig(os.path.join(pasta, f"{name}.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
