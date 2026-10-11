"""Context window and cache bars, using only token counts reported by the API. The numbers are in the embed's
fields, so the bars are only named after them."""

import io

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

BG, INK = "#2b2d31", "#f2f3f5"
FREE, USED, CACHED = "#41434a", "#8796ff", "#76c5ac"


def percentage(part, total):
    fraction = part / total if total else 0
    return "<0.1%" if 0 < fraction < .001 else f"{fraction:.1%}"


def cache_known(tokens, cached):
    return cached is not None and 0 <= cached <= tokens


def render(tokens, window, cached=None):
    """Return a PNG. Cached tokens are part of input, rather than extra window usage."""
    bars = [("Context window", tokens / window, USED)]
    if cache_known(tokens, cached):
        bars.append(("Input cache", cached / tokens if tokens else 0, CACHED))
    fig = Figure(figsize=(8, .1 + .75 * len(bars)), facecolor=BG)
    ax = fig.add_axes((.02, 0, .96, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(len(bars) + .45, .4)
    ax.axis("off")
    for row, (title, fraction, color) in enumerate(bars, 1):
        ax.text(0, row - .3, title, fontsize=15, color=INK, va="center", fontfamily="DejaVu Sans",
                fontweight="bold")
        ax.add_patch(Rectangle((0, row), 1, .25, color=FREE, linewidth=0))
        ax.add_patch(Rectangle((0, row), min(max(fraction, 0), 1), .25, color=color, linewidth=0))
    with io.BytesIO() as output:
        fig.savefig(output, format="png", dpi=120, facecolor=BG)
        return output.getvalue()
