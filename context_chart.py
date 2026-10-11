"""Context window and cache bars, using only token counts reported by the API."""

import io

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

BG, INK, MUTED = "#2b2d31", "#f2f3f5", "#b5bac1"
FREE, USED, CACHED = "#41434a", "#8796ff", "#76c5ac"


def percentage(part, total):
    fraction = part / total if total else 0
    return "<0.1%" if 0 < fraction < .001 else f"{fraction:.1%}"


def render(tokens, window, cached=None):
    """Return a PNG. Cached tokens are part of input, rather than extra window usage."""
    cache_known = cached is not None and 0 <= cached <= tokens
    fig = Figure(figsize=(8, 4.2 if cache_known else 2.2), facecolor=BG)
    ax = fig.add_axes((.04, .04, .92, .92))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def text(x, y, value, *, size=16, color=INK, align="left", bold=False):
        ax.text(x, y, value, fontsize=size, color=color, ha=align, va="center",
                fontfamily="DejaVu Sans", fontweight="bold" if bold else "normal")

    def section(top, title, label, fraction, color, left, right):
        text(0, top, title, size=19, bold=True)
        text(1, top, label, color=MUTED, align="right")
        y, height = top - (.16 if cache_known else .30), .065 if cache_known else .12
        ax.add_patch(Rectangle((0, y), 1, height, color=FREE, linewidth=0))
        ax.add_patch(Rectangle((0, y), min(max(fraction, 0), 1), height, color=color, linewidth=0))
        text(0, y - (.075 if cache_known else .14), left, size=15, color=MUTED)
        text(1, y - (.075 if cache_known else .14), right, size=15, color=MUTED, align="right")

    section(.87, "Model window", f"{percentage(tokens, window)} used", tokens / window, USED,
            f"{tokens:,} input", f"{max(window - tokens, 0):,} free")
    if cache_known:
        section(.37, "This input", f"{percentage(cached, tokens)} cached", cached / tokens if tokens else 0,
                CACHED, f"{cached:,} cached", f"{tokens - cached:,} uncached")
    with io.BytesIO() as output:
        fig.savefig(output, format="png", dpi=120, facecolor=BG)
        return output.getvalue()
