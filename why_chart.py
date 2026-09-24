"""
The chart !why posts: one panel per word of a reply, showing the candidates jev weighed for it — how likely the model
thought each was, and its score once jev's penalties for repeating itself were applied. The best score was picked.
"""

import io
import math
import re

from matplotlib.figure import Figure

# Discord's dark theme, so the PNG sits in the chat without a frame
BG, INK, INK2, MUTED, GRID = "#313338", "#ffffff", "#c3c2b7", "#8a8980", "#45474d"
RAW, SCORE, PICKED = "#3a4a63", "#3987e5", "#eda100"
COLUMNS = 4
FONT = "DejaVu Sans"

# DejaVu has no emoji, which would come out as boxes
def plain(text):
    return re.sub(r"[\U00010000-\U0010ffff️]", "", text).strip()

def label(word):
    return {"\\n": "⏎ newline", "<END>": "(stop)"}.get(word, plain(word))

def clip(text, n):
    return text if len(text) <= n else text[:n - 1] + "…"

def clip_start(text, n):
    return text if len(text) <= n else "…" + text[-(n - 1):]


# panels: one per word, {"so_far": the reply before it, "picked": word, "ends": bool, "rows": [[word, prob, score]]}
# with rows best score first. Returns PNG bytes.
def render(bot_name, reply, panels):
    cols = min(len(panels), COLUMNS)
    grid_rows = math.ceil(len(panels) / cols)
    bars = max(len(p["rows"]) for p in panels)
    top = max(r[1] for p in panels for r in p["rows"])
    xmax = max(top * 100 * 1.55, 10)  # room for the "23% → 2.9%" labels

    header, panel_h = 1.25, 1.0 + 0.4 * bars
    fig = Figure(figsize=(4 * cols, header + panel_h * grid_rows + 0.3), facecolor=BG)
    fig.text(0.012, 1 - 0.3 / fig.get_figheight(), clip(f'Why {plain(bot_name)} said "{plain(reply)}"', 20 * cols),
             fontsize=18, fontweight="bold", color=INK, va="top", family=FONT)
    fig.text(0.012, 1 - 0.85 / fig.get_figheight(),
             "Wide pale bar: how likely the model thought the word was.   "
             "Thin bar: its score after the penalties for repeating itself.   The best score wins (gold).",
             fontsize=10.5, color=INK2, va="top", family=FONT, wrap=True)
    fig.subplots_adjust(left=0.1 if cols > 1 else 0.3, right=0.98, bottom=0.3 / fig.get_figheight(),
                        top=1 - (header + 0.55) / fig.get_figheight(), wspace=0.6, hspace=0.9 / panel_h * 2)

    for i, p in enumerate(panels):
        ax = fig.add_subplot(grid_rows, cols, i + 1)
        ax.set_facecolor(BG)
        rows = p["rows"][::-1]
        y = range(len(rows))
        picked = [w == p["picked"] for w, _, _ in rows]
        ax.barh(y, [pr * 100 for _, pr, _ in rows], height=0.72, color=RAW, zorder=1)
        ax.barh(y, [s * 100 for _, _, s in rows], height=0.36,
                color=[PICKED if k else SCORE for k in picked], zorder=2)
        for yy, (w, pr, s), k in zip(y, rows, picked):
            same = round(pr * 100, 1) == round(s * 100, 1)
            ax.text(pr * 100 + xmax * 0.02, yy, f"{pr:.0%}" if same else f"{pr:.0%} → {s * 100:.1f}%",
                    va="center", fontsize=9.5, color=INK if k else INK2, fontweight="bold" if k else "normal",
                    family=FONT)
        ax.tick_params(colors=MUTED, length=0, labelsize=9)
        ax.set_yticks(list(y), [clip(label(w), 14) for w, _, _ in rows], fontsize=11, family=FONT)
        for t, k in zip(ax.get_yticklabels(), picked):
            t.set_color(PICKED if k else INK)
            t.set_fontweight("bold" if k else "normal")
        ax.set_ylim(-0.6, bars - 0.4)  # same bar height in every panel, however many rows it has
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)
        so_far = f"{plain(bot_name)}: {clip_start(plain(p['so_far']), 22)}".rstrip()
        note = "  ·  stop + . ! ? pooled" if p["ends"] else ""
        ax.set_title(f"word {i + 1}{note}\n{so_far} ___", loc="left", fontsize=11, color=INK2, pad=8, family=FONT)

    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=150, facecolor=BG)
    return out.getvalue()
