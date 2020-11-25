import collections
from typing import Optional, Sequence, Union

import matplotlib
from matplotlib import lines
from matplotlib import patches
from matplotlib import ticker

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import portion

from bullinger import video


CONTEXT = 'contexte'
INVISIBLE = 'inv'
WITHOUT = 'sans'
ADULT = 'adulte'
RME = 'rme'
ADULT_COLOR = '#f2ae27'
BABY_COLOR = '#ebb5e6'
# Color as a function of the number of supports
SUPPORT_COLORS = ['o', '#abab03', '#32a852', '#117d99', '#2517a3', '#abab03']
FONT = {
    'family': 'sans',
    'color':  'k',
    'weight': 'normal',
    'size': 14,
}


def plot(v: Union[video.Video, pd.DataFrame],
         ax: Optional[matplotlib.axes.Axes] = None,
         actor_height: float = 0.6,
         cmap: str = 'tab10',
         tags: Optional[Sequence[str]] = None,
         legend: bool = True):
    """Plot a chronogram."""
    if isinstance(v, pd.DataFrame):
        v = video.Video(df=df)

    h = actor_height
    cmap = matplotlib.cm.get_cmap(cmap)
    tags = v.actors_df.tag.unique() if tags is None else tags
    tags = {t: (i + 1) / (len(tags) + 1) for (i, t) in enumerate(tags)}
    if ax is None:
        plt.figure(figsize=(9, 5))
        ax = plt.gca()

    def _add_box(row: pd.Series, height: int, show_text: bool):
        """Create a single box on the axis."""
        c = ADULT_COLOR if ADULT in row.actor else BABY_COLOR
        rect = patches.Rectangle(
            (row.start, height - h / 2), row.duration, h,
            linewidth=1, edgecolor='k', facecolor=cmap(tags.get(row.tag)),
            zorder=5
        )
        ax.add_patch(rect)
        if show_text:
            ax.text(row.start + row.duration / 2, height, row.tag,
                    fontdict=FONT, ha='center', va='center', zorder=10)

    def _add_legend():
        """Adds a custom legend."""
        custom = []
        labels = []
        for t, p in sorted(tags.items()):
            custom.append(lines.Line2D([0], [0], color=cmap(p), lw=7))
            labels.append(t)
            ax.legend(custom, labels, bbox_to_anchor=(1, 0.70), fontsize=15)

    def _get_context_color(row):
        if row.context == INVISIBLE:
            return 'k'
        if row.context == WITHOUT:
            return  'r'
        cnt = row.context.count(',') + 1
        return SUPPORT_COLORS[cnt]

    def _add_context_background(df):
        y_bounds = ax.get_ylim()
        ax.set_ylim(y_bounds[0], y_bounds[1] + 0.5)
        y_bounds = ax.get_ylim()
        for _, row in df.iterrows():    
            color = _get_context_color(row)
            rect = patches.Rectangle(
                (row.start, y_bounds[0]), row.duration, np.diff(y_bounds),
                linewidth=1, edgecolor='k', facecolor=color,
                alpha=0.2, zorder=1)
            ax.add_patch(rect)
            if row.duration < 0.5 or row.context == WITHOUT:
                continue
            font = {k: v for (k, v) in FONT.items()}
            font['color'] = color
            font['weight'] = 'bold'
            ax.text(row.start + (row.duration) / 2, y_bounds[1]*0.90,
                    '\n'.join(row.context.split(',')), fontdict=font,
                    ha='center',  va='center', zorder=10)

    def _beautify():
        ax.set_xlim(v.start, v.end)
        ax.set_xlabel('temps (sec)', fontsize=22)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        ax.set_title(v.vid, fontsize=22)

    actors = sorted(v.actors_df.actor.unique())
    heights = {a: h * (i + 0.5 + 0.5 * int(a=="bebe"))
                for (i, a) in enumerate(actors)}        
    max_height = np.max(list(heights.values()))
    longest_tags = v.df.groupby(['actor', 'tag']).agg({'duration': np.max})

    for i, row in v.actors_df.iterrows():
        is_longest = longest_tags.loc[row.actor, row.tag].duration == row.duration
        _add_box(row, heights[row.actor], is_longest)
        ax.plot([row.start, row.start], [-h / 2, max_height + h / 2],
                'k--', alpha=0.2)

    yticks = {ADULT: np.mean([y for (a, y) in heights.items() if a.startswith(ADULT)])}
    yticks.update({k: v for (k, v) in heights.items() if not k.startswith(ADULT)})
    for y in yticks.values():
        ax.plot([0, v.end], [y, y], 'k--', alpha=0.4)
    _add_context_background(v.context_df)
    ax.set_yticklabels(np.array(list(yticks.keys())), fontsize=18)
    ax.set_yticks(np.array(list(yticks.values())))
    _beautify()
    if legend:
        _add_legend()
    return ax
