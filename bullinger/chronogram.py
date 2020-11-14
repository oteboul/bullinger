import collections
from typing import Optional, Sequence

import matplotlib
from matplotlib import lines
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import portion

from bullinger import video


class Chronogram:

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

    def __init__(self,
                 height: int = 6,
                 width: int = 9,
                 actor_height: float = 0.6,
                 cmap: str = 'tab10',
                 tags: Optional[Sequence[str]] = None):
        self.width = width
        self.height = height
        self.actor_height = actor_height
        self.cmap = matplotlib.cm.get_cmap(cmap)
        self.tags = tags

    def make(self, v: video.Video, ax=None, legend=True):
        h = self.actor_height
        if ax is None:
            plt.figure(figsize=(self.width, self.height))
            ax = plt.gca()

        actors = sorted(v.actors_df.actor.unique())
        heights = {a: h * (i + 0.5 + 0.5 * int(a=="bebe"))
                   for (i, a) in enumerate(actors)}        
        max_height = np.max(list(heights.values()))
        tags = v.actors_df.tag.unique() if self.tags is None else self.tags
        tags = {t: (i + 1) / (len(tags) + 1) for (i, t) in enumerate(tags)}
        longest_tags = v.df.groupby(['actor', 'tag']).agg({'duration': np.max})

        for i, row in v.actors_df.iterrows():
            is_longest = longest_tags.loc[row.actor, row.tag].duration == row.duration
            self._add_box(row, heights[row.actor], tags, is_longest, ax)
            ax.plot([row.start, row.start], [-h / 2, max_height + h / 2],
                    'k--', alpha=0.2)

        adult = self.ADULT
        yticks = {adult: np.mean([y for (a, y) in heights.items() if a.startswith(adult)])}
        yticks.update({k: v for (k, v) in heights.items() if not k.startswith(adult)})
        for y in yticks.values():
            ax.plot([0, v.end], [y, y], 'k--', alpha=0.4)
        self._add_context_background(v.context_df, ax)
        ax.set_yticklabels(list(yticks.keys()), fontsize=18)
        ax.set_yticks(list(yticks.values()))
        self._beautify(v, ax)
        if legend:
            self._add_legend(ax, tags)
        return ax

    def _add_box(self, row: pd.Series, height: int, tags, show_text: bool, ax):
        """Create a single box on the axis."""
        c = self.ADULT_COLOR if self.ADULT in row.actor else self.BABY_COLOR
        h = self.actor_height
        rect = patches.Rectangle(
            (row.start, height - h / 2), row.duration, h,
            linewidth=1, edgecolor='k', facecolor=self.cmap(tags.get(row.tag)),
            zorder=5
        )
        ax.add_patch(rect)
        if show_text:
            ax.text(row.start + row.duration / 2, height, row.tag,
                    fontdict=self.FONT, ha='center', va='center', zorder=10)

    def _add_legend(self, ax, tags):
        """Adds a custom legend."""
        custom = []
        labels = []
        for t, p in sorted(tags.items()):
            custom.append(lines.Line2D([0], [0], color=self.cmap(p), lw=7))
            labels.append(t)
            ax.legend(custom, labels, bbox_to_anchor=(1, 0.70), fontsize=15)

    def _get_context_color(self, row):
        if row.context == self.INVISIBLE:
            return 'k'
        if row.context == self.WITHOUT:
            return  'r'

        cnt = row.context.count(',') + 1
        return self.SUPPORT_COLORS[cnt]

    def _add_context_background(self, df, ax):
        y_bounds = ax.get_ylim()
        ax.set_ylim(y_bounds[0], y_bounds[1] + 0.5)
        y_bounds = ax.get_ylim()
        for _, row in df.iterrows():    
            color = self._get_context_color(row)
            rect = patches.Rectangle(
                (row.start, y_bounds[0]), row.duration, np.diff(y_bounds),
                linewidth=1, edgecolor='k', facecolor=color,
                alpha=0.2, zorder=1)
            ax.add_patch(rect)
            if row.duration < 0.5 or row.context == self.WITHOUT:
                continue
            font = {k: v for (k, v) in self.FONT.items()}
            font['color'] = color
            font['weight'] = 'bold'
            ax.text(row.start + (row.duration) / 2, y_bounds[1]*0.90,
                    '\n'.join(row.context.split(', ')), fontdict=font,
                    ha='center',  va='center', zorder=10)

    def _beautify(self, v, ax):
        ax.set_xlim(v.start, v.end)
        ax.set_xlabel('temps (sec)', fontsize=22)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        ax.set_title(v.vid, fontsize=22)