import collections
import matplotlib
from matplotlib import lines
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np


class AnnotationsVisualizer(object):
    CONTEXT = 'contexte'

    def __init__(
            self, va, height=8, width=12, actor_height=0.7, cmap='tab20',
            tags=None):
        self.va = va
        self.df = self.va.df
        self.width = width
        self.height = height
        self.actor_height = actor_height
        self.cmap = matplotlib.cm.get_cmap(cmap)
        self.tags = tags

    def chronogram(self, ax=None):
        df = self.df[self.df.actor != self.CONTEXT]
        actors = {a: i for (i, a) in enumerate(df.actor.unique())}
        tags = df.tag.unique() if self.tags is None else self.tags
        tags = {t: (i + 1) / (len(tags)+1) for (i, t) in enumerate(tags)}

        if ax is None:
            plt.figure(figsize=(self.width, self.height))
            ax = plt.gca()

        h = self.actor_height
        for i, row in self.df.iterrows():
            if row.actor in actors:
                rect = patches.Rectangle(
                    (row.start, actors[row.actor] - h / 2), row.duration, h,
                    linewidth=1, edgecolor='k', facecolor=self.cmap(tags[row.tag]),
                    zorder=10
                )
                ax.add_patch(rect)

        for i in actors.values():
            ax.plot([0, self.va.max_x], [i, i], 'k--', alpha=0.4)
        for i, row in self.df.iterrows():
            if row.actor in actors:
                ax.plot(
                    [row.start, row.start],
                    [-h / 2, np.max(list(actors.values())) + h / 2],
                    'k--', alpha=0.2)

        self.add_context_background(ax)

        leg_tags = collections.OrderedDict(sorted(
            {k:v for (k, v) in tags.items() if k in df.tag.unique()}.items()))
        custom_lines = [
            lines.Line2D([0], [0], color=self.cmap(v), lw=4)
            for v in leg_tags.values()

        ]
        ax.legend(
            custom_lines, list(leg_tags.keys()),
            fontsize=16, loc='upper right', bbox_to_anchor=(1.2, 0.8), ncol=1
        )

        ax.set_yticklabels(list(actors.keys()), fontsize=18)
        ax.set_yticks(np.arange(len(actors)))
        self.set_titles(ax)
        return ax

    def add_context_background(self, ax):
        colors = {'inv': 'k', 'sans': 'red', 'avec': 'green'}
        y_bounds = ax.get_ylim()
        for i, row in self.va.get_df(self.CONTEXT).iterrows():
            rect = patches.Rectangle(
                (row.start, y_bounds[0]), row.duration, np.diff(y_bounds),
                linewidth=1, edgecolor='k', facecolor=colors[row.tag],
                alpha=0.1, zorder=1)
            ax.add_patch(rect)

    def set_titles(self, ax):
        ax.set_xlim(self.va.min_x, self.va.max_x)
        ax.set_xlabel('temps (sec)', fontsize=22)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        ax.set_title(self.df.video_id.unique()[0], fontsize=24)

    def activity(self, ax=None, num_points=1000, relative_sigma=0.01):
        curves = {
            'adulte': self.va.to_activity(
                'adulte,adulte bis', num_points, relative_sigma),
            'bébé': self.va.to_activity('bébé', num_points, relative_sigma)
        }

        if ax is None:
            plt.figure(figsize=(self.width, self.height))
            ax = plt.gca()

        for label, curve in curves.items():
            ax.plot(curve[:, 0], curve[:, 1], lw=3, label=label)
        self.add_context_background(ax)

        self.set_titles(ax)
        ax.legend(prop=dict(size=18))
        return ax

    def stimulus(self, ax=None):
        if ax is None:
            plt.figure(figsize=(self.width, self.height))
            ax = plt.gca()

        d = self.va.stimulus_distribution
        ax.bar(d.keys(), d.values)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
            tick.label.set_rotation(45)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        ax.legend(prop=dict(size=18))
        return ax
