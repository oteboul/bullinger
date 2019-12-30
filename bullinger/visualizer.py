import matplotlib
from matplotlib import lines
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np


class AnnotationsVisualizer(object):

    def __init__(
            self, va, height=8, width=12, actor_height=0.7, cmap='tab20'):
        self.va = va
        self.df = self.va.df
        self.width = width
        self.height = height
        self.actor_height = actor_height
        self.cmap = matplotlib.cm.get_cmap(cmap)

    def chronogram(self):
        actors = {a: i for (i, a) in enumerate(self.df.actor.unique())}
        tags = self.df.tag.unique()
        tags = {t: (i + 1) / (len(tags)+1) for (i, t) in enumerate(tags)}

        plt.figure(figsize=(self.width, self.height))
        ax = plt.gca()

        h = self.actor_height
        for i, row in self.df.iterrows():
            rect = patches.Rectangle(
                (row.start, actors[row.actor] - h / 2), row.duration, h,
                linewidth=1, edgecolor='k', facecolor=self.cmap(tags[row.tag])
            )
            ax.add_patch(rect)

        for i in actors.values():
            plt.plot([0, self.va.max_x], [i, i], 'k--', alpha=0.4)
        for i, row in self.df.iterrows():
            plt.plot(
                [row.start, row.start],
                [-h / 2, np.max(list(actors.values())) + h / 2],
                'k--', alpha=0.2)

        custom_lines = [
            lines.Line2D([0], [0], color=self.cmap(t), lw=4)
            for t in tags.values()
        ]
        ax.legend(
            custom_lines, list(tags.keys()),
            fontsize=16, loc='upper right', bbox_to_anchor=(1.2, 0.8), ncol=1
        )

        plt.xlabel('temps (sec)', fontsize=22)
        plt.yticks(np.arange(len(actors)), list(actors.keys()), fontsize=18)
        plt.xticks(fontsize=18)
        plt.title(self.df.video_id.unique()[0], fontsize=24)
        return ax

    def activity(self, num_points=1000, relative_sigma=0.01):
        curves = {
            'adulte': self.va.to_activity(
                'adulte,adulte bis', num_points, relative_sigma),
            'bébé': self.va.to_activity('bébé', num_points, relative_sigma)
        }

        plt.figure(figsize=(self.width, self.height))
        ax = plt.gca()
        for label, curve in curves.items():
            plt.plot(curve[:, 0], curve[:, 1], lw=3, label=label)
        y_max = np.max([np.max(c[:, 1]) for c in curves.values()])
        colors = {'inv': 'k', 'sans': 'red', 'avec': 'green'}
        for i, row in self.va.get_df('contexte').iterrows():
            rect = patches.Rectangle(
                (row.start, 0), row.duration, y_max,
                linewidth=1, edgecolor='k', facecolor=colors[row.tag],
                alpha=0.1,
            )
            ax.add_patch(rect)

        plt.xlabel('temps (sec)', fontsize=22)
        plt.xticks(fontsize=18)
        plt.title(self.df.video_id.unique()[0], fontsize=24)
        plt.legend(fontsize=18)
        return ax
