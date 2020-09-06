import collections
import matplotlib
from matplotlib import lines
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import portion


class AnnotationsVisualizer(object):

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

    def __init__(self, height=8, width=12, actor_height=0.6, cmap='tab10',
                 tags=None):
        self.width = width
        self.height = height
        self.actor_height = actor_height
        self.cmap = matplotlib.cm.get_cmap(cmap)
        self.tags = tags

    def new_ax(self):
        plt.figure(figsize=(self.width, self.height))
        return plt.gca()

    def chronogram(self, cohort, video_id, ax=None, legend=True):
        df = cohort.df[cohort.df.video_id == video_id]

        in_df = df[(df['actor'] != self.CONTEXT) &
                   (~df.actor.str.startswith(self.RME))]

        adults = in_df[in_df.actor.str.startswith(self.ADULT)].actor.unique()
        offset = 0
        actors = {}
        actorsticks = []
        for a in sorted(in_df.actor.unique()):
            has_adult = len([x.startswith(self.ADULT) for x in actors.keys()]) > 0
            offset += float(a not in adults) + self.actor_height * has_adult
            if has_adult and a in adults:
                actorsticks[-1][1] += self.actor_height / 2
            else:
                actorsticks.append([a, offset])
            actors[a] = offset

        tags = in_df.tag.unique() if self.tags is None else self.tags
        tags = {t: (i + 1) / (len(tags) + 1) for (i, t) in enumerate(tags)}

        if ax is None:
            ax = self.new_ax()

        h = self.actor_height
        for i, row in in_df.iterrows():
            c = self.ADULT_COLOR if self.ADULT in row.actor else self.BABY_COLOR
            if row.actor in actors:
                rect = patches.Rectangle(
                    (row.start, actors[row.actor] - h / 2), row.duration, h,
                    linewidth=1, edgecolor='k', facecolor=self.cmap(tags.get(row.tag)),
                    zorder=5
                )
                ax.add_patch(rect)
                biggest = np.max(df[(df['actor']==row.actor) & (df['tag']==row.tag)].duration)
                if row.duration == biggest:
                    ax.text(row.start + row.duration / 2, actors[row.actor],
                            row.tag, fontdict=self.FONT,
                            ha='center', va='center', zorder=10)

        x_bounds = [np.min(df.start), np.max(df.end)]
        for _, i in actorsticks:
            ax.plot([0, x_bounds[1]], [i, i], 'k--', alpha=0.4)
        for i, row in in_df.iterrows():
            if row.actor in actors:
                ax.plot(
                    [row.start, row.start],
                    [-h / 2, np.max(list(actors.values())) + h / 2],
                    'k--', alpha=0.2)

        if legend:
            custom_lines = []
            labels = []
            for t, p in sorted(tags.items()):
                custom_lines.append(lines.Line2D([0], [0], color=self.cmap(p), lw=7))
                labels.append(t)
            ax.legend(custom_lines, labels, bbox_to_anchor=(1, 0.70), fontsize=15)

        df2 = cohort.context_df
        self.add_context_background(df2[df2.video_id == video_id], ax)
        ax.set_yticklabels([x[0] for x in actorsticks], fontsize=18)
        ax.set_yticks([x[1] for x in actorsticks])
        self.set_titles(df, ax, video_id)
        return ax

    def get_context_color(self, row):
        if row.context == self.INVISIBLE:
            return 'k'
        if row.context == self.WITHOUT:
            return  'r'

        cnt = row.context.count(',') + 1
        return self.SUPPORT_COLORS[cnt]

    def add_context_background(self, df, ax):
        y_bounds = ax.get_ylim()
        ax.set_ylim(y_bounds[0], y_bounds[1] + 0.5)
        y_bounds = ax.get_ylim()

        df = df[(df['actor'] == self.CONTEXT) | (df.context.notnull())] 
        for _, row in df.iterrows():    
            color = self.get_context_color(row)
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

    def set_titles(self, df, ax, title=''):
        ax.set_xlim(np.min(df.start), np.max(df.end))
        ax.set_xlabel('temps (sec)', fontsize=22)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        ax.set_title(title, fontsize=22)

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
