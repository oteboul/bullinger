import collections
import matplotlib
from matplotlib import lines
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
                    linewidth=1, edgecolor='k',
                    facecolor=self.cmap(tags[row.tag]), zorder=10
                )
                ax.add_patch(rect)

        for i in actors.values():
            ax.plot([0, self.va.max_x], [i, i], 'k--', alpha=0.4)
        bounds = ax.get_ylim()
        for i, row in self.df.iterrows():
            if row.actor in actors:
                ax.plot([row.start, row.start], bounds, 'k--', alpha=0.2)

        self.add_context_background(ax)

        leg_tags = collections.OrderedDict(sorted(
            {k: v for (k, v) in tags.items() if k in df.tag.unique()}.items()))
        custom_lines = [
            lines.Line2D([0], [0], color=self.cmap(v), lw=4)
            for v in leg_tags.values()

        ]
        ax.legend(
            custom_lines, list(leg_tags.keys()),
            fontsize=16, loc='upper right', bbox_to_anchor=(1.2, 0.6), ncol=1
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
        r, s, ra, inst = self.va.metrics()
        m_a = self.va.metrics(installation=True)
        m_s = self.va.metrics(installation=False)
        ratios = []
        for name, ms in zip(['avec', 'sans'], [m_a, m_s]):
            if not np.isnan(ms[1]):
                ratios.append("{} {:.2f}".format(name, ms[2]))
        ratios = " | ".join(ratios)

        ax.set_title(
            "{} ({}), Semestre {}\n"
            "Stimulus: {:.2f}, Reponse: {:.2f} ({})".format(
                self.va.baby, self.va.group, self.va.semester, s, r/s, ratios),
            fontsize=24)

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


class AggregationVisualizer(object):
    def __init__(self, agg, height=8, width=12):
        self.agg = agg
        self.df = agg.metrics_df
        self.height = height
        self.width = width

    def baby(self, name, idx=None):
        vas = self.agg.per_baby[name]
        if idx is not None:
            vas = [vas[idx]]

        vas.sort(key=lambda x: x.semester)
        k = len(vas)
        fig, axes = plt.subplots(1, k, figsize=(k * self.width, self.height))
        for i, va in enumerate(vas):
            viewer = AnnotationsVisualizer(
                va, actor_height=0.6, tags=self.agg.tags)
            ax = axes[i] if k > 1 else axes
            viewer.chronogram(ax=ax)
        if k > 1:
            for i in range(1, k):
                axes[i].get_yaxis().set_ticks([])

    def scatter(self):
        fig, axes = plt.subplots(
            2, 3, figsize=(self.width * 3, self.height * 2))
        for i, s in enumerate([1, 2]):
            for j, installation in enumerate(['all', 'avec', 'sans']):
                ax = axes[i, j]
                for ad in [True, False]:
                    df = self.df[(self.df.ad == ad) & (self.df.semester == s)]
                    ax.scatter(df['stimu_{}'.format(installation)],
                               df['resp_{}'.format(installation)],
                               s=200, label='AD' if ad else 'TD', zorder=2)
                bounds = ax.get_ylim()
                ax.plot([0.0, 1.0], bounds, 'k--', alpha=0.4, zorder=1)
                ax.legend(fontsize=18)
                ax.set_title(
                    'Semestre {}, {} installation'.format(s, installation),
                    fontsize=18)
                ax.set_xlabel('Intensité du Stimulus', fontsize=18)
                ax.set_ylabel('Facteur de Réponse', fontsize=18)

    def distribution(self, installation='all'):
        fig, axes = plt.subplots(1, 2, figsize=(self.width * 2, self.height))
        for i, s in enumerate([1, 2]):
            ax = axes[i]

            for ad in [True, False]:
                df = self.df[(self.df.ad == ad) & (self.df.semester == s)]
                h = df['ratio_{}'.format(installation)]

                # The bar plot
                vs, edges = np.histogram(h, density=True)
                label = 'AD' if ad else 'TD'
                bin_width = 0.1
                num_bins = int(np.ceil((np.max(h) - np.min(h)) / bin_width))
                ax.hist(
                    h, bins=num_bins,
                    label="{}: {:.2f}".format(label, np.nanmedian(h)),
                    alpha=0.8)

            ax.set_xlim(0, 2.0)
            ax.set_title('Semester {}'.format(s), fontsize=20)
            ax.legend(fontsize=18)
            lab = 'Taux de réponse'
            ax.set_xlabel('{} ({})'.format(lab, installation), fontsize=20)

    def stimuli(self):
        fig, axes = plt.subplots(
            1, 2, figsize=(self.width * 2, self.height * 1), sharey='row')
        for j, s in enumerate(range(1, 3)):
            ax = axes[j]
            ad, v_ad = self.agg.average_stimulus(
                semester=s, relative=False, autists=True)
            td, v_td = self.agg.average_stimulus(
                semester=s, relative=False, autists=False)
            pd.DataFrame({'ad': ad, 'td': td}).plot.bar(
                rot=45, fontsize=18, ax=ax)
            ax.set_title('Semester {} (AD={} | TD={})'.format(
                s, int(v_ad), int(v_td)), fontsize=22)
