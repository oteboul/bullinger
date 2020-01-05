import collections
import difflib
import matplotlib
from matplotlib import lines
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bullinger.annotations import VideoAnnotations


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

    def chronogram(self, ax=None, add_legend=True):
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

        if add_legend:
            leg_tags = collections.OrderedDict(sorted(
                {
                    k: v for (k, v) in tags.items()
                    if k in df.tag.unique()
                }.items()))
            custom_lines = [
                lines.Line2D([0], [0], color=self.cmap(v), lw=4)
                for v in leg_tags.values()

            ]
            ax.legend(
                custom_lines, list(leg_tags.keys()),
                fontsize=16, loc='upper right',
                bbox_to_anchor=(1.2, 0.6), ncol=1
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
        scores = []
        for name, ms in zip(['avec', 'sans'], [m_a, m_s]):
            if not np.isnan(ms[1]):
                scores.append("{} {:.2f}".format(name, ms[2]))
        scores = " | ".join(scores)

        ax.set_title(
            "{}. ({}), Semestre {}\n"
            "Stimulus: {:.0%}, Reponse: {:.0%}".format(
                self.va.baby[0], self.va.group, self.va.semester, s, r),
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


def select(df, patterns):
    masks = []
    for p in patterns:
        masks.append([p in x for x in df.index.to_numpy()])
    return df[np.any(np.stack(masks, axis=1), axis=1)]


def longuest_common(strs):
    if not strs:
        return ""

    result = strs[0]
    for curr in strs[1:]:
        match = difflib.SequenceMatcher(
            None, result, curr).find_longest_match(0, len(result), 0, len(curr))
        result = result[match.a: match.a + match.size]
    return result


class AggregationVisualizer(object):
    NAME_MAP = {
        'resp': 'réponse',
        'instal': 'installation',
        'stimul': 'stimulation',
        'all': 'général',
        'semester': 'semestre'
    }

    def __init__(self, agg, height=8, width=12):
        self.agg = agg
        self.df = agg.metrics_df
        self.height = height
        self.width = width

    def baby(self, name, idx=None, **kwargs):
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
            viewer.chronogram(ax=ax, **kwargs)
        if k > 1:
            for i in range(1, k):
                axes[i].get_yaxis().set_ticks([])

    def scatter(
            self, semesters=None, installations=None, plot_score=False,
            plot_means=False, cmap='viridis', median=False):
        if installations is None:
            installations = ['all', 'avec', 'sans']
        if semesters is None:
            semesters = [1, 2]
        if not isinstance(semesters, list):
            semesters = [semesters]
        if not isinstance(installations, list):
            installations = [installations]

        num_instals = len(installations)
        num_semester = len(semesters)

        fig, axes = plt.subplots(
            num_instals, num_semester,
            figsize=(self.width * num_semester, self.height * num_instals))

        for i, s in enumerate(semesters):
            for j, installation in enumerate(installations):
                if num_semester > 1 and num_instals > 1:
                    ax = axes[i, j]
                elif num_instals == 1 and num_instals == 1:
                    ax = axes
                else:
                    ax = axes[i+j]

                s_df = self.df[(self.df.semester == s)]
                stimul_col = 'stimu_{}'.format(installation)
                resp_col = 'resp_{}'.format(installation)
                instal_col = 'instal_{}'.format(installation)
                s_df = s_df[
                    s_df[instal_col] > VideoAnnotations.INSTALLATION_THRESHOLD]
                s_df = s_df[(s_df[stimul_col] > 0.2) | (s_df[resp_col] > 0.2)]
                for ad in [False, True]:
                    df = s_df[s_df.ad == ad]
                    sc = ax.scatter(
                        df[stimul_col], df[resp_col],
                        s=200, ec='k',
                        label='AD' if ad else 'TD', zorder=3)
                    if plot_means:
                        mean_fn = np.median if median else np.mean
                        sc2 = ax.scatter(
                            mean_fn(df[stimul_col]),
                            mean_fn(df[resp_col]),
                            s=300, marker='s', ec='k', color=sc.get_facecolor(),
                            zorder=3)
                        mu = sc2.get_offsets()[0].data
                        color = sc.get_facecolor()[0]
                        ax.plot(
                            [0, mu[0]], [0, mu[1]], linestyle='-', zorder=3,
                            lw=3, color=color)

                        angle_rad = np.arctan(mu[1]/mu[0])
                        angle = angle_rad * 180 / np.pi
                        arc_w = np.sqrt(np.sum(mu**2)) / 2**ad
                        txt_pos = 0.55 * arc_w * np.array(
                            [np.cos(angle_rad/2), np.sin(angle_rad/2)])

                        for ann_size, ann_col in [(27, 'k'), (26, color)]:
                            ax.annotate(
                                "{:.0f}°".format(angle),
                                txt_pos,
                                color=ann_col, fontsize=ann_size, zorder=3)

                        angle_plot = patches.Arc(
                            [0, 0], arc_w, arc_w, theta1=0.0, theta2=angle,
                            lw=3, color=color, zorder=3)
                        ax.add_patch(angle_plot)

                for _, row in s_df.iterrows():
                    ax.annotate(
                        "{}.".format(row.baby[0]),
                        (0.01 + row[stimul_col], 0.015 + row[resp_col]),
                        fontsize=13, color=(1.0, 1.0, 1.0)
                    )

                m = 1.05
                x = np.linspace(0, m, 1000)
                X, Y = np.meshgrid(x, x)
                Z = np.ones(X.shape)
                alpha = 0.0
                if plot_score:
                    Z = VideoAnnotations.score(X, Y)
                    alpha = 0.8
                im = ax.imshow(
                    Z, extent=[0, m, 0, m], origin='lower',
                    alpha=alpha, zorder=1, cmap=cmap)
                if plot_score:
                    fig.colorbar(im, ax=ax)
                ax.set_autoscale_on(False)

                if not plot_means:
                    ax.plot([0.0, m], [0, m], 'k--', alpha=0.9, zorder=2)
                ax.legend(fontsize=18)
                ax.set_title(
                    'Semestre {}, {} installation'.format(s, installation),
                    fontsize=18)
                ax.set_xlabel('Temps relatif de stimulation', fontsize=18)
                ax.set_ylabel('Temps relatif de réponse', fontsize=18)
        return axes

    def distribution(self, installation='all'):
        fig, axes = plt.subplots(1, 2, figsize=(2 * self.width, self.height))
        for i, s in enumerate([1, 2]):
            ax = axes[i]

            for ad in [False, True]:
                df = self.df[(self.df.ad == ad) & (self.df.semester == s)]
                h = df['score_{}'.format(installation)]

                # The bar plot
                vs, edges = np.histogram(h, density=True)
                label = 'AD' if ad else 'TD'
                bin_width = 0.1
                num_bins = int(np.ceil((np.max(h) - np.min(h)) / bin_width))
                ax.hist(
                    h, bins=num_bins,
                    label="{}: {:.2f}".format(label, np.nanmedian(h)),
                    alpha=0.8)

            ax.set_title('Semester {}'.format(s), fontsize=20)
            ax.legend(fontsize=18)
            lab = 'Taux de réponse'
            ax.set_xlabel('{} ({})'.format(lab, installation), fontsize=20)

    def stimuli(self, relative=False):
        fig, axes = plt.subplots(
            2, 1, figsize=(self.width * 1, self.height * 2), sharey='col')
        for j, s in enumerate(range(1, 3)):
            ax = axes[j]
            ad, v_ad = self.agg.average_stimulus(
                semester=s, relative=relative, autists=True)
            td, v_td = self.agg.average_stimulus(
                semester=s, relative=relative, autists=False)
            pd.DataFrame({'TD': td, 'AD': ad}).plot.bar(
                rot=45, fontsize=18, ax=ax)
            ax.legend(fontsize=18)
            ax.set_title('Semestre {}'.format(s), fontsize=22)
            ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(14)

    def stats_in_bar(self, pattern='resp', s=None, ax=None, percentage=True):
        df = self.agg.responses(as_index=True)

        df2 = df.unstack().transpose()
        if s is not None:
            df2 = df2.iloc[df2.index.get_level_values('semester') == s]
        df2 = select(df2, pattern.split(','))
        df2.columns = ['TD', 'AD']

        if ax is None:
            plt.figure(figsize=(8, 5))
            ax = plt.gca()

        df2.plot.bar(rot=0, ax=ax)
        ax.legend(fontsize=14)

        labels = []
        for x in df2.index.to_numpy():
            parts = []
            if isinstance(x, tuple):
                parts = [df2.index.levels[1].name, str(x[1])]
                x = x[0]
            parts += list(reversed(x.split('_')))

            labels.append(' '.join(
                [self.NAME_MAP.get(t, t).title() for t in parts]))

        title = longuest_common(labels)
        ax.set_title(title, fontsize=20)
        ax.set_xticklabels(
            [lab.replace(title, '') for lab in labels], fontsize=16)
        if percentage:
            ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
