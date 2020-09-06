import datetime
import functools
import os.path
import logging
import numpy as np
import pandas as pd
import portion

from bullinger import utils


class VideoAnnotations(object):
    """The annotations of a single video, annotated by ELAN."""

    INVISIBLE = 'inv'
    SUPPORT = 'appui'
    CONTEXT = 'contexte'
    ADULT = 'adulte'
    BABY = 'bébé'
    WITHOUT = 'sans'
    COLUMNS = ['actor', 'video_id', 'start', 'end', 'duration', 'tag']

    def __init__(self, filename):
        self.filename = filename
        self.semester = 1 if '(0-6)' in filename else 2
        self.df = pd.read_csv(filename, sep='\t', header=None)
        self.ill_formed = False
        # In case there are no annotations, they are NaNs
        if np.any(pd.isnull(self.df[self.df.columns[-1]])):
            logging.error(f'Some annotations are missing in {filename}!')
            self.ill_formed = True
            return

        handle_durations = True
        if len(self.df.columns) > 6:
            self.df = self.df.drop(columns=[2, 4, 6])
            handle_durations = False

        self.df.columns = self.COLUMNS
        if handle_durations:
            for col in ['start', 'end', 'duration']:
                self.df[col] = self.df[col].apply(utils.parse_duration)

        for col in ['actor', 'tag']:
            self.df[col] = self.df[col].apply(lambda x: x.strip().lower())

        self.video_id = os.path.basename(filename)[:-4]
        self.df['video_id'] = self.video_id
        self.min_x = np.min(self.df.start)
        self.max_x = np.max(self.df.end)

    @property
    def support(self) -> pd.DataFrame:
        return self.df[
            (self.df['actor'].str.startswith(self.SUPPORT)) |
            (self.df.tag == self.INVISIBLE)]

    @property
    def baby(self) -> pd.DataFrame:
        return self.df[self.df['actor'] == self.BABY]

    @property
    def interactions(self) -> pd.DataFrame:
        return self.df[(~self.df['actor'].str.startswith(self.SUPPORT)) &
                       (self.df['actor'] != self.CONTEXT)]

    def _without_row(self) -> pd.DataFrame:
        result = {k: None for k in self.COLUMNS}
        result.update(start=self.min_x, end=self.max_x,
                      actor=self.CONTEXT + '2', tag=self.WITHOUT)
        return pd.DataFrame([result])

    def _overlap_actor_tag(self, x, y):
        """Merges two rows for tag overlap."""
        sep = ', '
        tag = sep.join([x[1], y[1]])
        actor = sep.join([x[0], y[0]]) if x[0] != y[0] else x[0]
        parts = [x.strip() for x in tag.split(sep)]
        if len(parts) > 1 and self.WITHOUT in parts:
            parts = [x for x in parts if x != self.WITHOUT]
        tag = sep.join(parts)

        if x[1] == self.INVISIBLE or y[1] == self.INVISIBLE:
            actor, tag = self.CONTEXT, self.INVISIBLE
        elif x[1] == self.WITHOUT and y[1] != self.WITHOUT and y[0] != self.BABY:
            actor, tag = y
        elif x[1] != self.WITHOUT and y[1] == self.WITHOUT and x[0] != self.BABY:
            actor, tag = x
        elif x[0] == self.BABY or y[0] == self.BABY:
            actor = self.BABY
        return actor, tag

    def overlap_df(self, df):
        result = portion.IntervalDict()
        for actor in df.actor.unique():
            a_df = df[df.actor == actor]
            curr = portion.IntervalDict()
            for i, row in a_df.iterrows():
                text = row.tag
                if row.actor.startswith(self.SUPPORT):
                    text = row.actor[len(self.SUPPORT):]
                curr[portion.closed(row.start, row.end + 0.1)] = (row.actor, text)
            result = result.combine(curr, how=self._overlap_actor_tag)
        return result

    @property
    def invisible_df(self):
        return self.df[self.df.tag == self.INVISIBLE]

    def to_context(self, with_baby=True):
        """From multiple actors for context to a single one."""
        df = self.support
        if with_baby:
            df = pd.concat([self.support, self.baby, self.invisible_df])

        df = pd.concat([df, self._without_row()])
        intervals = self.overlap_df(df)
        rows = []
        baby_tags = set(self.baby.tag.unique())
        for intervs, label in intervals.items():
            actor = label[0] if self.BABY in label[0] else self.CONTEXT
            context = [x for x in label[1].split(', ') if x not in baby_tags]
            intag = [t for t in baby_tags if t in label[1]]
            tag = intag[0] if intag else None
            support = len(context) if context != [self.WITHOUT] else 0
            for i in intervs:
                rows.append({
                    'actor': actor,
                    'video_id': self.video_id,
                    'start': i.lower,
                    'end': i.upper,
                    'duration': i.upper - i.lower,
                    'tag': tag,
                    'support': support,
                    'context': ', '.join(context)
                })
        return pd.DataFrame(sorted(rows, key=lambda x: x['start']))

    @property
    def with_context(self):
        """Merge the support actor of a single video."""
        return pd.concat([self.interactions, self.to_context()], sort=True)

    @property
    def adult_df(self):
        return self.df[self.df.actor.str.contains(self.ADULT)]

    @property
    def stimulus_distribution(self):
        return self.adult_df.groupby('tag').duration.sum()

    def get_df(self, labels):
        return self.df.query(
            ' | '.join([f'actor=="{v}"' for v in labels.split(',')]))

    @property
    def installation_df(self, presence=None):
        df = self.get_df(self.CONTEXTE)
        if presence is not None:
            tag = 'avec' if presence else 'sans'
            df = df[df.tag == tag]
        return df

    def gaussian_smoothing(self, labels, t, kernels):
        df = self.get_df(labels)
        d = np.zeros(t.shape)
        for _, row in df.iterrows():
            d += np.sum(kernels[(t >= row.start) * (t < row.end)], axis=0)
        return d

    def to_activity(self, labels, num_points=1000, relative_sigma=0.01):
        t = np.linspace(self.min_x, self.max_x, num_points)
        s = relative_sigma * (self.max_x - self.min_x)
        kernels = np.array([np.exp(-0.5 * (t-i)**2/s**2) for i in t])
        curve = self.gaussian_smoothing(labels, t, kernels)
        return np.stack([t, curve], axis=1)

    @staticmethod
    def get_intervals(df):
        result = portion.Interval()
        for i in np.stack([df.start, df.end], axis=1):
            result = result.union(portion.closed(*i))
        return result

    def intervals(self, stimulus=False, installation=None):
        df = self.adult_df if stimulus else self.get_df('bébé')
        result = self.get_intervals(df)
        if installation is not None:
            result = result.intersection(
                self.get_intervals(self.get_installation_df(installation)))
        return result

    def metrics(self, installation=None):
        resp = utils.length(
            self.intervals(stimulus=False, installation=installation))
        stim = utils.length(
            self.intervals(stimulus=True, installation=installation))
        instal = utils.length(
            self.get_intervals(self.get_installation_df(installation)))
        return (
            resp / instal if instal > 0.0 else np.nan,
            stim / instal if instal else np.nan)
