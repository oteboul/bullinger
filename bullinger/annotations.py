import logging
import numpy as np
import pandas as pd
import intervals

class VideoAnnotations(object):
    """The annotations of a single video, annotated by ELAN."""

    COLUMNS = ['actor', 'video_id', 'start', 'end', 'duration', 'tag']

    def __init__(self, filename):
        self.filename = filename
        self.semester = 1 if '(0-6)' in filename else 2
        self.df = pd.read_csv(filename, sep='\t', header=None)
        self.ill_formed = False
        # In case there are no annotations, they are NaNs
        if np.any(pd.isnull(self.df[self.df.columns[-1]])):
            logging.error('Some annotations are missing!')
            self.ill_formed = True
            return

        if len(self.df.columns) > 6:
            self.df = self.df.drop(columns=[2, 4, 6])
        self.df.columns = self.COLUMNS
        for col in ['actor', 'tag']:
            self.df[col] = self.df[col].apply(lambda x: x.strip().lower())

        self.min_x = np.min(self.df.start)
        self.max_x = np.max(self.df.end)

    @property
    def stimulus_distribution(self):
        return self.get_adult_df().groupby('tag').duration.sum()

    def get_df(self, labels):
        return self.df.query(
            ' | '.join([f'actor=="{v}"' for v in labels.split(',')]))

    def get_installation_df(self, presence=None):
        df = self.get_df('contexte')
        if presence is not None:
            tag = 'avec' if presence else 'sans'
            df = df[df.tag == tag]
        return df

    def get_adult_df(self):
        return self.df[self.df.actor.str.contains('adulte')]

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
        result = intervals.Interval()
        for i in np.stack([df.start, df.end], axis=1):
            result = result.union(intervals.closed(*i))
        return result

    def interval_length(self, intervs):
        try:
            return sum([x.upper - x.lower for x in intervs])
        except Exception as e:
            return 0.0

    def intervals(self, stimulus=False, installation=None):
        df = self.get_adult_df() if stimulus else self.get_df('bébé')
        result = self.get_intervals(df)
        if installation is not None:
            result = result.intersection(
                self.get_intervals(self.get_installation_df(installation)))
        return result

    def metrics(self, installation=None):
        resp = self.interval_length(
            self.intervals(stimulus=False, installation=installation))
        stim = self.interval_length(
            self.intervals(stimulus=True, installation=installation))
        instal = self.interval_length(
            self.get_intervals(self.get_installation_df(installation)))
        return (
            resp / instal if instal > 0.0 else np.nan,
            stim / instal if instal else np.nan)
