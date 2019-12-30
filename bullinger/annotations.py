import logging
import numpy as np
import pandas as pd


class VideoAnnotations(object):
    """The annotations of a single video, annotated by ELAN."""

    COLUMNS = ['actor', 'video_id', 'start', 'end', 'duration', 'tag']

    def __init__(self, filename):
        self.filename = filename
        self.df = pd.read_csv(filename, sep='\t', header=None)
        # In case there are no annotations, they are NaNs
        if np.any(pd.isnull(self.df[self.df.columns[-1]])):
            logging.error('Some annotations are missing!')
            return

        self.df.columns = self.COLUMNS
        for col in ['actor', 'tag']:
            self.df[col] = self.df[col].apply(lambda x: x.strip().lower())

        self.min_x = np.min(self.df.start)
        self.max_x = np.max(self.df.end)

    def get_df(self, labels):
        return self.df.query(
            ' | '.join([f'actor=="{v}"' for v in labels.split(',')]))

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
