from absl import logging
import collections
from concurrent import futures
import glob
import os.path
import re

import pandas as pd
import pathlib
import portion
import numpy as np

from bullinger import annotations
from bullinger import utils
from bullinger import video


class Cohort(annotations.Annotations):
    """The whole annotation data for all the cohort."""

    def __init__(self, folder: str, num_workers: int = 20):
        self.folder = folder
        for suffix in ['**/*.txt', '**.txt']:
            self.filenames = glob.glob(os.path.join(self.folder, suffix))
            if self.filenames:
                break
        self._num_workers = min(num_workers, len(self.filenames))
        self._groups = self._may_load_groups()
        self._videos = self._read_annotations()
        self.df = pd.concat([v.df for v in self._videos]).reset_index()
        super().__init__(self.df)

    def __len__(self):
        return len(self._videos)

    def __iter__(self):
        yield from self._videos

    @property
    def num_annotations(self):
        return sum([v.num_annotations for v in self])

    def _may_load_groups(self):
        groups = None
        for folder in [self.folder, str(pathlib.Path(__file__).parents[1])]:
            candidates = glob.glob(os.path.join(folder, '**/*.csv'))
            if candidates:
                df = pd.read_csv(candidates[0])
                df.columns = [i for i in range(len(df.columns))]
                df = df[[0, 1]]
                df[0] = df[0].apply(utils.format_name)
                groups = dict(df.groupby([0, 1]).agg('count').reset_index().values)
        return groups

    def _read_annotations(self):
        def read_one(filename):
            return video.Video(filename=filename, groups=self._groups)

        with futures.ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            videos = executor.map(read_one, self.filenames)
        return list(videos)

    @property
    def summary(self):
        agg = {'video_id': 'nunique', 'baby': 'nunique', 'duration': np.sum}
        return self.context_df.groupby(['semester', 'group']).agg(agg)
