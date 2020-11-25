from absl import logging
import collections
from concurrent import futures
import glob
import os.path
import re
from typing import Dict, Sequence

import pandas as pd
import pathlib
import portion
import numpy as np

from bullinger import annotations
from bullinger import utils
from bullinger import video


class Cohort(annotations.Annotations):
    """The whole annotation data for all the cohort."""

    GROUPBY = ['semester', 'group']
    AGG_FUNCS = ['mean', 'sem']

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
        self.df = self.df.drop(columns=['index'])
        super().__init__(self.df)

    def __len__(self) -> int:
        return len(self._videos)

    def __iter__(self):
        yield from self._videos

    def __getitem__(self, i: int):
        return self._videos[i]

    @property
    def tags(self) -> Sequence[str]:
        return sorted(self.df.tag.unique())

    @property
    def interaction_tags(self) -> Sequence[str]:
        return sorted(self.actors_df.tag.unique())

    @property
    def num_annotations(self) -> int:
        return sum([v.num_annotations for v in self])

    def _may_load_groups(self) -> Dict[str, str]:
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

    def _read_annotations(self) -> Sequence[video.Video]:
        def read_one(filename):
            return video.Video(filename=filename, groups=self._groups)

        with futures.ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            videos = executor.map(read_one, self.filenames)
        return list(videos)

    @property
    def summary(self) -> pd.DataFrame:
        agg = {'video_id': 'nunique', 'baby': 'nunique', 'duration': np.sum}
        return self.context_df.groupby(['semester', 'group']).agg(agg)

    def aggregate(self, video_fn) -> pd.DataFrame:
        """Aggregates metrics per video into a single big DataFrame.
        
        Args:
          video_fn: A function that takes as input a video.Video and returns a
            pd.Series.

        Returns:
          A pd.DataFrame grouped by semester and group and which values are
          aggregated using the mean and the standard error of the mean.
        """

        def fn(v):
            result = video_fn(v)
            for col in self.GROUPBY:
                if col not in result.index and col in v.df.columns:
                    result[col] = v.df[col].iloc[0]
            return pd.DataFrame(result)

        df = (pd.concat([fn(v) for v in self], axis=1)
                .transpose()
                .reset_index()
                .drop(columns=['index']))
        for col in df.columns:
            if col not in self.GROUPBY:
                df[col] = df[col].astype(np.float)

        return df.groupby(self.GROUPBY).agg(self.AGG_FUNCS)