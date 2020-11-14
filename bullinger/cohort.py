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

from bullinger import video, utils



class Cohort:
    """The whole annotation data for all the cohort."""
    SUPPORT = 'appui'
    CONTEXT = 'contexte'
    INVISIBLE = 'inv'

    def __init__(self, folder: str, num_workers: int = 20, filter_out=None):
        self.folder = folder
        for suffix in ['**/*.txt', '**.txt']:
            self.filenames = glob.glob(os.path.join(self.folder, suffix))
            if self.filenames:
                break
        self._num_workers = min(num_workers, len(self.filenames))
        self._groups = self._may_load_groups()
        self._videos = self._read_annotations()
        self.df = pd.concat([v.df for v in self._videos]).reset_index()

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
        return videos

    def extract(self, video_id: str, observer: str) -> pd.DataFrame:
        return self.df[(self.df['video_id'] == video_id) &
                       (self.df['observer'] == observer)]

    @property
    def actors(self):
        return self.df[(self.df.actor != self.CONTEXT) &
                       (~self.df.actor.str.startswith('rme'))]

    @property
    def tags(self):
        return set(self.actors.tag.unique()) - {None}

    @property
    def support(self) -> pd.DataFrame:
        return self.df[self.df['actor'] == self.CONTEXT]

    @property
    def interactions(self) -> pd.DataFrame:
        return self.df[self.df['actor'] != self.CONTEXT]

    @property
    def baby_context(self):
        return self.df[~self.df.context.isnull()]

    def get_total_time(self):
        df2 = self.df.groupby(['video_id', 'group', 'semester', 'baby']).agg({'start': 'min', 'end': 'max'}).reset_index()
        df2['duration'] = df2.end - df2.start
        total_df = df2.groupby(['group', 'semester']).agg({'duration': 'sum'})
        return total_df.rename(columns={'duration': 'total'})

    @property
    def stimulus_distribution(self):
        return self.adult_df.groupby('tag').duration.sum()

    @property
    def agreement_df(self):
        if not self.has_observer:
            return

        intervals = collections.defaultdict(list)
        for x in self.full_df.groupby(['video_id', 'actor', 'observer']):
            for tag, interv in utils.to_intervals(x[1]).items():
                intervals[(x[0][:2]) + (tag,)].append(interv)

        result = []
        for (vid, actor, tag), intervs in intervals.items():
            w = np.max([utils.length(x) for x in intervs])
            curr = {'video_id': vid, 'actor': actor, 'weight': w}
            if len(intervs) < 2:
                curr['agree'] = 0.0
            elif len(intervs) == 2:
                intersection = intervs[0].intersection(intervs[1])
                curr['agree'] = utils.length(intersection) / w
            result.append(curr)
        return pd.DataFrame(result)

    @property
    def agreement(self):
        if not self.has_observer:
            return 0.0

        df = self.agreement_df
        df['a'] = df.agree * df.weight
        return np.sum(df.a) / np.sum(df.weight)