from absl import logging
import collections
import glob
import os.path
import re

import pandas as pd
import pathlib
import portion
import numpy as np

from bullinger import annotations, utils

    

class AnnotatedCohort:
    """The whole annotation data for all the cohort."""
    SUPPORT = 'appui'
    CONTEXT = 'contexte'
    INVISIBLE = 'inv'

    def __init__(self, folder, filter_out=None):
        self.folder = folder
        self.has_observer = False
        for suffix in ['**/*.txt', '**.txt']:
            self.filenames = glob.glob(os.path.join(self.folder, suffix))
            if self.filenames:
                self.has_observer = True
                break

        self.groups = None
        for folder in [self.folder, str(pathlib.Path(__file__).parents[1])]:
            candidates = glob.glob(os.path.join(folder, '**/*.csv'))
            if candidates:
                df = pd.read_csv(candidates[0])
                df.columns = [i for i in range(len(df.columns))]
                df = df[[0, 1]]
                df[0] = df[0].apply(utils.format_name)
                self.groups = dict(
                    df.groupby([0, 1]).agg('count').reset_index().values)

        self.df = None
        self.full_df = None
        for filename in self.filenames:
            try:
                va = annotations.VideoAnnotations(filename)
            except Exception as e:
                logging.warning(f'Cannot open {filename}: {e}')
                continue

            if va.ill_formed:
                continue

            df = va.with_context
            context_df = va.to_context(with_baby=False)
            observer = filename.split('/')[1]
            baby = pathlib.Path(filename).parent.name.strip()
            df['baby'] = baby
            if self.groups is not None:
                df['group'] = self.groups.get(baby, '?')
                va.df['group'] = self.groups.get(baby, '?')
            if self.has_observer:
                df['observer'] = observer
                va.df['observer'] = observer
            df['semester'] = va.semester
            if self.df is None:
                self.df = df
                self.full_df = va.df
                self.context_df = context_df
            else:
                self.df = pd.concat([self.df, df])
                self.full_df = pd.concat([self.full_df, va.df])
                self.context_df = pd.concat([self.context_df, context_df])
        self.df = self.df.reset_index()
        self.full_df = self.full_df.reset_index()
        self.df.support = self.df.support.fillna(0).astype(int)
        # Init is assimilated with a response for now.
        self.df.replace('init', 'rep', inplace=True)

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