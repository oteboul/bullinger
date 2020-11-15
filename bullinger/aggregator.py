import collections
import glob
import logging
import os.path
import numpy as np
import pandas as pd
import scipy.stats

from bullinger import annotations
from bullinger import utils


class Aggregator(annotations.Annotations):
    """Aggregate statistics over a cohort."""
    GROUPBY = ['group', 'semester']

    def __init__(self, cohort):
        super().__init__(cohort.df)
        self.cohort = cohort
        self.df = self.cohort.df

    def context(self, visible=False) -> pd.DataFrame:
        df = self.df[(~self.df.context.isnull())]
        if visible:
            df = df[df.context != 'inv']
        return df

    def totals(self, visible=False) -> pd.DataFrame:
        """Total time per group and semester, optionnally discarding inv."""
        df = self.context(visible=visible)
        df = df.groupby(self.GROUPBY).agg({'duration': 'sum'})
        return df.rename(columns={'duration': 'total'})

    def stimulations(self, per_tag=False, relative=True):
        """Computes stimulations (absolute or not) per group and semester."""
        df = self.df[self.df.actor.str.startswith('adult')]
        add = ['tag'] if per_tag else []
        df = df.groupby(self.GROUPBY + add).agg({'duration': 'sum'})

        if not relative:
            return df

        df = df.reset_index()
        df = df.join(self.totals(), self.GROUPBY)
        df['relative'] = df.duration / df.total
        if per_tag:
            df2 = self.stimulations(per_tag=False, relative=False)
            df2 = df2.rename(columns={'duration': 'stimulation'})
            df = df.join(df2, self.GROUPBY)
            df['in_stimulation'] = df.duration / df.stimulation
        return df        

    def supports(self, relative=True, min_support=0, max_support=3):
        df = self.context(visible=True)
        df = df[(df.support <= max_support) & (df.support >= min_support)]
        df = df.groupby(self.GROUPBY + ['support']).agg({'duration': 'sum'})
        df = df.rename(columns={'duration': 'support_time'})

        if relative:
            total_df = self.totals(visible=True)
            df = df.join(total_df, on=self.GROUPBY)
            df['relative'] = df.support_time / df.total
            df['minutes'] = df.support_time / 60
        return df

    @property
    def responds(self):
        df = self.context(visible=True)
        df['response'] = df.duration * (df.tag == 'rep')
        df = df.groupby(self.GROUPBY).agg(
            {'duration': 'sum', 'response': 'sum'}).reset_index('semester')
        df['relative'] = df.response / df.duration
        return df

    def responds_with_support(self, min_support=0, max_support=5):
        groupby = self.GROUPBY + ['support']

        df = self.context(visible=True)
        df = df[(df.support <= max_support) & (df.support >= min_support)]
        df = df[(df.tag == 'rep')].groupby(groupby).agg({'duration': 'sum'})
        df = df.join(self.supports(relative=True), on=groupby)
        df = df.rename(columns={'duration': 'response'})
        df['proba'] = df.response / df.support_time
        df['minutes'] = df.response / 60
        return df.reset_index('semester')

    def responses_distribution(self, min_support=0, max_support=5):
        groupby = self.GROUPBY + ['support']
        df = self.context(visible=True)
        df = df[(df.support <= max_support) & (df.support >= min_support)]
        df = df[(df.tag == 'rep')]
        return df.groupby(groupby).agg({'duration': 'sum'})

    @property
    def invisible(self):
        return 1.0 - self.totals(visible=True) / self.totals(visible=False)

    def per_video_responses(self):
        groupby = self.GROUPBY + ['video_id']
        df = self.context(visible=True)
        df_tag_supp = (df
            .groupby(['support', 'tag'] + groupby)
            .agg({'duration': np.sum})
            .reset_index('tag'))
        df_supp = (df
            .groupby(['support'] + groupby)
            .agg({'duration': np.sum})
            .rename(columns={'duration': 'support_duration'}))
        result_df = df_supp.join(df_tag_supp).reset_index()
        result_df['ratio'] = result_df['duration'] / result_df['support_duration']
        result_df = result_df[result_df.tag == 'rep']
        return result_df

    def per_video(self):

        def _aux(df):
            result = {}
            for k in ['baby'] +  self.GROUPBY:
                result[k] = df[k].unique()[0]

            df = df[(~df.context.isnull())]
            result['duration'] = df.duration.sum()
            
            stimul = utils.get_intervals(df[df.actor.str.startswith('adult')])
            result['stimulation'] = utils.length(stimul) / result['duration']

            total_responses = df[df.tag.isin(['rep', 'init'])].duration.sum()
            result['response'] = total_responses / result['duration']
        
            return pd.Series(result)

        return self.df.groupby('video_id').apply(_aux).reset_index()

    def statistical_test(self,
                         semester: int = 2,
                         support: int = 3,
                         min_duration: float = 0.0,
                         threshold: float = 0.05,
                         test_fn=scipy.stats.ttest_ind) -> float:
        """Test whether two sets of samples are statistically different.

        H0: the distributions of all samples are equal.
        H1: the distributions of one or more samples are not equal.

        Args:
          semester: the semester to look at.
          support: the number of supports to look at.
          min_duration: the minimum duration of responses to be considered in a
            video.
          threshold: the threshold to accept H0 (above it).
          test_fn: test functions from scipy. This includes ttest_ind, kruskal,
            f_oneway (ANOVA), mannwhitneyu.


        Returns:
          A Tuple[bool, float], which first element is the result of the test
          (keep or reject equality hypothesis) and the p-value itself.
        """
        df = self.per_video_responses()
        df = df[(df.semester == semester) &
                (df.support == support) &
                (df.duration > min_duration)]
        samples = [df[(df.group == group)].ratio for group in ('AD', 'TD')]
        pvalue = test_fn(*samples).pvalue
        return pvalue > threshold, pvalue