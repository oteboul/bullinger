import collections
import glob
import logging
import os.path
import numpy as np
import pandas as pd

import bullinger.annotations


class Aggregator:
    """Aggregate statistics over a cohort."""
    GROUPBY = ['group', 'semester']

    def __init__(self, cohort):
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

    def supports(self, relative=True):
        df = self.context(visible=True)
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

    @property
    def responds_with_support(self):
        groupby = self.GROUPBY + ['support']

        df = self.context(visible=True)
        df = df[(df.tag == 'rep')].groupby(groupby).agg({'duration': 'sum'})
        df = df.join(self.supports(relative=True), on=groupby)
        df = df.rename(columns={'duration': 'response'})
        df['proba'] = df.response / df.support_time
        df['minutes'] = df.response / 60
        return df.reset_index('semester')
    
    @property
    def invisible(self):
        return 1.0 - self.totals(visible=True) / self.totals(visible=False)


