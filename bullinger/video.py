import collections
import os
from typing import Dict, Optional

import pandas as pd
import pathlib
import portion
import numpy as np

from bullinger import intervals


class Video:
   
    def __init__(self,
                 filename: Optional[str] = None,
                 groups: Optional[Dict[str, str]] = None,
                 df: Optional[pd.DataFrame] = None,
                 support: str = 'appui',
                 without: str = 'sans',
                 context_col: str = 'context',
                 fill_no_support=True):
        if filename is None and df is None:
            raise ValueError("One of `filename` or `df` should be specified")

        self.filename = filename
        self.support = support
        self.without = without
        self.context_col = context_col
        if filename is not None:
            self.name = pathlib.Path(filename).parent.name.strip()
            self.df = self._reads_df()
        else:
            self.df = df
            self.name = self.df.name[0]
        if groups is not None:
            self.df['group'] = groups.get(self.name, '?')

        self.num_annotations = self.df.shape[0]
        if self.context_col not in self.df.columns:
            self.df[self.context_col] = np.nan
        if fill_no_support:
            self._adds_no_support()
        self._add_context()

    @property
    def vid(self):
        if self.df.shape[1] > 0:
            return self.df.video_id.iloc[0]
        return ''

    @property
    def start(self):
        return self.df.start.min()

    @property
    def end(self):
        return self.df.end.max()

    @property
    def duration(self):
        return intervals.Interval.from_dataframe(self.context_df).length

    @property
    def shape(self):
        return self.df.shape

    def _reads_df(self):
        df = pd.read_csv(self.filename, sep='\t', header=None)
        if df.shape[1]  > 6:
            df = df.drop(columns=[2, 4, 6])
        df.columns = ['actor', 'video_id', 'start', 'end', 'duration', 'tag']
        df['baby'] = df.video_id.apply(lambda x: x.split('_')[0].title())
        df['semester'] = df.video_id.str.contains(r'\(6-12\)').astype(int)+1
        df['name'] = self.name
        for col in ['actor', 'tag']:
            df[col] = df[col].apply(lambda x: x.strip().lower())
            # remove diacritics
            df[col] = df[col].str.normalize('NFKD').str.encode(
                'ascii', errors='ignore').str.decode('utf-8')
        return df

    @property
    def support_df(self):
        df = self.df
        return df[df.actor.str.startswith(self.support)]

    @property
    def context_df(self):
        df = self.df
        return df[df.actor.str.startswith(self.support) | (df.actor == 'contexte')]

    @property
    def actors_df(self) -> pd.DataFrame:
        df = self.df
        return df[df.actor.isin(['bebe', 'adulte', 'adulte bis'])]

    @property
    def visible(self) -> pd.DataFrame:
        return self.df[self.df.tag != 'inv']

    @property
    def stimulations(self) -> pd.DataFrame:
        df = self.df
        return df[df.actor.str.startswith('adult')]

    @property
    def responses(self) -> pd.DataFrame:
        df = self.df
        return df[df.tag.isin(['rep', 'init'])]
        
    def _adds_no_support(self):
        df = self.df
        full = portion.closed(self.start, self.end)
        context_tags = intervals.tags_from_dataframe(self.context_df)
        all_support = intervals.Interval(*context_tags.values())
        for i in full - all_support:
            if not i.empty:
                df = df.append({'start': i.lower,
                                'end': i.upper,
                                'actor': self.support,
                                'tag': self.without,
                                'duration': i.upper - i.lower
                                }, ignore_index=True)
        df = df.fillna(df.mode().iloc[0])
        self.df = df
        
    def _expand_with_context(self, actor='bebe', actor_df=None, tag=None):
        """Replaces the rows of the actor by the context expanded ones."""
        df = self.df        

        if actor_df is None:
            actor_df = df[(df.actor == actor) & (pd.isnull(df.context))]
        ctx_df = intervals.breaks_per_tag(actor_df, self.context_df, col='tag')

        if ctx_df.size:
            ctx_df['context'] = ctx_df.tag.apply(
                lambda x: ','.join(sorted(x.split(',')[1:])))
            ctx_df['tag'] = ctx_df.tag.apply(lambda x: x.split(',')[0]) if tag is None else tag
            ctx_df['num_supports'] = ctx_df.context.apply(
                lambda x: len(x.split(',')) - int(x==self.without))
            ctx_df['actor'] = actor

            df = pd.concat([df, ctx_df]).reset_index().drop(columns=['index'])
            df = df.drop(df[(df.actor==actor) & (pd.isnull(df.context))].index)
        self.df = df

    def _add_context(self):
        """Adds the context to all the real actors."""
        df = self.df
        actors = set(['bebe', 'adulte', 'adulte bis'])
        for actor in actors.intersection(self.df.actor.unique()):
            self._expand_with_context(actor, tag=None)
        self._expand_with_context(actor='contexte', actor_df=self.support_df, tag='-')
        df = self.df
        df = df.drop_duplicates()
        df = df.fillna(df.mode().iloc[0])
        df = df.drop(self.support_df.index)

        inv_idx = df.context.str.contains('inv')
        df.loc[inv_idx, ['context', 'tag', 'num_supports']] = ['inv', 'inv', -1]
        df.loc[df.tag == '-', 'tag'] = 'support'
        self.df = df.sort_values(by='start')

    @property
    def summary(self) -> pd.Series:
        result = {'duration': self.duration}
        stimul = intervals.Interval.from_dataframe(self.stimulations).length
        result['stimulation'] = stimul / self.duration
        result['response'] = self.responses.duration.sum() / self.duration
        return pd.Series(result)

    def responses_per(self, groupby='num_supports') -> pd.DataFrame:
        df1 = (self.visible
                .groupby(groupby)
                .agg(lambda x: intervals.Interval.from_dataframe(x).length)[['duration']]
                .rename(columns={'duration': 'total'}))
        df2 = self.responses.groupby(groupby).agg({'duration': sum}).rename(columns={'duration': 'responses'})
        result = df1.join(df2).reset_index().fillna(0.0)
        result['relative'] = result.responses / result.total
        return result

    def trim_no_stimulations(self, margin: int = 30) -> Video:
        """Remove the context where there is no stimulation."""
        i = intervals.from_dataframe(self.stimulations).expand_right(margin)
        return Video(df=intervals.filter_by(self.df, i), fill_no_support=False)

