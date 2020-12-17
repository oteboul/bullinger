import collections
import os
from typing import Dict, Optional, Sequence

import pandas as pd
import pathlib
import portion
import numpy as np

from bullinger import annotations
from bullinger import intervals
from bullinger import utils


class Video(annotations.Annotations):
   
    def __init__(self,
                 filename: Optional[str] = None,
                 df: Optional[pd.DataFrame] = None,
                 groups: Optional[Dict[str, str]] = None,
                 fill_no_support: bool = True,
                 process_context: bool = True):
        super().__init__(None)

        if filename is None and df is None:
            raise ValueError("One of `filename` or `df` should be specified")

        self.filename = filename
        if filename is not None:
            self.name = pathlib.Path(filename).parent.name.strip()
            self.original_df = self._reads_df()
            self.df = self.original_df
        else:
            self.df = df
            self.name = self.df.baby.iloc[0]
        if groups is not None:
            self.df['group'] = groups.get(self.name, '?')

        # Gets the number of annotations before splitting them according to the
        # context.
        self.num_annotations = self.df.shape[0]
        context_col = 'context'
        if context_col not in self.df.columns:
            self.df[context_col] = np.nan
        if fill_no_support:
            self._add_no_support()
        if process_context:
            self._add_context()
        self.df['duration'] = self.df['end'] - self.df['start']
        # Enforce that the invisble corresponds to the actor context.
        self.df.loc[self.df.tag == self._invisible, ['actor']] = self._context

    @property
    def vid(self):
        if self.df.shape[1] > 0:
            return self.df.video_id.iloc[0]
        return ''

    @property
    def group(self):
        if self.df.shape[1] > 0:
            return self.df.group.iloc[0]
        return ''

    @property
    def semester(self):
        if self.df.shape[1] > 0:
            return self.df.semester.iloc[0]
        return ''

    @property
    def start(self):
        return self.df.start.min()

    @property
    def end(self):
        return self.df.end.max()

    @property
    def duration(self):
        return intervals.from_dataframe(self.context_df).length

    @property
    def content(self) -> str:
        if self.filename is None:
            return ''
        with open(self.filename) as fp:
            return fp.read()

    @property
    def constants(self) -> pd.Series:
        """Returns a Series with all the constants: baby, group, name etc."""
        return pd.Series({
            'group': self.group,
            'semester': self.semester,
            'id': self.vid,
            'name': self.name,
            'duration': self.duration,
        })

    @property
    def normalized_video_name(self):
        result = os.path.basename(self.filename)[:-4]
        i = result.index('(')
        result = result[:i].replace('-', '_') + result[i:]
        if result.endswith(')'):
            result = result[:result.rfind('(')]
        return result
    
    def _reads_df(self):
        df = pd.read_csv(self.filename, sep='\t', header=None)
        if df.shape[1] > 6:
            df = df.drop(columns=[2, 4, 6])
        df.columns = ['actor', 'video_id', 'start', 'end', 'duration', 'tag']
        df['video_id'] = self.normalized_video_name
        df['semester'] = df.video_id.str.contains(r'\(6-12\)').astype(int)+1
        df['baby'] = self.name
        for col in ['actor', 'tag']:
            try:
                df[col] = df[col].apply(lambda x: x.strip().lower())
            except AttributeError as e:
                print(e, self.filename)
            # remove diacritics
            df[col] = df[col].str.normalize('NFKD').str.encode(
                'ascii', errors='ignore').str.decode('utf-8')
        cols = ['start', 'end', 'duration']
        for col in cols:
            if df[col].dtype != float:
                df.loc[:, col] = df[col].apply(utils.parse_duration)
        return df
        
    def _add_no_support(self):
        df = self.df
        full = portion.closed(self.start, self.end)
        context_tags = intervals.tags_from_dataframe(self.context_df)
        all_support = intervals.Interval(*context_tags.values())
        for i in full - all_support:
            if not i.empty:
                df = df.append({'start': i.lower,
                                'end': i.upper,
                                'actor': self._support,
                                'tag': self._without,
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
            ctx_df['tag'] = (ctx_df.tag.apply(lambda x: x.split(',')[0])
                             if tag is None else tag)
            ctx_df['num_supports'] = ctx_df.context.apply(
                lambda x: len(x.split(',')) - int(x == self._without))
            ctx_df['actor'] = actor
            df = pd.concat([df, ctx_df]).reset_index().drop(columns=['index'])
            df = df.drop(df[(df.actor==actor) & (pd.isnull(df.context))].index)
        self.df = df

    def _add_context(self):
        """Adds the context to all the real actors."""
        df = self.df
        actors = set(['bebe', 'bebe bis', 'adulte', 'adulte bis'])
        for actor in actors.intersection(self.df.actor.unique()):
            self._expand_with_context(actor, tag=None)
        self._expand_with_context(actor='contexte', actor_df=self.context_df, tag='-')
        df = self.df
        df = df.drop_duplicates()
        df = df.fillna(df.mode().iloc[0])
        df = df.drop(self.support_df.index)

        inv_idx = df.context.str.contains(self._invisible)
        df.loc[inv_idx, ['context', 'tag', 'num_supports']] = [self._invisible, self._invisible, -1]
        df.loc[df.tag == '-', 'tag'] = 'support'
        self.df = df.sort_values(by='start')

    @property
    def summary(self) -> pd.Series:
        result = {'duration': self.duration}
        stimul = intervals.from_dataframe(self.stimulations).length
        result['stimulation'] = stimul / self.duration
        result['response'] = self.responses.duration.sum() / self.duration
        return pd.Series(result)

    def responses_per(self, groupby='num_supports') -> pd.DataFrame:
        df1 = (self.visible
                .groupby(groupby)
                .agg(lambda x: intervals.from_dataframe(x).length)[['duration']]
                .rename(columns={'duration': 'total'}))
        df2 = self.responses.groupby(groupby).agg({'duration': sum}).rename(columns={'duration': 'responses'})
        result = df1.join(df2).reset_index().fillna(0.0)
        result['relative'] = result.responses / result.total
        return result

    def trim_no_stimulations(self, margin: int = 30):
        """Remove the context where there is no stimulation."""
        i = intervals.from_dataframe(self.stimulations).expand_right(margin)
        return Video(df=intervals.filter_by(self.df, i), fill_no_support=False)

    def sequences(self, tolerance: float = 3.0) -> Sequence[intervals.Interval]:
        """Extract the individual sequences of interactions."""
        if not self.stimulations.shape[0]:
            return []

        stim = intervals.from_dataframe(self.stimulations)
        resp = intervals.from_dataframe(self.responses)
        seqs = resp.expand_right(tolerance).union(stim.expand_right(tolerance))
        events = resp.union(stim)
        
        result = []
        for seq in seqs:
            curr = seq.intersection(events)
            result.append(intervals.closed(curr.lower,curr.upper))
        return result
            

