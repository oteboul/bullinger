from typing import Dict, Iterable

import collections
import portion
import numpy as np
import pandas as pd

class Interval(portion.Interval):

    @property
    def length(self):
        if self.empty:
            return 0.0
        return np.sum([x.upper - x.lower for x in self])
    
    def expand_right(self, offset):
        parts = [portion.closed(i.lower, i.upper + offset) for i in self]
        return self.__class__(*parts)

    def union(self, other):
        return self.__class__(super().union(other))

    def intersection(self, other):
        return self.__class__(super().intersection(other))

    @staticmethod
    def from_dataframe(df: pd.DataFrame):
        result = Interval()
        for i in np.stack([df.start, df.end], axis=1):
            result = result.union(portion.closed(*i))
        return result


def tags_from_dataframe(df: pd.DataFrame,
                        col: str = 'tag') -> Dict[str, Interval]:
    """Returns a dictionary which keys are tags and values its intervals."""
    result = collections.defaultdict(Interval)     
    for idx, row in df.iterrows():
        result[row[col]] = result[row[col]] | portion.closed(row['start'], row['end'])
    return result


def breaks(d1: Dict[str, Interval],
           d2: Dict[str, Interval]) -> Dict[str, Interval]:
    """Breaks d1 partition intersecting with d2."""
    result = d1
    for t2, i2 in d2.items():
        res = collections.defaultdict(Interval)
        for t1, i1 in result.items():
            intervs = [(','.join([t1, t2]), i1.intersection(i2)), (t1, i1 - i2)]
            for k, i in intervs:
                if not i.empty:
                    res[k] = res[k].union(i)
        result = res
    return result

def breaks_per_tag(df1: pd.DataFrame,
                   df2: pd.DataFrame,
                   col: str = 'tag') -> Dict[str, Interval]:
    d1 = tags_from_dataframe(df1, col=col)
    d2 = tags_from_dataframe(df2, col=col)

    result = pd.DataFrame()
    for k, vs in breaks(d1, d2).items():
        for i in vs:
            if not i.empty and i.upper > i.lower:
                result = result.append({
                    'start': i.lower,
                    'end': i.upper,
                    'duration': i.upper - i.lower,
                    col: k,
                }, ignore_index=True)
    return result


def filter_by(df: pd.DataFrame, interv: Interval):

    def _intersect(row):
        inter = interv.intersection(portion.closed(row.start, row.end))
        if inter.empty:
            return 0, 0
        else:
            return inter.lower, inter.upper

    df['bounds'] = df.apply(_intersect, axis=1)
    df['start'] = df.bounds.apply(lambda x: x[0])
    df['end'] = df.bounds.apply(lambda x: x[1])
    df = df.drop(columns=['bounds'])
    df = df[df.end > df.start]
    return df
