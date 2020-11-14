from typing import Iterable
import datetime
import collections
import portion
import numpy as np
import pandas as pd


def to_intervals(df):
    """Returns a dictionary which keys are tags and values its intervals."""
    result = collections.defaultdict(portion.Interval)     
    for idx, row in df.iterrows():
        result[row['tag']] = result[row['tag']] | portion.closed(row['start'], row['end'])
    return result

def merge_intervals(intervs: Iterable[portion.Interval]) -> portion.Interval:
    result = portion.Interval()
    for i in intervs:
        result = result | i
    return result


def from_dataframe(df):
    result = portion.Interval()
    for i in np.stack([df.start, df.end], axis=1):
        result = result.union(portion.closed(*i))
    return result


def length(interv):
    """Length of an interval."""
    if interv.empty:
        return 0.0
    return sum([x.upper - x.lower for x in interv])


def parse_duration(duration: str) -> float:
    if isinstance(duration, float):
        return duration

    ref = '00:00:00.000'
    time_format = '%H:%M:%S.%f'
    d1 = datetime.datetime.strptime(duration, time_format)
    d2 = datetime.datetime.strptime(ref, time_format)
    return (d1 - d2).total_seconds()


def format_name(name: str) -> str:
    name = name.strip()
    parts = []
    for part in name.split():
        if part[-1].upper() == part[-1]:
            parts.append(part[:-1].title() + part[-1])
        else:
            parts.append(part.title())
    return ' '.join(parts)