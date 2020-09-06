import datetime
import collections
import portion
import pandas as pd


def to_intervals(df):
    """Returns a dictionary which keys are tags and values its intervals."""
    result = collections.defaultdict(portion.Interval)     
    for idx, row in df.iterrows():
        result[row['tag']] = result[row['tag']] | portion.closed(row['start'], row['end'])
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
    if name[-1].upper() == name[-1]:
        return name[:-1].title() + name[-1]
    return name.title()