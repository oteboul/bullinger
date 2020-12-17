"""The different analysis of a single video."""

import functools
import numpy as np
import pandas as pd

from bullinger import cohort
from bullinger import intervals
from bullinger import video


def to_series(func):
    """A decorator to make sure the return type is pd.Series."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, pd.Series):
            return result
        elif isinstance(result, dict):
            return pd.Series(result)
        else:
            return pd.Series({'result': result})
    return wrapper


def responses_per_supports(v: video.Video) -> pd.Series:
    df = v.responses_per(groupby='num_supports')
    df = pd.pivot_table(df, 'relative', index='num_supports').transpose()
    return df.iloc[0]


def support_duration(v: video.Video) -> pd.Series:    
    df = v.visible
    df = df.groupby('num_supports').agg(
        lambda x: intervals.from_dataframe(x).length)[['duration']]
    df['duration'] /= df.duration.sum()
    df = pd.pivot_table(df, 'duration', index='num_supports').transpose()
    return df.iloc[0]


@to_series
def invisible_time(v: video.Video):
    return {'invisible': v.df[v.df.tag == 'inv'].duration.sum() / v.duration}


@to_series
def response_rate(v: video.Video, tolerance: float = 3.0):
    """How many stimulations are responded ?."""
    if not v.stimulations.shape[0]:
        return np.nan
    
    resp = intervals.from_dataframe(v.responses)
    stim = intervals.from_dataframe(v.stimulations)
    seqs = v.sequences(tolerance)
    count, total = 0, 0
    for seq in seqs:
        if stim.intersection(seq).empty:
            continue

        count += int(not seq.intersection(resp).empty)
        total += 1
    return count / total


@to_series
def fair_relative_duration(v: video.Video, offset: int = 3.0):
    if not v.stimulations.shape[0]:
        return np.nan
    
    v = v.trim_no_stimulations(offset)
    return v.responses.duration.sum() / v.duration
    

@to_series
def starts(v: video.Video, tolerance: int = 3):
    """When *there is* an interaction, how often does the baby starts it?"""
    resp = intervals.from_dataframe(v.responses)
    stim = intervals.from_dataframe(v.stimulations)
    seqs = v.sequences(tolerance)
    count, total = 0, 0
    for seq in seqs:
        curr_stim = stim.intersection(seq)
        curr_resp = resp.intersection(seq)
        if curr_stim.empty or curr_resp.empty:
            continue
            
        count += int(curr_stim.lower > curr_resp.lower)
        total += 1
    return count / total if total else np.nan


@to_series
def turn_taking(v: video.Video, tolerance: int = 5):
    """When there is an interaction, how many back and forth?"""
    resp = intervals.from_dataframe(v.responses)
    stim = intervals.from_dataframe(v.stimulations)
    seqs = v.sequences(tolerance)
    count, total = 0, 0
    for seq in seqs:
        curr_stim = stim.intersection(seq)
        curr_resp = resp.intersection(seq)
        if curr_stim.empty or curr_resp.empty:
            continue
            
        count += len(curr_resp)
        total += 1
    return count / total if total else np.nan


@to_series
def unanswered_init(v: video.Video, tolerance: int = 5):
    """The baby starts, but is not answered back."""
    resp = intervals.from_dataframe(v.responses)
    stim = intervals.from_dataframe(v.stimulations)
    seqs = v.sequences(tolerance)
    count, total = 0, 0
    for seq in seqs:
        curr_stim = stim.intersection(seq)
        curr_resp = resp.intersection(seq)
        if curr_resp.empty:
            continue
            
        count += int(curr_stim.empty)
        total += 1
    return count / total if total else np.nan


@to_series
def average_response_duration(v: video.Video):
    """The baby starts, but is not answered back."""
    return v.responses.duration.mean()