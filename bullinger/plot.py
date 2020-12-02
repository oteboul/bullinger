import functools
import itertools
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd


def readable_ax(ax,
                title: Optional[str] = None,
                xlabel: Optional[str] = None,
                ylabel: Optional[str] = None,
                rotation: Optional[int] = 0):
    ax.legend(fontsize=18)
    for axis in (ax.xaxis, ax.yaxis):
        for tick in axis.get_major_ticks():
            tick.label.set_fontsize(18) 
            if rotation is not None:
                tick.label.set_rotation(rotation)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=18)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=18)
    if title is not None:
        ax.set_title(title, fontsize=24)


def plot_agg(df: pd.DataFrame,
             metric: Optional[str] = None,
             ax=None,
             percentage: Optional[bool] = None,
             title: Optional[str] = None,
             **kwargs):
    """Plots the result of a groupby aggregation.
    
    If no metric is given plots all the columns, with one plot per semester.
    """
    df = pd.pivot_table(
        df, values=metric, index=df.index.names[0], columns=[df.index.names[1]])
    aggfuncs = df.columns.get_level_values(0).unique()
    mu = df.loc[:, df.columns.get_level_values(0)==aggfuncs[0]]
    err = df.loc[:, df.columns.get_level_values(0)==aggfuncs[1]]
    mu.columns = mu.columns.get_level_values(1)
    err.columns = err.columns.get_level_values(1)

    if (mu.index.dtype == float and
       (mu.index.astype(int).values - mu.index.values).sum() == 0.0):
        mu.index = mu.index.astype(int)
        err.index = err.index.astype(int)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 5))
    if 'ec' not in kwargs:
        kwargs['ec'] = 'k'
    mu.plot.bar(yerr=err, ax=ax, **kwargs)

    if np.all((mu <= 1.0) & (mu >= 0.0), axis=None) and percentage != False:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=None))
    title = metric if title is None else title
    readable_ax(ax, title=title, xlabel='semester', ylabel=metric, rotation=None)
    return ax


def per_semester(df: pd.DataFrame, axes=None, vertical=False, **kwargs):
    semesters = df.index.get_level_values(0).unique()
    if axes is None:
        plot_kw = dict(sharex=True) if vertical else dict(sharey=True)
        n_rows, n_cols = 1, semesters.size
        n_rows, n_cols = (n_cols, n_rows) if vertical else (n_rows, n_cols)
        _, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), **plot_kw)

    if 'ec' not in kwargs:
        kwargs['ec'] = 'k'

    cols = df.columns.get_level_values(1)
    aggfuncs = cols.unique()
    for i, s in enumerate(semesters):
        ax = axes[i]
        mu, err = (df.loc[s, cols==fn].transpose() for fn in aggfuncs)
        mu.index = mu.index.get_level_values(0)
        err.index = err.index.get_level_values(0)
        if (mu.index.dtype == float and
            (mu.index.astype(int).values - mu.index.values).sum() == 0.0):
            mu.index = mu.index.astype(int)
            err.index = err.index.astype(int)
        mu = mu.dropna()
        err = err.dropna()
        mu.plot.bar(ax=ax, yerr=err, **kwargs)
        readable_ax(ax, xlabel=f'Semestre {s:.0f}', rotation=None)
        if np.all((mu <= 1.0) & (mu >= 0), axis=None):
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
    return axes
    