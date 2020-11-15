import datetime
import functools
import os.path
import logging
import numpy as np
import pandas as pd
import portion


class Annotations:
    """The annotations of a single video, annotated by ELAN."""

    def __init__(self, df):
        self.df = df
        self._invisible = 'inv'
        self._support = 'appui'
        self._context = 'contexte'
        self._adult = 'adulte'
        self._baby = 'bebe'
        self._without = 'sans'

    @property
    def shape(self):
        return self.df.shape

    @property
    def support_df(self):
        df = self.df
        return df[df.actor.str.startswith(self._support)]

    @property
    def context_df(self):
        df = self.df
        support = df.actor.str.startswith(self._support)
        return df[support | (df.actor == self._context)]

    @property
    def actors_df(self) -> pd.DataFrame:
        df = self.df
        adult = df.actor.str.startswith(self._adult)
        baby = df.actor == self._baby
        return df[adult | baby]

    @property
    def visible(self) -> pd.DataFrame:
        return self.df[self.df.tag != self._invisible]

    @property
    def stimulations(self) -> pd.DataFrame:
        df = self.df
        return df[df.actor.str.startswith(self._adult)]

    @property
    def responses(self) -> pd.DataFrame:
        df = self.df
        return df[df.actor == self._baby]
        