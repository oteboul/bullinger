import collections
import glob
import logging
import os.path
import numpy as np
import pandas as pd

import bullinger.annotations


class Aggregator(object):
    """Aggregate statistics over several individuals."""

    def __init__(self, folder):
        self.folder = folder
        # Who is autistic
        candidates = glob.glob(os.path.join(self.folder, '*.csv'))
        if candidates:
            df = pd.read_csv(candidates[0])
            self.autists = set(df[df.group == 'AD'].baby.unique())
        else:
            self.autists = set()

        self.filenames = glob.glob(
            os.path.join(self.folder, '**/*.txt'), recursive=True)
        self.per_baby = collections.defaultdict(list)
        for filename in self.filenames:
            if os.path.basename(filename).startswith('__'):
                continue

            baby = os.path.basename(os.path.dirname(filename))
            try:
                ann = bullinger.annotations.VideoAnnotations(
                    filename, self.is_autistic(baby))
            except Exception as e:
                logging.error(e)
                continue
            self.per_baby[baby].append(ann)

        self.tags = set()
        for ll in self.per_baby.values():
            for x in ll:
                try:
                    self.tags.update(x.df.tag.unique())
                except Exception as e:
                    logging.error("{} df has no tag column".format(x.filename))
        self.tags = list(self.tags)

    def is_autistic(self, baby):
        return baby in self.autists

    @property
    def tds(self):
        return set(self.per_baby.keys()) - self.autists

    def average_stimulus(self, semester=2, relative=True, autists=None):
        result = pd.Series()
        total = 0.0
        for baby, vas in self.per_baby.items():
            if autists is not None and autists != self.is_autistic(baby):
                continue

            for va in vas:
                if not va.ill_formed and va.semester == semester:
                    total += 1.0
                    p = va.stimulus_distribution
                    result = result.add(p, fill_value=0.0)

        result /= total
        if relative:
            result /= np.sum(result)
        return result, total

    @property
    def metrics_df(self):
        result = []
        babies = []
        for baby, vas in self.per_baby.items():
            for va in vas:
                if va.ill_formed:
                    continue

                babies.append(baby)
                result.append(
                    (va.semester, self.is_autistic(baby)) +
                    va.metrics(None) +
                    va.metrics(True) +
                    va.metrics(False)
                )
        df = pd.DataFrame(np.array(result))
        df.columns = [
            'semester', 'ad',
            'resp_all', 'stimu_all', 'instal_all', 'score_all',
            'resp_avec', 'stimu_avec', 'instal_avec', 'score_avec',
            'resp_sans', 'stimu_sans', 'instal_sans', 'score_sans'
        ]
        df = df.astype({
            'semester': int, 'ad': bool,
        })
        df['baby'] = babies
        return df

    def responses(self, median=False, as_index=False):
        df = self.metrics_df

        def clean_median(x):
            return np.nanmedian(x[x < np.inf])

        def clean_mean(x):
            x = x[x < np.inf]
            return np.nanmean(x)

        agg_fn = clean_median if median else clean_mean
        return df.groupby(['ad', 'semester'], as_index=as_index).agg(agg_fn)
