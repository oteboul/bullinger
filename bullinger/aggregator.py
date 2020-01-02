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
        self.filenames = glob.glob(os.path.join(self.folder, '**/*.txt'))
        self.per_baby = collections.defaultdict(list)
        for filename in self.filenames:
            if os.path.basename(filename).startswith('__'):
                continue

            baby = os.path.basename(os.path.dirname(filename))
            try:
                ann = bullinger.annotations.VideoAnnotations(filename)
            except Exception as e:
                logging.error(e)
                continue
            self.per_baby[baby].append(ann)

        candidates = glob.glob(os.path.join(self.folder, '*csv'))
        if candidates:
            df = pd.read_csv(candidates[0])
            self.autists = set(df[df.group == 'AD'].baby.unique())

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
                    if relative:
                        p /= np.sum(p)
                    result = result.add(p, fill_value=0.0)

        result /= total
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
                    va.metrics(False))
        df = pd.DataFrame(np.array(result))
        df.columns = [
            'semester', 'ad',
            'r_all', 's_all', 'r_avec', 's_avec', 'r_sans', 's_sans'
            ]
        df = df.astype({
            'semester': int, 'ad': bool,
        })
        df['baby'] = babies
        return df
