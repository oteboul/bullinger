import itertools
import matplotlib.pyplot as plt
from matplotlib import ticker

from bullinger import plot
from bullinger import visualizer


def readable_ax(ax, title, xlabel, ylabel):
    ax.legend(fontsize=18)
    for axis in (ax.xaxis, ax.yaxis):
        for tick in axis.get_major_ticks():
            tick.label.set_fontsize(18) 
            tick.label.set_rotation(0)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    if title is not None:
        ax.set_title(title, fontsize=24)


def plot_many_chronograms(c, num_rows=4, num_cols=4):
    video_ids = c.df.video_id.unique().tolist()
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 11, num_rows * 7))
    viz = visualizer.AnnotationsVisualizer(tags=c.tags)
    for i, j in itertools.product(range(num_rows), range(num_cols)):
        viz.chronogram(c, video_ids[i * num_cols + j], legend=j==num_cols-1, ax=axes[i, j])
    fig.tight_layout()


def plot_totals(agg, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(9, 4))
        ax = fig.gca()
        
    table = agg.totals().reset_index().pivot_table('total', ['semester'], ['group']) / 60    
    table.plot.bar(rot=0, ax=ax)
    readable_ax(ax, 'Temps total de video', 'Semestre', 'Minutes')


def plot_stimulations(agg):
    df = agg.stimulations(per_tag=False, relative=True)
    
    _, axes = plt.subplots(1, 2, figsize=(16, 4))
    template = 'Temps {} de stimulation'
    titles = [template.format(x) for x in ['absolu', 'relatif']]
    ylabels = ['Minutes', 'Temps relatif']
    for i, col in enumerate(['duration', 'relative']):
        ax = axes[i]
        table = df.pivot_table(col, ['semester'], ['group']) / (60 if i == 0 else 1)
        table.plot.bar(rot=0, ax=ax)
        readable_ax(ax, titles[i], 'Semestre', ylabels[i])


def plot_per(df,
             categories='support',
             rows=['support_time', 'proba'],
             cat_label='Nombre d\'appuis',
             labels=['Minutes', 'Probabilité de réponse'],
             flip=False,
            ):
    num_rows, num_cols = len(rows), 2
    if flip:
        num_rows, num_cols = num_cols, num_rows
    height, width = num_rows * 5, num_cols * 9
    _, axes = plt.subplots(
        num_rows, num_cols, figsize=(width, height), sharey=True)

    titles = ['Premier semestre', 'Deuxième semestre']    
    for i in range(2):
        for j, pivot in enumerate(rows):
            row, col = (i, j) if flip else (j, i)
            ax = axes[row, col] if len(rows) > 1 else axes[i]
            curr_df = df[(df.semester == i + 1)]
            table = curr_df.pivot_table(pivot, categories, ['group'])
            table.plot.bar(ax=ax)
            readable_ax(ax, titles[i], cat_label, labels[j])
    plt.tight_layout()


def plot_invisible(agg, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(9, 4))
        ax = fig.gca()
    df = agg.invisible
    df.pivot_table('total', ['semester'], ['group']).plot.bar(rot=0, ax=ax)
    readable_ax(ax, 'Fraction Invisible', 'Semestre', 'Temps relatif')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=1))
    return df


def plot_response(agg, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(9, 4))
        ax = fig.gca()
    df = agg.responds()
    df.pivot_table('relative', 'semester', ['group']).plot.bar(ax=ax)
    readable_ax(ax, 'Le bébé répond', 'Semestre', 'Temps relatif')
    return df

    