from pprint import pprint
from typing import Optional
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import click
from sklearn.preprocessing import MultiLabelBinarizer

@click.group()
def cli():
    pass

@cli.command()
@click.option('--predictions', '-p', type=click.Path(exists=True), required=True, multiple=True)
@click.option('--output', type=click.Path(exists=False), required=True)
@click.option('--norm-all', 'norm', is_flag=True, flag_value='all')
@click.option('--norm-pred', 'norm', is_flag=True, flag_value='pred')
@click.option('--norm-true', 'norm', is_flag=True, flag_value='true')
@click.option('--show', is_flag=True)
def heatmap(
    predictions: list[Path],
    output: Path,
    norm: Optional[str],
    show: bool,
):
    df = pd.concat([pd.read_json(pred) for pred in predictions])
    labels = [l for l in df.labels.explode().unique()]
    fig = plt.figure(figsize=(10,10))
    mat = confusion_matrix(df.labels.explode(), df.preds.explode(), labels=labels, normalize=norm)
    g = sns.heatmap(mat, xticklabels=labels, yticklabels=labels, square=True, annot=True, cbar=False)
    plt.savefig(output)
    if show:
        fig.show()
        input()

@cli.command()
@click.option('--predictions', '-p', type=click.Path(exists=True), required=True, multiple=True)
@click.option('--average', type=str, required=True)
def multilabel(
    predictions: list[Path],
    average: str
):
    df = pd.concat([pd.read_json(pred) for pred in predictions])
    classes = [l for l in df.labels.explode().str.split('-').explode().unique() if l != 'O']
    print('Labels: ', classes)
    mlb = MultiLabelBinarizer(classes=classes)
    labels = mlb.fit_transform(df.labels.explode().str.split('-'))
    preds = mlb.transform(df.preds.explode().str.split('-'))
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average=average)
    pprint(dict(precision=p * 100, recall = r * 100, f1=f * 100))

@cli.command()
@click.option('--predictions', '-p', type=click.Path(exists=True), required=True, multiple=True)
@click.option('--average', type=str, required=True)
def multiclass(
    predictions: list[Path],
    average: str
):
    df = pd.concat([pd.read_json(pred) for pred in predictions])
    labels = df.labels.explode()
    preds = df.preds.explode()
    if average.lower() == 'none':
        average = None
    classes = [l for l in labels.unique() if l != 'O']
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average=average, labels=classes)
    metrics = {}
    if average is None:
        for i, cls in enumerate(classes):
            metrics[f"{cls}_precision"] = p[i]
            metrics[f"{cls}_recall"] = r[i]
            metrics[f"{cls}_f1"] = f[i]
    else:
        metrics = dict(precision=p * 100, recall = r * 100, f1=f * 100)

    pprint(metrics)

if __name__ == "__main__":
    cli()
