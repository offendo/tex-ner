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
@click.option('--predictions', type=click.Path(exists=True), required=True)
@click.option('--output', type=click.Path(exists=False), required=True)
@click.option('--norm-all', 'norm', is_flag=True, flag_value='all')
@click.option('--norm-pred', 'norm', is_flag=True, flag_value='pred')
@click.option('--norm-true', 'norm', is_flag=True, flag_value='true')
@click.option('--show', is_flag=True)
def heatmap(
    predictions: Path,
    output: Path,
    norm: Optional[str],
    show: bool,
):
    df = pd.read_json(predictions)
    labels = [l for l in df.labels.explode().unique() if l != 'O']
    fig = plt.figure(figsize=(10,10))
    mat = confusion_matrix(df.labels.explode(), df.preds.explode(), labels=labels[1:], normalize=norm)
    g = sns.heatmap(mat, xticklabels=labels[1:], yticklabels=labels[1:], square=True, annot=True, cbar=False)
    plt.savefig(output)
    if show:
        fig.show()
        input()

@cli.command()
@click.option('--predictions', type=click.Path(exists=True), required=True)
@click.option('--average', type=str, required=True)
def multilabel(
    predictions: Path,
    average: str
):
    df = pd.read_json(predictions)
    classes = [l for l in df.labels.explode().str.split('-').explode().unique() if l != 'O']
    print('Labels: ', classes)
    mlb = MultiLabelBinarizer(classes=classes)
    labels = mlb.fit_transform(df.labels.explode().str.split('-'))
    preds = mlb.transform(df.preds.explode().str.split('-'))
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average=average)
    pprint(dict(precision=p * 100, recall = r * 100, f1=f * 100))

if __name__ == "__main__":
    cli()
