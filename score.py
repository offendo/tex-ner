from pprint import pprint, pformat
from typing import Optional
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import click
import json
from sklearn.preprocessing import MultiLabelBinarizer


def prettyprint(d):
    s = pformat(d, indent=4, width=50)
    print(json.dumps(d, indent=2))
    return json.dumps(d)

def pretty(f):
    return f"{f:0.2f}"

@click.group()
def cli():
    pass

@cli.command()
@click.option('--predictions', '-p', type=click.Path(exists=True), multiple=True)
@click.option('--name', '-n', type=str, default=None)
@click.option('--type', '-t', type=str, default="default")
@click.option('--output', type=click.Path(exists=False), required=True)
@click.option('--norm-all', 'norm', is_flag=True, flag_value='all')
@click.option('--norm-pred', 'norm', is_flag=True, flag_value='pred')
@click.option('--norm-true', 'norm', is_flag=True, flag_value='true')
@click.option('--show', is_flag=True)
def heatmap(
    predictions: list[Path | str],
    name: str,
    type: str,
    output: Path,
    norm: Optional[str],
    show: bool,
):
    if name is not None:
        predictions = [f"results/{name}/{type}.test.preds.json", f"results/{name}/{type}.val.preds.json"]
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
@click.option('--predictions', '-p', type=click.Path(exists=True), multiple=True)
@click.option('--name', '-n', type=str, default=None)
@click.option('--type', '-t', type=str, default="default")
@click.option('--average', type=str, required=False, default=None)
def multilabel(
    predictions: list[Path | str],
    name: str,
    type: str,
    average: str
):
    if name is not None:
        predictions = [f"results/{name}/{type}.test.preds.json", f"results/{name}/{type}.val.preds.json"]
    df = pd.concat([pd.read_json(pred) for pred in predictions])
    classes = [l for l in df.labels.explode().str.split('-').explode().unique() if l != 'O']
    mlb = MultiLabelBinarizer(classes=classes)
    labels = mlb.fit_transform(df.labels.explode().str.split('-'))
    preds = mlb.transform(df.preds.explode().str.split('-'))
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average=average)
    if average is None:
        metrics = {}
        for pp, rr, ff, c in zip(p, r, f, classes):
            metrics[c] = dict(precision=pretty(pp * 100), recall =pretty(rr * 100), f1=pretty(ff * 100))
    else:
        metrics = dict(precision=pretty(p * 100), recall = pretty(r * 100), f1=pretty(f * 100))
    prettyprint(metrics)

@cli.command()
@click.option('--predictions', '-p', type=click.Path(exists=True), multiple=True)
@click.option('--name', '-n', type=str, default=None)
@click.option('--type', '-t', type=str, default="default")
@click.option('--average', type=str, required=False, default=None)
def multiclass(
    predictions: list[Path | str],
    name: str,
    type: str,
    average: str
):
    if name is not None:
        predictions = [f"results/{name}/{type}.test.preds.json", f"results/{name}/{type}.val.preds.json"]
    df = pd.concat([pd.read_json(pred) for pred in predictions])
    labels = df.labels.explode()
    preds = df.preds.explode()
    classes = [l for l in labels.unique() if l != 'O']
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average=average, labels=classes)
    metrics = {}
    if average is None:
        metrics = {}
        for pp, rr, ff, c in zip(p, r, f, classes):
            metrics[c] = dict(precision=pretty(pp * 100), recall =pretty(rr * 100), f1=pretty(ff * 100))
    else:
        metrics = dict(precision=pretty(p * 100), recall = pretty(r * 100), f1=pretty(f * 100))

    prettyprint(metrics)

if __name__ == "__main__":
    cli()
