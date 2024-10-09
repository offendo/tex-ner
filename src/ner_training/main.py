#!/usr/bin/env python3

import logging
import os
from pathlib import Path
from pprint import pformat
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import click
import evaluate
import numpy as np
import pandas as pd
import ray
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import wandb
from datasets import Dataset, DatasetDict
from more_itertools import chunked, flatten
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from torchcrf import CRF
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorForTokenClassification,
    EvalPrediction,
    PretrainedConfig,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import TokenClassifierOutput

from ner_training.data import load_data, load_mmd_data
from ner_training.model import BertWithCRF, StackedBertWithCRF
from ner_training.utils import *

set_seed(42)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
metric = evaluate.combine([f1_metric, precision_metric, recall_metric])


logging.basicConfig(level=logging.INFO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Running on device: {DEVICE}")


class CRFTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, crf: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.bert.device) if class_weights is not None else None
        if class_weights is not None:
            logging.info(f"Using class weights: {class_weights}")
        self.crf = crf

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        if hasattr(outputs, "loss"):
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def run_predict(
    model: BertWithCRF, dataset: Iterable, batch_size: int, data_collator: Callable, label2id: dict[str, int]
):
    preds = []
    logits = []
    labels = []
    for batch in chunked(dataset, n=batch_size):
        inputs = data_collator(batch)
        batch_out = model.forward(
            input_ids=inputs["input_ids"].to(model.bert.device),
            attention_mask=inputs["attention_mask"].to(model.bert.device),
            labels=inputs["labels"].to(model.bert.device) if "labels" in inputs else None,
        )
        logits.extend(batch_out.logits.detach().numpy())
        if hasattr(batch_out, "predictions") and batch_out.predictions is not None:
            preds.extend(batch_out.predictions)
        if "labels" in inputs:
            labels.extend(inputs["labels"])

    # Decode
    return dict(predictions=preds, labels=labels, logits=logits)


def load_model(
    pretrained_model_name: str | Path,
    label2id: dict[str, int],
    debug: bool,
    crf: bool,
    context_len: int,
    dropout: float = 0.0,
    checkpoint: str | Path | None = None,
    randomize_last_layer: bool = False,
    freeze_base: bool = False,
    freeze_crf: bool = False,
    stacked: bool = False,
    crf_loss_reduction: str = "token_mean",
    add_second_max_to_o: Optional[bool] = None,
):
    id2label = {v: k for k, v in label2id.items()}

    if stacked:
        model = StackedBertWithCRF(
            pretrained_model_name,
            label2id=label2id,
            id2label=id2label,
            context_len=context_len,
            dropout=dropout,
            debug=debug,
            crf=crf,
        )
        logging.info(f"Loaded StackedBertWithCRF model with base of {pretrained_model_name}")
    else:
        model = BertWithCRF(
            pretrained_model_name,
            label2id=label2id,
            id2label=id2label,
            context_len=context_len,
            dropout=dropout,
            debug=debug,
            crf=crf,
            crf_loss_reduction=crf_loss_reduction,
            add_second_max_to_o=add_second_max_to_o,
        )
        logging.info(f"Loaded BertWithCRF model with base of {pretrained_model_name}")

    if checkpoint is not None:
        state_dict = {}
        from safetensors import safe_open

        with safe_open(Path(checkpoint, "model.safetensors"), framework="pt", device=DEVICE) as file:  # type:ignore
            for k in file.keys():
                state_dict[k] = file.get_tensor(k)
        if isinstance(model, BertWithCRF):
            model.load_state_dict(state_dict)
            logging.info(f"Loaded checkpoint from {Path(checkpoint, 'model.safetensors')}")
        elif isinstance(model, StackedBertWithCRF) and "bert.classifier.bias" in state_dict:
            model.base.load_state_dict(state_dict)
            logging.info(f"Loaded checkpoint for base model from {Path(checkpoint, 'model.safetensors')}")
        elif isinstance(model, StackedBertWithCRF) and "base.bert.classifier.bias" in state_dict:
            model.load_state_dict(state_dict)
            logging.info(f"Loaded checkpoint for base model from {Path(checkpoint, 'model.safetensors')}")

    # Freeze BERT if needed
    if freeze_base:
        if hasattr(model.bert, "roberta"):
            bert = model.bert.roberta
        elif hasattr(model.bert, "bert"):
            bert = model.bert.bert
        for param in bert.parameters():
            param.requires_grad = False
        logging.info("Froze base model.")

    # Freeze CRF if needed
    if freeze_crf and model.crf is not None:
        for param in model.crf.parameters():
            param.requires_grad = False
        logging.info("Froze crf.")

    # Randomize the last encoder layer, apparently this can help generlization
    if randomize_last_layer:
        last_layer = None
        if hasattr(model.bert, "roberta"):
            last_layer = model.bert.roberta.encoder.layer[-1]
        elif hasattr(model.bert, "bert"):
            last_layer = model.bert.bert.encoder.layer[-1]
        if last_layer is not None:
            for name, param in last_layer.named_parameters():
                if "weight" in name and "LayerNorm" not in name:
                    torch.nn.init.xavier_normal_(param)
        logging.info("Randomized weights of last layer.")

    return model


def make_compute_metrics(label2id):
    def compute_metrics(eval_out: EvalPrediction):
        logits_and_preds = eval_out.predictions
        labels = eval_out.label_ids
        assert isinstance(labels, np.ndarray)

        if isinstance(logits_and_preds, tuple):
            logits = logits_and_preds[0]
            preds = logits_and_preds[1]
        else:
            logits = logits_and_preds
            preds = np.argmax(logits, axis=-1)

        classes = [cls for cls in label2id.values() if cls != 0]

        ls = [l for l in labels.ravel() if l != -100]
        ps = [p for p, l in zip(preds.ravel(), labels.ravel()) if l != -100]

        if len(label2id) == 2:
            p, r, f, _ = precision_recall_fscore_support(ls, ps, average="binary", pos_label=1)
        else:
            p, r, f, _ = precision_recall_fscore_support(ls, ps, average="micro", labels=classes)
        return dict(precision=p, recall=r, f1=f)

    return compute_metrics


@click.group("cli")
def cli():
    pass


@click.command()
@click.option("--model", type=str)
@click.option("--crf", is_flag=True)
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option("--output_dir", type=click.Path(exists=True, writable=True, file_okay=False, resolve_path=True))
@click.option("--definition", is_flag=True)
@click.option("--theorem", is_flag=True)
@click.option("--proof", is_flag=True)
@click.option("--example", is_flag=True)
@click.option("--reference", is_flag=True)
@click.option("--name", is_flag=True)
@click.option("--context_len", default=512, type=int)
@click.option("--batch_size", default=8)
@click.option("--learning_rate", default=1e-4)
@click.option("--weight_decay", default=0.0)
@click.option("--steps", default=500)
@click.option("--warmup_ratio", default=0.05)
@click.option("--label_smoothing_factor", default=0.1)
@click.option("--dropout", default=0.0)
@click.option("--scheduler", default="linear")
@click.option("--logging_steps", default=10)
@click.option("--debug", is_flag=True)
@click.option("--use_class_weights", is_flag=True)
@click.option("--randomize_last_layer", is_flag=True)
@click.option("--freeze_base", is_flag=True)
@click.option("--freeze_crf", is_flag=True)
@click.option("--examples_as_theorems", is_flag=True)
@click.option("--train_only_tags", "-n", type=click.Choice(["name", "reference"]), default=None, multiple=True)
@click.option("--checkpoint", type=click.Path(exists=True, resolve_path=True), default=None)
@click.option("--stacked", is_flag=True)
@click.option("--crf_loss_reduction", type=click.Choice(["mean", "sum", "token_mean"]), default="token_mean")
@click.option("--add_second_max_to_o", is_flag=True)
def train(
    model: str,
    crf: bool,
    definition: bool,
    theorem: bool,
    proof: bool,
    example: bool,
    name: bool,
    reference: bool,
    context_len: int,
    data_dir: Path,
    output_dir: Path,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    steps: int,
    warmup_ratio: float,
    label_smoothing_factor: float,
    dropout: float,
    scheduler: str,
    logging_steps: int,
    debug: bool,
    use_class_weights: bool,
    randomize_last_layer: bool,
    freeze_base: bool,
    freeze_crf: bool,
    examples_as_theorems: bool,
    train_only_tags: list[str] | None,
    checkpoint: Path | None,
    stacked: bool,
    crf_loss_reduction: str,
    add_second_max_to_o: bool,
):
    label2id = create_multiclass_labels(definition, theorem, proof, example, name, reference)
    logging.info(f"Label map: {label2id}")
    ner_model = load_model(
        model,
        crf=crf,
        context_len=context_len,
        label2id=label2id,
        debug=debug,
        dropout=dropout,
        checkpoint=checkpoint,
        randomize_last_layer=randomize_last_layer,
        freeze_base=freeze_base,
        freeze_crf=freeze_crf,
        stacked=stacked,
        crf_loss_reduction=crf_loss_reduction,
        add_second_max_to_o=add_second_max_to_o,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Data loading
    data = load_data(
        data_dir,
        tokenizer,
        context_len=context_len,
        label2id=label2id,
        examples_as_theorems=examples_as_theorems,
        train_only_tags=train_only_tags,
    )
    collator = DataCollatorForTokenClassification(tokenizer, padding=True, label_pad_token_id=PAD_TOKEN_ID)

    # Build the trainer
    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        max_steps=steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        label_smoothing_factor=label_smoothing_factor,
        optim="adamw_torch",
        lr_scheduler_type=scheduler,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        use_cpu=DEVICE == "cpu",
    )

    if use_class_weights:
        classes = list(label2id.values())
        labels = (
            list(flatten(data["train"]["labels"])) + classes
        )  # add the classes back in so we have at least 1 example
        class_weights = torch.tensor(
            compute_class_weight("balanced", classes=np.array(classes), y=np.array(labels)), dtype=torch.float32
        )
    else:
        class_weights = None

    trainer = CRFTrainer(
        model=ner_model,
        args=args,
        data_collator=collator,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        compute_metrics=make_compute_metrics(label2id),
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(str(Path(output_dir) / "checkpoint-best"))


@click.command()
@click.option("--model", type=str)
@click.option("--crf", is_flag=True)
@click.option("--checkpoint", default=None, type=click.Path(exists=True))
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False, writable=True, resolve_path=True))
@click.option("--output_dir", type=click.Path(exists=True, writable=True, file_okay=False, resolve_path=True))
@click.option("--definition", is_flag=True)
@click.option("--theorem", is_flag=True)
@click.option("--proof", is_flag=True)
@click.option("--example", is_flag=True)
@click.option("--reference", is_flag=True)
@click.option("--name", is_flag=True)
@click.option("--context_len", default=512, type=int)
@click.option("--overlap_len", default=512, type=int)
@click.option("--batch_size", default=8)
@click.option("--debug", is_flag=True)
@click.option("--examples_as_theorems", is_flag=True)
@click.option("--train_only_tags", "-n", type=click.Choice(["name", "reference"]), default=None, multiple=True)
@click.option("--stacked", is_flag=True)
@click.option("--add_second_max_to_o", is_flag=True)
def test(
    model: str,
    crf: bool,
    checkpoint: Path | None,
    definition: bool,
    theorem: bool,
    proof: bool,
    example: bool,
    name: bool,
    reference: bool,
    context_len: int,
    overlap_len: int,
    data_dir: Path,
    output_dir: Path,
    batch_size: int,
    debug: bool,
    examples_as_theorems: bool,
    train_only_tags: list[str],
    stacked: bool,
    add_second_max_to_o: bool,
):
    label2id = create_multiclass_labels(definition, theorem, proof, example, name, reference)
    logging.info(f"Label map: {label2id}")

    id2label = {v: k for k, v in label2id.items()}
    ner_model = load_model(
        model,
        crf=crf,
        context_len=context_len,
        label2id=label2id,
        debug=debug,
        checkpoint=checkpoint,
        dropout=0.0,
        stacked=stacked,
        add_second_max_to_o=add_second_max_to_o,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Data loading
    data = load_data(
        data_dir,
        label2id=label2id,
        tokenizer=tokenizer,
        context_len=context_len,
        examples_as_theorems=examples_as_theorems,
        train_only_tags=train_only_tags,
    )
    collator = DataCollatorForTokenClassification(tokenizer, padding=True, label_pad_token_id=PAD_TOKEN_ID)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        use_cpu=DEVICE == "cpu",
    )
    trainer = CRFTrainer(
        model=ner_model,
        args=args,
        data_collator=collator,
        train_dataset=data["train"],
        compute_metrics=make_compute_metrics(label2id),
    )
    # Run eval on 'test' and 'val'
    for split in ["test", "val"]:
        if isinstance(ner_model, StackedBertWithCRF) or (hasattr(ner_model, "crf") and ner_model.crf is None):
            logits, labels, metrics = trainer.predict(data[split])  # type:ignore
            preds = np.argmax(logits, axis=-1)
            logging.info(pformat(metrics))
        else:
            (logits, preds), labels, metrics = trainer.predict(data[split])  # type:ignore
            preds = np.argmax(logits, axis=-1)
            logging.info(pformat(metrics))

        output = {
            "labels": [[id2label[l] for l in ll if l != PAD_TOKEN_ID] for ll in labels],
            "logits": [[p for p, l in zip(pp, ll) if l != PAD_TOKEN_ID] for pp, ll in zip(logits, labels)],
            "preds": [[id2label[p] for p, l in zip(pp, ll) if l != PAD_TOKEN_ID] for pp, ll in zip(preds, labels)],
            "tokens": [[tokenizer.convert_ids_to_tokens(i) for i in item] for item in data[split]["input_ids"]],
        }
        test_df = pd.DataFrame(output)
        test_df.to_json(Path(output_dir, f"{split}.preds.json"))


@click.command()
@click.option("--model", type=str)
@click.option("--crf", is_flag=True)
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option("--output_dir", type=click.Path(exists=True, writable=True, file_okay=False, resolve_path=True))
@click.option("--definition", is_flag=True)
@click.option("--theorem", is_flag=True)
@click.option("--proof", is_flag=True)
@click.option("--example", is_flag=True)
@click.option("--reference", is_flag=True)
@click.option("--name", is_flag=True)
@click.option("--use_class_weights", is_flag=True)
@click.option("--context_len", default=512, type=int)
@click.option("--steps", default=500)
@click.option("--logging_steps", default=10)
@click.option("--trials", default=50)
@click.option("--debug", is_flag=True)
@click.option("--examples_as_theorems", is_flag=True)
@click.option("--train_only_tags", "-n", type=click.Choice(["name", "reference"]), default=None, multiple=True)
@click.option("--stacked", is_flag=True)
@click.option("--crf_loss_reduction", type=click.Choice(["mean", "sum", "token_mean"]), default="token_mean")
@click.option("--add_second_max_to_o", is_flag=True)
def tune(
    model: str,
    crf: bool,
    definition: bool,
    theorem: bool,
    proof: bool,
    example: bool,
    name: bool,
    reference: bool,
    use_class_weights: bool,
    context_len: int,
    data_dir: Path,
    output_dir: Path,
    steps: int,
    trials: int,
    logging_steps: int,
    debug: bool,
    examples_as_theorems: bool,
    train_only_tags: list[str] | None,
    stacked: bool,
    crf_loss_reduction: str,
    add_second_max_to_o: bool,
):
    label2id = create_multiclass_labels(definition, theorem, proof, example, name, reference)

    def raytune_hp_space(trial):
        return {
            "learning_rate": ray.tune.loguniform(1e-6, 1e-1),
            "per_device_train_batch_size": ray.tune.choice([4, 8, 16, 32]),
            "warmup_ratio": ray.tune.uniform(0.0, 0.1),
            "weight_decay": ray.tune.loguniform(1e-6, 1e-3),
            "lr_scheduler_type": ray.tune.choice(["linear", "cosine", "inverse_sqrt"]),
            "label_smoothing_factor": ray.tune.uniform(0.0, 0.1),
            "max_steps": ray.tune.uniform(1000, 4000),
        }

    def make_model_init(*args, **kwargs):
        def model_init(trial):
            return load_model(*args, **kwargs)

        return model_init

    def compute_objective(metrics: dict[str, float]):
        return metrics["eval_f1"]

    # Data loading
    tokenizer = AutoTokenizer.from_pretrained(model)
    data = load_data(
        data_dir,
        tokenizer,
        context_len=context_len,
        label2id=label2id,
        examples_as_theorems=examples_as_theorems,
        train_only_tags=train_only_tags,
    )
    collator = DataCollatorForTokenClassification(tokenizer, padding=True, label_pad_token_id=PAD_TOKEN_ID)
    args = TrainingArguments(
        disable_tqdm=True,
        output_dir=str(output_dir),
        optim="adamw_torch",
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        metric_for_best_model="f1",
        use_cpu=DEVICE == "cpu",
    )
    trainer = CRFTrainer(
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        compute_metrics=make_compute_metrics(label2id),
        tokenizer=tokenizer,
        model_init=make_model_init(
            model,
            label2id=label2id,
            debug=debug,
            crf=crf,
            context_len=context_len,
            stacked=stacked,
            add_second_max_to_o=add_second_max_to_o,
        ),
        data_collator=collator,
    )
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        compute_objective=compute_objective,
        backend="ray",
        hp_space=raytune_hp_space,
        n_trials=trials,
    )
    logging.info("Completed hyperparameter search.")
    logging.info(best_trial)

    save_path = Path(output_dir, "best-trial.pt")
    with open(save_path, "wb") as f:
        torch.save(best_trial, f)
        logging.info(f"Saved hyperparameter search results to {save_path}.")

    return best_trial


@click.command()
@click.option("--model", type=str)
@click.option("--crf", is_flag=True)
@click.option("--checkpoint", default=None, type=click.Path(exists=True))
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False, writable=True, resolve_path=True))
@click.option("--output_dir", type=click.Path(exists=True, writable=True, file_okay=False, resolve_path=True))
@click.option("--definition", is_flag=True)
@click.option("--theorem", is_flag=True)
@click.option("--proof", is_flag=True)
@click.option("--example", is_flag=True)
@click.option("--reference", is_flag=True)
@click.option("--name", is_flag=True)
@click.option("--context_len", default=512, type=int)
@click.option("--batch_size", default=8)
@click.option("--debug", is_flag=True)
@click.option("--add_second_max_to_o", is_flag=True)
def predict(
    model: str,
    crf: bool,
    checkpoint: Path | None,
    definition: bool,
    theorem: bool,
    proof: bool,
    example: bool,
    name: bool,
    reference: bool,
    context_len: int,
    data_dir: Path,
    output_dir: Path,
    batch_size: int,
    debug: bool,
    add_second_max_to_o: bool,
):
    label2id = create_multiclass_labels(definition, theorem, proof, example, name, reference)
    logging.info(f"Label map: {label2id}")

    id2label = {v: k for k, v in label2id.items()}
    ner_model = load_model(
        model,
        crf=crf,
        context_len=context_len,
        label2id=label2id,
        debug=debug,
        checkpoint=checkpoint,
        dropout=0.0,
        add_second_max_to_o=add_second_max_to_o,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Data loading
    data = load_mmd_data(
        data_dir,
        tokenizer=tokenizer,
        context_len=context_len,
    )
    collator = DataCollatorForTokenClassification(tokenizer, padding=True, label_pad_token_id=PAD_TOKEN_ID)

    # Run the predictions
    ner_model.eval()
    loader = DataLoader(data.remove_columns(["file"]), batch_size=batch_size, shuffle=False, collate_fn=collator)
    all_predictions = []
    for idx, batch in enumerate(tqdm(loader), start=1):
        if idx % 100 == 0:
            total = len(all_predictions)
            start = max([total - 100, 0])
            output = {
                "file": data["file"][start:total],
                "preds": [[id2label[p] for p in pp if p != PAD_TOKEN_ID] for pp in all_predictions[start:total]],
                "tokens": [
                    [tokenizer.convert_ids_to_tokens(i) for i in item] for item in data["input_ids"][start:total]
                ],
            }
            test_df = pd.DataFrame(output)
            # flatten each file's output into one row
            test_df = test_df.groupby("file").agg(lambda xs: [y for x in xs for y in x]).reset_index()
            test_df.to_json(Path(output_dir, f"mmd.preds-{idx}.json"))
            Path(output_dir, f"mmd.preds-{idx-1}.json").unlink(missing_ok=True)

        batch_out = ner_model.forward(
            input_ids=batch["input_ids"].to(ner_model.bert.device),
            attention_mask=batch["attention_mask"].to(ner_model.bert.device),
            labels=batch["labels"].to(ner_model.bert.device) if "labels" in batch else None,
        )
        assert isinstance(batch_out.predictions, torch.Tensor)
        all_predictions.extend(batch_out.predictions.tolist())

    output = {
        "file": data["file"],
        "preds": [[id2label[p] for p in pp if p != PAD_TOKEN_ID] for pp in all_predictions],
        "tokens": [[tokenizer.convert_ids_to_tokens(i) for i in item] for item in data["input_ids"]],
    }
    test_df = pd.DataFrame(output)

    # flatten each file's output into one row
    test_df = test_df.groupby("file").agg(lambda xs: [y for x in xs for y in x]).reset_index()
    test_df.to_json(Path(output_dir, f"mmd.preds.json"))

    # Conver the predictions to a list of annotations
    # TODO figure out how to get start/end indices
    for idx, row in test_df.iterrows():
        annos = convert_tags_to_annotations(tokens=row.tokens, tags=row.preds, tokenizer=tokenizer)
        anno_df = pd.DataFrame.from_records(annos)
        anno_df.to_json(Path(output_dir, f"{Path(row.file).stem}.annos.json"))


if __name__ == "__main__":
    cli.add_command(train)
    cli.add_command(test)
    cli.add_command(predict)
    cli.add_command(tune)
    cli()
