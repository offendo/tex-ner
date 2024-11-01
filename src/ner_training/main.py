#!/usr/bin/env python3

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import evaluate
import numpy as np
import pandas as pd
import ray
import torch
from datasets import Dataset, DatasetDict
from icecream import ic
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, EvalPrediction, HfArgumentParser, Trainer, TrainingArguments, set_seed

from ner_training.collator import DataCollatorForTokenClassification
from ner_training.config import Config
from ner_training.data import load_dataset
from ner_training.model import BertWithCRF, StackedBertWithCRF
from ner_training.trainer import CRFTrainer
from ner_training.utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
metric = evaluate.combine([f1_metric, precision_metric, recall_metric])


logging.basicConfig(level=logging.INFO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Running on device: {DEVICE}")


@dataclass
class TrainingObject:
    model: BertWithCRF | StackedBertWithCRF
    dataset: Dataset | DatasetDict
    label2id: dict[str, int]
    trainer: Trainer

    def __iter__(self):
        for item in [self.model, self.dataset, self.label2id, self.trainer]:
            yield item

    @classmethod
    def setup(cls, config: Config, training_args: TrainingArguments):
        label2id = create_multiclass_labels(
            definition=config.definition,
            theorem=config.theorem,
            proof=config.proof,
            example=config.example,
            name=config.name,
            reference=config.reference,
            use_preset=config.use_preset,
        )
        model = load_model(config)
        # Data loading
        dataset = load_dataset(config)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        training_args.use_cpu = DEVICE == "cpu"
        collator = DataCollatorForTokenClassification(
            tokenizer, padding=True, label_pad_token_id=PAD_TOKEN_ID, tag_pad_token_id=0
        )
        trainer = CRFTrainer(
            model=model,
            args=training_args,
            data_collator=collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"] if dataset["val"] else None,
            compute_metrics=make_compute_metrics(label2id),
            callbacks=(
                [FreezeBaseAfterStepsCallback(config.freeze_base_after_steps)]
                if config.freeze_base_after_steps >= 0
                else None
            ),
        )

        return cls(model, dataset, label2id, trainer)


def load_model(config: Config):
    ModelClass = StackedBertWithCRF if config.stacked else BertWithCRF
    model = ModelClass(config)
    logging.info(f"Loaded {ModelClass} model with base of {config.model_name_or_path}")

    if config.checkpoint is not None:
        state_dict = {}
        from safetensors import safe_open

        with safe_open(
            Path(config.checkpoint, "model.safetensors"), framework="pt", device=DEVICE
        ) as file:  # type:ignore
            for k in file.keys():
                state_dict[k] = file.get_tensor(k)
        if isinstance(model, BertWithCRF):
            model.load_state_dict(state_dict)
            logging.info(f"Loaded checkpoint from {Path(config.checkpoint, 'model.safetensors')}")
        elif isinstance(model, StackedBertWithCRF) and "bert.classifier.bias" in state_dict:
            model.base.load_state_dict(state_dict)
            logging.info(f"Loaded checkpoint for base model from {Path(config.checkpoint, 'model.safetensors')}")
        elif isinstance(model, StackedBertWithCRF) and "base.bert.classifier.bias" in state_dict:
            model.load_state_dict(state_dict)
            logging.info(f"Loaded checkpoint for base model from {Path(config.checkpoint, 'model.safetensors')}")

    # Freeze BERT if needed
    if config.freeze_base:
        if hasattr(model.bert, "roberta"):
            bert = model.bert.roberta
        elif hasattr(model.bert, "bert"):
            bert = model.bert.bert
        for param in bert.parameters():
            param.requires_grad = False
        logging.info("Froze base model.")

    # Freeze CRF if needed
    if config.freeze_crf and model.crf is not None:
        for param in model.crf.parameters():
            param.requires_grad = False
        logging.info("Froze crf.")

    # Randomize the last encoder layer, apparently this can help generlization
    if config.randomize_last_layer:
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
            logits, preds = logits_and_preds  # type:ignore
            bert_preds = np.argmax(logits, axis=-1)
        else:
            logits = logits_and_preds
            preds = np.argmax(logits, axis=-1)
            bert_preds = None

        classes = [cls for cls in label2id.values() if cls != 0]

        ls = [l for l in labels.ravel() if l != -100]
        ps = [p for p, l in zip(preds.ravel(), labels.ravel()) if l != -100]
        if bert_preds is not None:
            bps = [p for p, l in zip(bert_preds.ravel(), labels.ravel()) if l != -100]

        if len(label2id) == 2:
            p, r, f, _ = precision_recall_fscore_support(ls, ps, average="binary", pos_label=1)
        else:
            p, r, f, _ = precision_recall_fscore_support(ls, ps, average="micro", labels=classes)
            bp, br, bf, _ = precision_recall_fscore_support(ls, bps, average="micro", labels=classes)
        return dict(precision=p, recall=r, f1=f, bert_precision=bp, bert_recall=br, bert_f1=bf)

    return compute_metrics


def train(config: Config, training_args: TrainingArguments):
    training_obj = TrainingObject.setup(config, training_args)
    training_obj.trainer.train()
    training_obj.trainer.save_model(str(Path(training_args.output_dir) / "checkpoint-best"))

    state_dict = {}
    from safetensors import safe_open
    from safetensors.torch import save_file

    # Checkpoint averaging
    if not config.average_checkpoints:
        return

    logging.info("Averaging checkpoints")
    checkpoints = [
        Path(training_args.output_dir, ckpt)
        for ckpt in os.listdir(training_args.output_dir)
        if "checkpoint" in ckpt and "best" not in ckpt
    ]
    logging.info(f"Found {len(checkpoints)} checkpoints")
    for checkpoint in checkpoints:
        with safe_open(Path(checkpoint, "model.safetensors"), framework="pt", device="cpu") as file:  # type:ignore
            for k in file.keys():
                if k in state_dict:
                    state_dict[k] += file.get_tensor(k) / len(checkpoints)
                else:
                    state_dict[k] = file.get_tensor(k) / len(checkpoints)

    # Save the averaged checkpoints
    avg_path = Path(training_args.output_dir, "checkpoint-avg", "model.safetensors")
    avg_path.parent.mkdir(exist_ok=True, parents=True)
    save_file(state_dict, avg_path)
    logging.info(f"Saved at {avg_path}")


def test(config: Config, training_args: TrainingArguments):
    model, dataset, label2id, trainer = TrainingObject.setup(config, training_args)
    id2label = {v: k for k, v in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    # Run eval on 'test' and 'val'
    for split in ["test", "val"] + (["train"] if config.predict_on_train else []):
        if isinstance(model, StackedBertWithCRF) or (hasattr(model, "crf") and model.crf is None):
            logits, labels, metrics = trainer.predict(dataset[split], metric_key_prefix=split)  # type:ignore
            preds = np.argmax(logits, axis=-1)
            logging.info(pformat(metrics))
        else:
            (logits, preds), labels, metrics = trainer.predict(dataset[split], metric_key_prefix=split)  # type:ignore
            # preds = np.argmax(logits, axis=-1)
            logging.info(pformat(metrics))

        output = {
            "labels": [[id2label[l] for l in ll if l != PAD_TOKEN_ID] for ll in labels],
            "logits": [[p for p, l in zip(pp, ll) if l != PAD_TOKEN_ID] for pp, ll in zip(logits, labels)],
            "preds": [[id2label[p] for p, l in zip(pp, ll) if l != PAD_TOKEN_ID] for pp, ll in zip(preds, labels)],
            "tokens": [[tokenizer.convert_ids_to_tokens(i) for i in item] for item in dataset[split]["input_ids"]],
        }
        test_df = pd.DataFrame(output)
        if config.output_name is not None:
            test_df.to_json(Path(training_args.output_dir, f"{config.output_name}.{split}.preds.json"))
        else:
            test_df.to_json(Path(training_args.output_dir, f"{split}.preds.json"))


def tune(config: Config, training_args: TrainingArguments):
    label2id = create_multiclass_labels(
        definition=config.definition,
        theorem=config.theorem,
        proof=config.proof,
        example=config.example,
        name=config.name,
        reference=config.reference,
        use_preset=config.use_preset,
    )
    # Data loading
    dataset = load_dataset(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    training_args.use_cpu = DEVICE == "cpu"
    collator = DataCollatorForTokenClassification(
        tokenizer, padding=True, label_pad_token_id=PAD_TOKEN_ID, tag_pad_token_id=0
    )

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
    trainer = CRFTrainer(
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        compute_metrics=make_compute_metrics(label2id),
        tokenizer=tokenizer,
        model_init=make_model_init(config),
        data_collator=collator,
    )
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        compute_objective=compute_objective,
        backend="ray",
        hp_space=raytune_hp_space,
        n_trials=config.trials,
    )
    logging.info("Completed hyperparameter search.")
    logging.info(best_trial)

    save_path = Path(training_args.output_dir, "best-trial.pt")
    with open(save_path, "wb") as f:
        torch.save(best_trial, f)
        logging.info(f"Saved hyperparameter search results to {save_path}.")

    return best_trial


def predict(config: Config, training_args: TrainingArguments):

    model, dataset, label2id, trainer = TrainingObject.setup(config, training_args)
    id2label = {v: k for k, v in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    # Run the predictions
    model.eval()
    to_remove = ["file"] + (["tag"] if "tag" in dataset.column_names else [])
    loader = DataLoader(
        dataset.remove_columns(to_remove),
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=trainer.data_collator,
    )
    all_predictions = []
    all_logits = []
    for idx, batch in enumerate(tqdm(loader), start=1):
        if idx % 100 == 0:
            total = len(all_predictions)
            start = max([total - 100, 0])
            output = {
                "file": dataset["file"][start:total],
                "preds": [[id2label[p] for p in pp if p != PAD_TOKEN_ID] for pp in all_predictions[start:total]],
                "logits": [
                    [l for p, l in zip(pp, ll) if p != PAD_TOKEN_ID]
                    for pp, ll in zip(all_predictions[start:total], all_logits[start:total])
                ],
                "tokens": [
                    [tokenizer.convert_ids_to_tokens(i) for i in item] for item in dataset["input_ids"][start:total]
                ],
                "tag": dataset["tag"][start:total],
            }
            test_df = pd.DataFrame(output)

            # flatten each file's output into one row
            test_df = test_df.groupby("file").agg(lambda xs: [y for x in xs for y in x]).reset_index()
            test_df.to_json(Path(training_args.output_dir, f"mmd.preds-{idx}.json"))

        batch_out = model.forward(
            input_ids=batch["input_ids"].to(model.bert.device),
            attention_mask=batch["attention_mask"].to(model.bert.device),
            labels=batch["labels"].to(model.bert.device) if "labels" in batch else None,
        )
        assert isinstance(batch_out.predictions, torch.Tensor)
        all_predictions.extend(batch_out.predictions.tolist())
        all_logits.extend(batch_out.logits.tolist())

    output = {
        "file": dataset["file"],
        "preds": [[id2label[p] for p in pp if p != PAD_TOKEN_ID] for pp in all_predictions],
        "logits": [[l for p, l in zip(pp, ll) if p != PAD_TOKEN_ID] for pp, ll in zip(all_predictions, all_logits)],
        "tokens": [[tokenizer.convert_ids_to_tokens(i) for i in item] for item in dataset["input_ids"]],
        "tag": dataset["tag"],
    }
    test_df = pd.DataFrame(output)

    # flatten each file's output into one row
    test_df = test_df.groupby("file").agg(lambda xs: [y for x in xs for y in x]).reset_index()
    test_df.to_json(Path(training_args.output_dir, f"mmd.preds.json"))

    # Conver the predictions to a list of annotations
    # TODO figure out how to get start/end indices
    for idx, row in test_df.iterrows():
        annos = convert_tags_to_annotations(tokens=row.tokens, tags=row.preds, tokenizer=tokenizer)
        anno_df = pd.DataFrame.from_records(annos)
        anno_df.to_json(Path(training_args.output_dir, f"{Path(row.file).stem}.annos.json"))


if __name__ == "__main__":
    parser = HfArgumentParser([Config, TrainingArguments])  # type:ignore
    config, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    logging.info("Set seed to {training_args.seed}")

    if config.run_train:
        train(config, training_args)
    if config.run_test:
        test(config, training_args)
    if config.run_predict:
        predict(config, training_args)
    if config.run_tune:
        tune(config, training_args)
