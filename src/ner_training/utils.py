#!/usr/bin/env python3
import logging

import pandas as pd
import json
from pathlib import Path
from more_itertools import flatten
from transformers import (
    BatchEncoding,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
)

PAD_TOKEN_ID = -100


class FreezeBaseAfterStepsCallback(TrainerCallback):

    def __init__(self, freeze_after_step) -> None:
        super().__init__()
        self.freeze_base_after_step = freeze_after_step

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == self.freeze_base_after_step:
            logging.info(f"Reached step {self.freeze_base_after_step}, freezing bert weights.")
            for param in kwargs["model"].bert.parameters():
                param.requires_grad = False
            logging.info(f"Now resetting the learning rate to {args.learning_rate}")

            # Reset the learning rate
            scheduler = kwargs["scheduler"]
            scheduler._initial_step()


def create_multiclass_labels(
    definition: bool = False,
    theorem: bool = False,
    proof: bool = False,
    example: bool = False,
    name: bool = False,
    reference: bool = False,
    add_prefix: bool = False,
):
    class_names = tuple(
        k
        for k, v in dict(
            definition=definition,
            theorem=theorem,
            proof=proof,
            example=example,
            name=name,
            reference=reference,
        ).items()
        if v
    )

    labels = []
    for arg in sorted(class_names):
        new = []
        for label in labels:
            new.append(f"{label}-{arg}")
        new.append(arg)
        labels.extend(new)

    if add_prefix:
        labels = [f"B-{lab}" for lab in labels] + [f"I-{lab}" for lab in labels]
    labels = ["O"] + labels
    for lab in labels:
        if "name" in lab and "reference" in lab:
            labels.remove(lab)

    return {lab: idx for idx, lab in enumerate(labels)}


def convert_label_to_idx(tags: list[str], label2id: dict[str, int]):
    filtered_tags = sorted([t for t in tags if t in label2id])
    if len(filtered_tags) == 0:
        return label2id["O"]
    return label2id["-".join(filtered_tags)]


def align_annotations_to_tokens(tokens: BatchEncoding, char_tags: list[str]) -> list[list[str]]:
    """Converts character-level annotations to token-level

    Parameters
    ----------
    tokens : BatchEncoding
        Output of a huggingface tokenizer on text
    char_tags : list[list[str]]
        Character-level tags (list of tags per character)

    Returns
    -------
    list[str]
        Token-level tags (list of tags per token)
    """

    aligned_tags: list[list[str]] = []
    for idx in range(len(tokens.input_ids)):
        span = tokens.token_to_chars(idx)
        if span is None:
            continue

        tags_for_token = list(set(char_tags[span.start : span.end]))
        if "O" in tags_for_token and len(tags_for_token) > 1:
            tags_for_token.remove("O")

        # Ensure that we only have B- or I- but not both
        for tag in tags_for_token:
            b = tag.replace("I-", "B-")
            i = tag.replace("B-", "I-")
            if b in tags_for_token and i in tags_for_token:
                tags_for_token.remove(i)
        aligned_tags.append(tags_for_token)
    return aligned_tags


def align_tokens_annotations_to_words(tokens: BatchEncoding, token_tags: list[list[str]]) -> list[set[str]]:
    """Converts token-level annotations to word-level

    Parameters
    ----------
    tokens : BatchEncoding
        Output of a huggingface tokenizer on text
    char_tags : list[list[str]]
        Token-level tags (list of tags per character)

    Returns
    -------
    list[set[str]]
        Word-level tags (list of tags per word)
    """

    aligned_tags: list[set[str]] = []
    for idx in range(len(tokens.input_ids)):
        word_idx = tokens.token_to_word(idx)
        tags_for_word = list(set(token_tags[idx]))
        if "O" in tags_for_word and len(tags_for_word) > 1:
            tags_for_word.remove("O")

        # Ensure that we only have B- or I- but not both
        for tag in tags_for_word:
            b = tag.replace("I-", "B-")
            i = tag.replace("B-", "I-")
            if b in tags_for_word and i in tags_for_word:
                tags_for_word.remove(i)
        if word_idx >= len(aligned_tags):
            aligned_tags.append(set(tags_for_word))
        else:
            aligned_tags[word_idx].update(tags_for_word)
    return aligned_tags


def create_name_or_ref_tags(data: pd.DataFrame, tokenizer: PreTrainedTokenizer, force_reference: bool = True):
    defs_and_thms = data[data.tag.isin(["theorem", "definition", "example"])]
    records = []
    for idx, outer in defs_and_thms.iterrows():
        inner = data[
            (data.fileid == outer.fileid)
            & (data.start >= outer.start)
            & (data.end <= outer.end)
            & (data.tag.isin(["name", "reference"]))
        ]
        assert isinstance(inner, pd.DataFrame)

        def find_all(text, pattern):
            """Non-regex version of re.findall because I don't want patterns"""
            start = 0
            while True:
                start = text.find(pattern, start)
                if start == -1:
                    return
                yield start
                start += len(pattern)

        # Make sure the references are being repeated
        copies = []
        for idx, ref in inner[inner.tag == "reference"].iterrows():
            if len(ref.text) == 1:
                search_text = f" {ref.text.strip()} "
            else:
                search_text = ref.text
            for start in find_all(outer.text, search_text):
                # We have to offset the start index by the outer start index
                # since we're subtracting the same offset in the next loop
                start = start + outer.start
                end = start + len(ref.text)
                if start == ref.start:
                    continue
                other_tags = inner[
                    ((inner.start <= start) & (start <= inner.end)) | ((inner.start <= end) & (end <= inner.end))
                ]
                if len(other_tags) > 0:
                    continue
                logging.debug(f"Adding a copy of {ref.text}")
                copies.append(
                    dict(
                        start=start + outer.start,
                        end=start + outer.start + len(ref.text),
                        text=ref.text,
                        tag="reference",
                    )
                )
        merged = pd.concat([inner[["start", "end", "text", "tag"]], pd.DataFrame.from_records(copies)])

        if force_reference and (merged.tag == "reference").sum() == 0:
            continue

        # Now we need to merge all the examples
        current_records = []
        tex = outer.text
        tokens = tokenizer(outer.text)
        for i, row in merged.iterrows():
            new_start = row.start - outer.start
            new_end = row.end - outer.start
            row_tags = [
                (f"B-{row.tag}" if i == new_start else f"I-{row.tag}" if (new_start <= i <= new_end) else "O")
                for i in range(len(outer.text))
            ]
            aligned_tags = align_annotations_to_tokens(tokens, row_tags)
            current_records.append(aligned_tags)

        if not current_records:
            continue

        merged_tags = [list(set(flatten(tags_at_idx))) for tags_at_idx in zip(*current_records)]

        records.append({"tex": tex, "tokens": tokens, "tags": merged_tags})

    return pd.DataFrame.from_records(records)


def postprocess(tags: list[str]):
    return tags


def convert_tags_to_annotations(tokens: list[str], tags: list[str], tokenizer: PreTrainedTokenizer):
    annotations = []
    current_tags = set()
    current_annos = {
        "definition": [],
        "theorem": [],
        "proof": [],
        "example": [],
        "name": [],
        "reference": [],
    }
    for idx, (tok, tag) in enumerate(zip(tokens, postprocess(tags))):
        labels = [l for l in tag.split("-") if l != "O"]
        to_remove = set()
        for cur in current_tags:
            # If it's not in the new list, we finished an annotation
            if cur not in labels:
                toks = current_annos[cur]
                mmd = tokenizer.convert_tokens_to_string(toks)
                start = len(tokenizer.convert_tokens_to_string(tokens[: idx - len(toks)]))
                end = start + len(mmd)
                annotations.append(
                    {
                        "tex": mmd,
                        "tag": cur,
                        "end_token_idx": idx,
                        "start": start - len(tokenizer.bos_token),  # for some reason, this is offset by <s>
                        "end": end - 1,
                    }
                )
                current_annos[cur] = []
                to_remove.add(cur)
        current_tags.difference_update(to_remove)

        for lab in labels:
            current_annos[lab].append(tok)
            current_tags.add(lab)
    return annotations


def convert_iob_tags_to_conll(annos: str | Path, output_file: str | Path, tokenizer: PreTrainedTokenizer):
    with open(annos, "r") as f:
        data = json.load(f)

    tokens, tags = zip(*data["iob_tags"])

    tex = tokenizer.convert_tokens_to_string(tokens)
    enc = tokenizer(tex, add_special_tokens=False)
    word_level_tags = align_tokens_annotations_to_words(enc, tags)
    word_idxs = enc.word_ids()
    word_spans = []
    for token_idx in range(len(enc.input_ids)):
        word_idx = word_idxs[token_idx]
        span = enc.token_to_chars(token_idx)
        if word_idx is None or span is None:
            continue
        start, end = span
        if word_idx < len(word_spans):
            prev_start, prev_end = word_spans[word_idx]
            word_spans[word_idx] = (prev_start, end)
        else:
            word_spans.append((start, end))
    words = [tex[start:end] for start, end in word_spans]
    label2id = create_multiclass_labels(definition=True, theorem=True, proof=True, example=True)
    lines = []
    conll_tags = []
    index = 0
    for word, word_tags in zip(words, word_level_tags):
        if len(word.strip()) == 0:
            continue
        if len(word_tags) == 0:
            tag = "O"
        else:
            word_tags = [t.replace("I-", "").replace("B-", "") for t in word_tags]
            filtered_tags = sorted([t for t in word_tags if t in label2id])
            tag = ".".join(filtered_tags)

            # If we're at the beginning of a new tag, use 'B-'
            if index == 0 or tag not in conll_tags[index - 1]:
                tag = f"B-{tag}"
            else:
                tag = f"I-{tag}"

        conll_tags.append(tag)
        index += 1

        line = f"{word}\t{tag}"
        lines.append(line)
    conll = "\n".join(lines)
    with open(output_file, "w") as f:
        f.write(conll)
    return conll
