#!/usr/bin/env python3


import pandas as pd
from transformers import PreTrainedTokenizer, BatchEncoding


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


def create_name_or_ref_tags(tag: str, data: pd.DataFrame, tokenizer: PreTrainedTokenizer):
    defs_and_thms = data[data.tag.isin(["theorem", "definition"])]
    records = []
    for idx, outer in defs_and_thms.iterrows():
        inner = data[(data.fileid == outer.fileid) & (data.start >= outer.start) & (data.end <= outer.end)]
        for i, row in inner.iterrows():
            new_start = row.start - outer.start
            new_end = row.end - outer.start
            tags = [
                (
                    f"B-{tag}"
                    if row.tag == tag and i == new_start
                    else f"I-{tag}" if (new_start <= i <= new_end and row.tag == tag) else "O"
                )
                for i in range(len(outer.text))
            ]
            tokens = tokenizer(outer.text)
            aligned_tags = align_annotations_to_tokens(tokens, tags)
            records.append(
                {
                    "tex": outer.text,
                    "start": new_start,
                    "end": new_end,
                    "tags": aligned_tags,
                    "tokens": tokens["input_ids"],
                }
            )
    return pd.DataFrame.from_records(records)
