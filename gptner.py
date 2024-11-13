import json
import logging
import os
from argparse import ArgumentParser

import openai
import pandas as pd
import tiktoken
from more_itertools import chunked
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

INST = """In the following LaTeX document, extract entities of the following types and return them in JSON output.
1. definition
2. theorem
3. proof
4. example
Your output should be a single JSON with 4 keys corresponding to the 4 entity types above. Spans may be part of multiple entities. Do not hallucinate text.
"""
SYST = "You are an expert mathematician who is fluent in reading LaTeX and extracting information."

client = openai.OpenAI()
tokenizer = tiktoken.encoding_for_model("gpt-4o-mini-2024-07-18")

with open("./response_schema.json", "r") as f:
    schema = json.load(f)


def merge_neighbors(df: pd.DataFrame):
    new = []
    idx = 0
    while idx < len(df):
        row = df.iloc[idx]
        start = row.start
        end = row.end
        text = row.text
        while True:
            # Bounds checking
            if idx + 1 >= len(df):
                break

            # Should we merge these two?
            nxt = df.iloc[idx + 1]
            if not (nxt.start <= row.end + 1 and row.tag == nxt.tag):
                break
            # Do the merge
            idx += 1
            end = nxt.end
            text += nxt.text
        item = dict(tag=row.tag, text=text, start=start, end=end)
        new.append(item)
        idx += 1

    return pd.DataFrame(new)


def format_input(prompt, instruction, system):
    return [
        ChatCompletionSystemMessageParam(role="system", content=system),
        ChatCompletionUserMessageParam(role="user", content=instruction + "\n\n" + prompt),
    ]


def complete(mmd: str, model: str, max_len: int):
    preds = []

    tokens = tokenizer.encode(mmd)
    n_chunks = len(tokens) // max_len
    logging.info(f"Working in {n_chunks} chunks")
    for snippet in tqdm(chunked(tokens, n=max_len), total=n_chunks):
        snippet_text = tokenizer.decode(snippet)
        messages = format_input(snippet_text, INST, SYST)
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=schema,
        )
        output = completion.choices[0].message.content
        assert output is not None
        preds.append(json.loads(output))

    # Build a list of records
    annos = []
    for p in preds:
        for tag, texts in p.items():
            for t in texts:
                start = mmd.find(t)
                end = start + len(t)
                if start == -1:
                    logging.warning(f"Failed to find prediction in input string: {t}")
                    end = -1
                annos.append(dict(tag=tag, text=t, start=start, end=end))
    return pd.DataFrame.from_records(annos)


if __name__ == "__main__":
    parser = ArgumentParser("gpt-ner")
    parser.add_argument("--file", "-f", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--model", "-m", type=str, default="ft:gpt-4o-mini-2024-07-18:uc-santa-cruz-jlab-nlp::AQ2BxcPd")
    parser.add_argument("--max_len", "-l", type=int, default=512)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    with open(args.file, "r") as f:
        mmd = f.read()
        tokens = tokenizer.encode(mmd)
        logging.info(f"Read input from {args.file}: {len(mmd)} characters/{len(tokens)} tokens.")
        est_in = (len(tokens) / 1_000_000) * 0.150
        est_out = (len(tokens) / 1_000_000) * 0.600
        logging.info(f"Estimated price: ${est_in + est_out:0.2f}")

    df = complete(mmd, model=args.model, max_len=args.max_len)
    logging.info(f"Finished generating, got {len(df)} annotations.")

    df = df[df["start"] != -1]
    logging.info(f"Dropped rows whose start location went unfound, left with {len(df)} annotations.")

    df = merge_neighbors(df.sort_values(["tag", "start"]))  # type:ignore
    logging.info(f"Merged neighbors, left with {len(df)} annotations.")

    df.to_json(args.output)
    logging.info(f"Saved to {args.output}")
