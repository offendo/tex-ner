""" CLI to launch completion requests to OpenAI. Works with both real-time requests and the Batch API.

COMMANDS:
  texner {process, batch, monitor}

=======================================================

NAME:
  process
DESCRIPTION:
  Launch completion job using completion API (full price, real-time)
USAGE:
  texner process [-h] (--file FILE | --filelist FILELIST) --output OUTPUT --model MODEL --max_len MAX_LEN
    options:
      -h, --help           show this help message and exit
      --file FILE          Path to input file (if only processing one)
      --filelist FILELIST  Path to a file containing a list of input files
      --output OUTPUT      Path to output directory
      --model MODEL        Model ID to use
      --max_len MAX_LEN    Max input tokens per request

=======================================================

NAME:
  batch
DESCRIPTION:
  Launch completion job using Batch API (half price, async, 24h max turnaround time)
USAGE:
  texner batch [-h] (--file FILE | --filelist FILELIST) --output OUTPUT --model MODEL --max_len MAX_LEN
    options:
      -h, --help           show this help message and exit
      --file FILE          Path to input file (if only processing one)
      --filelist FILELIST  Path to a file containing a list of input files
      --output OUTPUT      Path to output directory
      --model MODEL        Model ID to use
      --max_len MAX_LEN    Max input tokens per request

=======================================================

NAME:
  add_names
DESCRIPTION:
  Launch completion job for name/refs using Batch API (half price, async, 24h max turnaround time)
USAGE:
  texner add_names [-h] (--file FILE | --filelist FILELIST) --output OUTPUT --model MODEL --max_len MAX_LEN
    options:
      -h, --help           show this help message and exit
      --file FILE          Path to input file (if only processing one) (should be an annos.json)
      --filelist FILELIST  Path to a file containing a list of input files (annos.json)
      --output OUTPUT      Path to output directory
      --model MODEL        Model ID to use
      --max_len MAX_LEN    Max input tokens per request

=======================================================

NAME:
  monitor
DESCRIPTION:
  Monitor existing Batch API completion job, and save the output once complete.
USAGE:
  texner monitor [-h] job_id --output OUTPUT
    positional arguments:
      job_id           If provided, don't launch any jobs, but just monitor the provided batched job ID

    options:
      -h, --help       show this help message and exit
      --output OUTPUT  Path to output directory

=======================================================

NAME:
  postprocess
DESCRIPTION:
  Postprocess Batch API completion job, and save the output annotations
USAGE:
  texner postprocess [-h] --predictions PREDICTIONS (--file FILE | --filelist FILELIST) --output OUTPUT
    options:
      -h, --help                 show this help message and exit
      --file FILE                Path to input file (if only processing one)
      --filelist FILELIST        Path to a file containing a list of input files
      --predictions PREDICTIONS  Path to input file (if only processing one)
      --output OUTPUT            Path to output directory
"""

import json
import logging
import os
import tempfile
import time
import re
from icecream import ic
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import openai
import pandas as pd
import tiktoken
from more_itertools import chunked
from openai.types import Batch
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# System instruction
SYST = "You are an expert mathematician who is fluent in reading LaTeX and extracting information."

# Instruction to be prepended to every input
INST = """In the following LaTeX document, extract entities of the following types and return them in JSON output.
1. definition
2. theorem
3. proof
4. example
Your output should be a single JSON with 4 keys corresponding to the 4 entity types above. Spans may be part of multiple entities. Do not hallucinate text.

{}
"""

INST_NAME = """In the following LaTeX snippet, extract every newly defined named entity (name) and reference to previously defined named entity (reference). Your output should be in XML.

{}
"""

INST_PAGE_FILTER = """Determine whether the following page from a textbook contains mathematical content ('math'), metadata about a textbook ('metadata') such as a table of contents or index, or is missing ('missing').

# Page: {}
"""

# This is a response schema which forces a particular JSON output:
# https://platform.openai.com/docs/guides/structured-outputs#how-to-use
with open("./response_schema.json", "r") as f:
    schema = json.load(f)


def merge_neighbors(df: pd.DataFrame) -> pd.DataFrame:
    """Merges neighboring annotations which may have been split by e.g., chunking

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame of annotations. Columns should be `["start", "end", "tag", "text"]`.

    Returns
    -------
    pd.DataFrame
        DataFrame with merged annotations.
    """

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
        item = row.to_dict()
        item.update(dict(tag=row.tag, text=text, start=start, end=end))
        new.append(item)
        idx += 1

    return pd.DataFrame(new)


def load_anno_files(files: list[str]) -> dict[str, list]:
    """Tokenizes files and returns dictionary of name --> examples

    Parameters
    ----------
    files : list[str]
        Input files to tokenize

    Returns
    -------
    dict[str, dict]
        Mapping of name to list of examples to process
    """
    mmds = {}
    for fname in files:
        df = pd.read_json(fname)
        mmds[Path(fname).name] = df[df.tag.isin({"theorem", "definition"})].to_dict(orient="records")
    return mmds


def load_and_tokenize_mmd_files(files: list[str]) -> dict[str, tuple[str, list[int]]]:
    """Tokenizes files and returns dictionary of name --> (contents, tokens)

    Parameters
    ----------
    files : list[str]
        Input files to tokenize

    Returns
    -------
    dict[str, tuple[str, list[int]]]
        Mapping of name to (content, tokens)
    """
    mmds = {}
    for fname in files:
        with open(fname, "r") as f:
            mmd = f.read()
            # Remove the missing page indicators
            pattern = r"^\[MISSING_PAGE_.*$"
            mmd = re.sub(pattern, "\n\n", mmd, flags=re.MULTILINE)
            tokens = tokenizer.encode(mmd)
            mmds[Path(fname).name] = (mmd, tokens)
    return mmds


def format_input(input_text: str, instruction: str, system: str) -> list[ChatCompletionMessageParam]:
    """Formats input into messages to feed to GPT

    Parameters
    ----------
    input_text : str
        Input text to process
    instruction : str
        Instruction to prepend `input_text`
    system : str
        System instruction

    Returns
    -------
    list[ChatCompletionMessageParam]
        `system` and `user` messages (no `assistant`)
    """

    return [
        ChatCompletionSystemMessageParam(role="system", content=system),
        ChatCompletionUserMessageParam(role="user", content=instruction.format(input_text)),
    ]


def format_batch_request(tokens: list, model: str, max_len: int, file_id: str) -> list[dict[str, str]]:
    """Create batch request which processes `tokens` in chunks of `max_len`

    Parameters
    ----------
    tokens : list
        List of tokens to process
    model : str
        Model ID to use (e.g., a fine-tuned model or `gpt-4o-mini`)
    max_len : int
        Max input tokens to process per request
    file_id : str
        Custom ID prefix for request. Each request will have an id in the format
        `{file_id}.{start}-{end}` where start/end are the indices of a chunk to process.

    Examples
    --------
    >>> tokenizer = tiktoken.encoding_for_model("gpt-4o-mini-2024-07-18")
    >>> with open('math_doc_example.mmd', 'r') as f: text = f.read()
    >>> tokens = tokenizer.encode(text)
    >>> requests = format_batch_request(tokens, 'gpt-4o-mini', 1024, "math_doc_example")
    """

    requests = []
    for idx, snippet in enumerate(chunked(tokens, n=max_len)):
        snippet_text = tokenizer.decode(snippet)
        messages = format_input(snippet_text, INST, SYST)
        body = dict(
            model=model,
            messages=messages,
            response_format=schema,
        )
        requests.append(
            dict(
                custom_id=f"{file_id}.{idx*max_len}-{(idx+1)*max_len}",
                method="POST",
                url="/v1/chat/completions",
                body=body,
            )
        )
    return requests


def format_add_names_request(examples: list[dict], model: str, max_len: int, file_id: str) -> list[dict[str, Any]]:
    """Create batch request which processes `tokens` in chunks of `max_len`

    Parameters
    ----------
    tokens : list
        List of tokens to process
    model : str
        Model ID to use (e.g., a fine-tuned model or `gpt-4o-mini`)
    max_len : int
        Max input tokens to process per request
    file_id : str
        Custom ID prefix for request. Each request will have an id in the format
        `{file_id}.{start}-{end}` where start/end are the indices of a chunk to process.

    Examples
    --------
    >>> example = "Definition 4.1.2: An abelian group is a group with a commutative binary operation."
    >>> requests = format_add_names_request([example], 'gpt-4o-mini', 1024, "math_doc_example")
    """

    requests = []
    added = {}
    for example in examples:
        text = example["text"]
        tag = example["tag"]
        start = example["start"]
        end = example["end"]
        if (start, end) in added:
            continue
        added[(start, end)] = text, tag
        if len(text) < 40:
            continue
        trunc = tokenizer.decode(tokenizer.encode(text)[:max_len])
        messages = format_input(trunc, INST_NAME, SYST)
        body = dict(
            model=model,
            messages=messages,
        )
        requests.append(
            dict(
                custom_id=f"{file_id}.{start}-{end}",
                method="POST",
                url="/v1/chat/completions",
                body=body,
            )
        )
    return requests


def format_page_filter_request(
    tokens: list, model: str, max_len: int, file_id: str, first_k_tokens: int | None = None
) -> list[dict[str, str]]:
    """Create batch request to filter pages which processes `tokens` in chunks of `max_len`

    Parameters
    ----------
    tokens : list
        List of tokens to process
    model : str
        Model ID to use (e.g., a fine-tuned model or `gpt-4o-mini`)
    max_len : int
        Max input tokens to process per request
    file_id : str
        Custom ID prefix for request. Each request will have an id in the format
        `{file_id}.{start}-{end}` where start/end are the indices of a chunk to process.
    first_k_tokens : int
        Only process the first k tokens in the input MMD file. If None, use all.

    Examples
    --------
    >>> tokenizer = tiktoken.encoding_for_model("gpt-4o-mini-2024-07-18")
    >>> with open('math_doc_example.mmd', 'r') as f: text = f.read()
    >>> tokens = tokenizer.encode(text)
    >>> requests = format_batch_request(tokens, 'gpt-4o-mini', 1024, "math_doc_example")
    """

    requests = []
    stop_idx = first_k_tokens or len(tokens)
    for idx, snippet in enumerate(chunked(tokens[:stop_idx], n=max_len)):
        snippet_text = tokenizer.decode(snippet)
        messages = format_input(snippet_text, INST_PAGE_FILTER, SYST)
        body = dict(
            model=model,
            messages=messages,
        )
        requests.append(
            dict(
                custom_id=f"{file_id}.{idx*max_len}-{(idx+1)*max_len}",
                method="POST",
                url="/v1/chat/completions",
                body=body,
            )
        )
    return requests


def batch_process(mmds: dict[str, tuple[str, list[int]]], model: str, max_len: int, output_dir: Path) -> pd.DataFrame:
    """Launches a job to process `requests` via the Batch API

    Parameters
    ----------
    mmds : dict[str, tuple[str, list[int]]]
        Dict from name to (contents, tokens)
    model : str
        Model ID to use (e.g., a fine-tuned model or `gpt-4o-mini`)
    max_len : int
        Max input tokens per request
    output_dir : Path
        Path to save output directory

    Returns
    -------
    pd.DataFrame
        Processed output (see OpenAI batch API response docs)

    Examples
    --------
    >>> mmds = load_and_tokenize_mmd_files(input_files)
    >>> df = batch_process(mmds, model='gpt-4o-mini', max_len=1024, output_dir=Path('./outputs'))
    """

    requests = []
    for name, (mmd, tokens) in mmds.items():
        reqs = format_batch_request(tokens=tokens, model=model, max_len=max_len, file_id=name)
        requests.extend(reqs)

    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl") as nt:
        pd.DataFrame.from_records(requests).to_json(nt.name, orient="records", lines=True)
        batch_input_file = client.files.create(file=Path(nt.name), purpose="batch")
    job_info = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "Running GPT classification on textbooks"},
    )
    return monitor(job_info.id, output_dir)


def batch_process_add_names(annos: dict[str, list], model: str, max_len: int, output_dir: Path) -> pd.DataFrame:
    """Launches a job to process `requests` via the Batch API

    Parameters
    ----------
    annos : dict[str, list]
        Dict from name to example records
    model : str
        Model ID to use (e.g., a fine-tuned model or `gpt-4o-mini`)
    max_len : int
        Max input tokens per request
    output_dir : Path
        Path to save output directory

    Returns
    -------
    pd.DataFrame
        Processed output (see OpenAI batch API response docs)

    Examples
    --------
    >>> annos = load_anno_files(input_files)
    >>> df = batch_process_add_names(annos, model='gpt-4o-mini', max_len=1024, output_dir=Path('./outputs'))
    """

    requests = []
    for name, examples in annos.items():
        reqs = format_add_names_request(examples, model=model, max_len=max_len, file_id=name)
        requests.extend(reqs)

    for chunk in chunked(requests, n=50_000):
        with tempfile.NamedTemporaryFile("w+", suffix=".jsonl") as nt:
            pd.DataFrame.from_records(chunk).to_json(nt.name, orient="records", lines=True)
            batch_input_file = client.files.create(file=Path(nt.name), purpose="batch")
        job_info = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "Adding names to previously classified objects"},
        )
    return monitor(job_info.id, output_dir)  # type:ignore


def batch_process_page_filter(
    mmds: dict[str, tuple[str, list[int]]], model: str, max_len: int, output_dir: Path, first_k_tokens: int = 24_000
) -> pd.DataFrame:
    """Launches a job to filter out chunks of `max_len`

    Parameters
    ----------
    mmds : dict[str, tuple[str, list[int]]]
        Dict from name to (contents, tokens)
    model : str
        Model ID to use (e.g., a fine-tuned model or `gpt-4o-mini`)
    max_len : int
        Max input tokens per request
    output_dir : Path
        Path to save output directory
    first_k_tokens : int = 24_000
        Only process the first k tokens in the input MMD file. If None, use all.

    Returns
    -------
    pd.DataFrame
        Processed output (see OpenAI batch API response docs)

    Examples
    --------
    >>> mmds = load_and_tokenize_mmd_files(input_files)
    >>> df = batch_process_page_filter(mmds, model='gpt-4o-mini', max_len=1024, output_dir=Path('./outputs'))
    """

    requests = []
    for name, (mmd, tokens) in mmds.items():
        reqs = format_page_filter_request(
            tokens=tokens, model=model, max_len=max_len, file_id=name, first_k_tokens=first_k_tokens
        )
        requests.extend(reqs)

    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl") as nt:
        pd.DataFrame.from_records(requests).to_json(nt.name, orient="records", lines=True)
        batch_input_file = client.files.create(file=Path(nt.name), purpose="batch")
    job_info = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "Running GPT page filtering on textbooks"},
    )
    return monitor(job_info.id, output_dir)


def process_file(tokens: list, model: str, max_len: int) -> pd.DataFrame:
    """Process a single file

    Parameters
    ----------
    tokens : list
        List of tokens to process.
    model : str
        Model ID to use (e.g., a fine-tuned model or `gpt-4o-mini`)
    max_len : int
        Max input tokens to process per request

    Returns
    -------
    pd.DataFrame
        Returns post-processed list of annotations

    Examples
    --------
    >>> tokens = tokenizer.encode(text)
    >>> df = batch_process(tokens, model='gpt-4o-mini', max_len=1024)
    """
    annos = []
    n_chunks = len(tokens) // max_len
    logger.info(f"Processing {n_chunks} chunks...")

    # Loop through each chunk of tokens
    for i, snippet in tqdm(enumerate(chunked(tokens, n=max_len)), total=n_chunks):

        # Format the input with instruction and system prompt
        snippet_text = tokenizer.decode(snippet)
        messages = format_input(snippet_text, INST, SYST)
        # Launch the completion request
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=schema,
        )
        # Process the output
        output = completion.choices[0].message.content
        assert output is not None
        pred = json.loads(output)

        # Hunt for the output annotation in the input snippet, and add the chunk offset to get the real start index
        for tag, texts in pred.items():
            for t in texts:
                start = snippet_text.find(t) + (i * max_len)
                end = start + len(t)
                if start == -1:
                    end = -1
                annos.append(dict(tag=tag, text=t, start=start, end=end))

    return pd.DataFrame.from_records(annos)


def process(mmds: dict[str, tuple[str, list[int]]], model: str, max_len: int, output_dir: Path):
    """Launches a job to process `requests` via real-time completion API

    Parameters
    ----------
    mmds : dict[str, tuple[str, list[int]]]
        Dict from name to (contents, tokens)
    model : str
        Model ID to use (e.g., a fine-tuned model or `gpt-4o-mini`)
    max_len : int
        Max input tokens per request
    output_dir : Path
        Path to save output directory

    Returns
    -------
    pd.DataFrame
        Postprocessed annotations (merged + cleaned)

    Examples
    --------
    >>> mmds = load_and_tokenize_mmd_files(input_files)
    >>> df = process(mmds, model='gpt-4o-mini', max_len=1024, output_dir=Path('./outputs'))
    """
    for name, (mmd, tokens) in mmds.items():
        df = process_file(tokens, model=model, max_len=max_len)
        logger.info(f"Finished generating for {name}, got {len(df)} annotations.")

        df = df[df["start"] != -1]
        logger.info(f"Dropped rows whose start location went unfound, left with {len(df)} annotations.")

        df = merge_neighbors(df.sort_values(["tag", "start"]))  # type:ignore
        logger.info(f"Merged neighbors, left with {len(df)} annotations.")

        df = df[df["text"].apply(len) <= 40]
        logger.info(f"Dropped items which were too short (<40 char)")

        outfile = Path(output_dir, Path(name).with_suffix(".annos.json").name)
        df.to_json(outfile)
        logger.info(f"Saved to {outfile}")


def monitor(job_id: str, output_dir: Path) -> pd.DataFrame:
    """Monitor existing Batch API job and save output when complete

    Parameters
    ----------
    job_id : str
        ID of job to monitor
    output_dir : Path
        Directory to save output directory (in jsonl)

    Returns
    -------
    pd.DataFrame
        DataFrame of downloaded annotations
    """

    job_info = client.batches.retrieve(job_id)
    job_status = job_info.status

    # Wait for the job to actually start, so we can get an accurate total
    logger.info("Waiting for batch to start...")
    while job_status not in {"in_progress", "cancelled", "failed", "expired", "completed", "finalizing"}:
        time.sleep(5)
        job_info = client.batches.retrieve(job_id)
        job_status = job_info.status
    if job_status in {"cancelled", "failed"}:
        logger.error(f"Failed job! Check dashboard for information on {job_id}")
        exit(1)
    logger.info("Starting!")

    # Wait until the job is complete, and update the progress bar to match
    pbar = tqdm(range(job_info.request_counts.total), total=job_info.request_counts.total)
    while job_status not in ["completed", "expired", "cancelled", "failed"]:
        job_info = client.batches.retrieve(job_id)
        job_status = job_info.status
        pbar.n = job_info.request_counts.completed
        pbar.refresh()
        pbar.set_description(f"Status: {job_status}")
        time.sleep(5)

    logger.info("Completed job!")

    # Download the output file
    output_contents = client.files.content(job_info.output_file_id).text
    records = []
    for line in output_contents.splitlines():
        records.append(json.loads(line))
    df = pd.DataFrame.from_records(records)

    # Save the output file to output_dir as a jsonl
    outfile = Path(output_dir, job_info.output_file_id).with_suffix(".jsonl")
    df.to_json(outfile, lines=True, orient="records")
    logger.info(f"Saved to {outfile}")

    return df


def extract_annos_from_xml(xml):
    pattern = r"<name>(?P<name>.*?)</name>|<reference>(?P<reference>.*?)</reference>"
    offset = 0
    # [(start, end, item), ...]
    tags = []
    for item in re.finditer(pattern, xml):
        if item.group("name") is not None:
            start = item.start() - offset
            end = start + len(item.group("name"))
            entity = item.group("name")
            tag = "name"
            offset += (item.end() - item.start()) - len(item.group("name"))
            tags.append(dict(start=start, end=end, tag=tag, text=entity))
        elif item.group("reference") is not None:
            start = item.start() - offset
            end = start + len(item.group("reference"))
            entity = item.group("reference")
            tag = "reference"
            offset += (item.end() - item.start()) - len(item.group("reference"))
            tags.append(dict(start=start, end=end, tag=tag, text=entity))
    return tags


def postprocess_names(path: Path | str, mmds: dict[str, list[dict]], output_dir: Path | str):
    # Read in the predicted annotations
    df = pd.read_json(path, lines=True)
    logger.info(f"Read {len(df)} predictions from {path}")

    # Process the request object to extract predictiosn
    df["preds"] = df.response.apply(lambda x: x["body"]["choices"][0]["message"]["content"]).apply(
        extract_annos_from_xml
    )
    df = df.dropna(subset=["preds"])
    file_id_and_range = df.custom_id.str.split(r"\.annos.json\.")
    df["file_id"] = file_id_and_range.apply(lambda x: Path(x[0]).with_suffix(".annos.json").name)
    df["parent_start"] = file_id_and_range.apply(lambda x: int(x[1].split("-")[0]))
    df["parent_end"] = file_id_and_range.apply(lambda x: int(x[1].split("-")[1]))
    logger.info(f"Split Custom IDs into file ID and range")

    # Hunt for the annotations in the actual text so we can get their start/end indices
    annos = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        file_id = row.file_id

        # For this item, find the associated parent annotation
        parent_annos = pd.DataFrame.from_records(mmds[file_id])
        parent = parent_annos[
            (parent_annos["start"] == row.parent_start) & (parent_annos["end"] == row.parent_end)
        ].iloc[0]

        # Now re-add the parent start so we get the real starting/ending indices
        names_and_refs = row.preds
        for pred in names_and_refs:
            pred["start"] += parent.start
            pred["end"] += parent.start
            pred["file_id"] = file_id
            pred["parent_start"] = parent.start
            pred["parent_end"] = parent.end
            pred["parent_text"] = parent.text
            pred["parent_tag"] = parent.tag
            annos.append(pred)

    anno_df = pd.DataFrame.from_records(annos)

    for file_id in anno_df.file_id.unique():
        file_annos = anno_df[anno_df["file_id"] == file_id]
        logger.info(f"For {file_id} we have {len(file_annos)} annotations")

        outfile = Path(output_dir, Path(file_id.replace(".annos.json", "")).with_suffix(".annos.json").name)
        file_annos.to_json(outfile)  # type:ignore
        logger.info(f"Saved {len(file_annos)} annotations to {outfile}")


def postprocess(path: Path | str, mmds: dict[str, tuple[str, list[int]]], output_dir: Path | str):
    def tryloads(j):
        try:
            return json.loads(j)
        except json.JSONDecodeError as e:
            return None

    # Read in the predicted annotations
    df = pd.read_json(path, lines=True)
    logger.info(f"Read {len(df)} predictions from {path}")

    # Process the request object to extract predictiosn
    df["preds"] = df.response.apply(lambda x: x["body"]["choices"][0]["message"]["content"]).apply(tryloads)
    df = df.dropna(subset=["preds"])
    file_id_and_range = df.custom_id.str.split(r"\.mmd\.")
    df["file_id"] = file_id_and_range.apply(lambda x: x[0] + ".mmd")
    df["file_start"] = file_id_and_range.apply(lambda x: int(x[1].split("-")[0]))
    df["file_end"] = file_id_and_range.apply(lambda x: int(x[1].split("-")[1]))
    logger.info(f"Split Custom IDs into file ID and range")

    # Hunt for the annotations in the actual text so we can get their start/end indices
    annos = []
    for i, row in df.iterrows():
        file_id = row.file_id
        # fs = row.file_start
        # fe = row.file_end

        # Grab the content and the relevant tokens, then detokenize
        content, tokens = mmds[file_id]
        # snippet_text = tokenizer.decode(tokens[fs : fe + 1])
        # file_offset = len(tokenizer.decode(tokens[:fs]))
        # Hunt for the output annotation in the input snippet, and add the chunk offset to get the real start index
        for tag, texts in row.preds.items():
            for t in texts:
                # start = snippet_text.find(t) + file_offset
                start = content.find(t)
                end = start + len(t)
                if start == -1:
                    end = -1
                annos.append(dict(file_id=file_id, tag=tag, text=t, start=start, end=end))

    anno_df = pd.DataFrame.from_records(annos)
    anno_df = anno_df[anno_df["start"] != -1]
    logger.info(f"Dropped rows whose start location went unfound, left with {len(anno_df)} annotations.")

    for file_id in anno_df.file_id.unique():
        file_annos = anno_df[anno_df["file_id"] == file_id]
        logger.info(f"For {file_id} we have {len(file_annos)} annotations")

        file_annos = merge_neighbors(file_annos.sort_values(["tag", "start"]))  # type:ignore
        logger.info(f"Merged neighbors, left with {len(file_annos)}")

        file_annos = file_annos[file_annos["text"].apply(len) >= 40]
        logger.info(f"Dropped items which were too short (<40 char), left with {len(file_annos)}")

        outfile = Path(output_dir, Path(file_id).with_suffix(".annos.json").name)
        file_annos.to_json(outfile)
        logger.info(f"Saved {len(file_annos)} annotations to {outfile}")


if __name__ == "__main__":

    # fmt:off
    parser = ArgumentParser("texner")
    parser.add_argument( '--log_level',default='warning', help='Provide logging level. Example --loglevel debug, default=warning' )

    subparsers = parser.add_subparsers(help='command', dest='command', required=True)

    # Launch job and get results in real-time
    process_parser = subparsers.add_parser('process')
    process_input = process_parser.add_mutually_exclusive_group(required=True)
    process_input.add_argument("--file", help="Path to input file (if only processing one)")
    process_input.add_argument("--filelist", help="Path to a file containing a list of input files")

    process_parser.add_argument("--output", required=True, help="Path to output directory")
    process_parser.add_argument("--model", type=str, required=True, help="Model ID to use")
    process_parser.add_argument("--max_len", type=int, required=True, help="Max input tokens per request")

    # Launch job via Batch API
    batch_parser = subparsers.add_parser('batch')
    batch_input = batch_parser.add_mutually_exclusive_group(required=True)
    batch_input.add_argument("--file", help="Path to input file (if only processing one)")
    batch_input.add_argument("--filelist", help="Path to a file containing a list of input files")

    batch_parser.add_argument("--output", required=True, help="Path to output directory")
    batch_parser.add_argument("--model", type=str, required=True, help="Model ID to use")
    batch_parser.add_argument("--max_len", type=int, required=True, help="Max input tokens per request")

    # Add names to existing predictions
    add_names_parser = subparsers.add_parser('add_names')
    add_names_input = add_names_parser.add_mutually_exclusive_group(required=True)
    add_names_input.add_argument("--file", help="Path to input .annos.json file (if only processing one)")
    add_names_input.add_argument("--filelist", help="Path to a file containing a list of input .annos.json files")

    add_names_parser.add_argument("--output", required=True, help="Path to output directory")
    add_names_parser.add_argument("--model", type=str, required=True, help="Model ID to use")
    add_names_parser.add_argument("--max_len", type=int, required=True, help="Max input tokens per request")

    # Filter out pages which we shouldn't be annotating
    page_filter_parser = subparsers.add_parser('page_filter')
    page_filter_input = page_filter_parser.add_mutually_exclusive_group(required=True)
    page_filter_input.add_argument("--file", help="Path to input .annos.json file (if only processing one)")
    page_filter_input.add_argument("--filelist", help="Path to a file containing a list of input files")

    page_filter_parser.add_argument("--output", required=True, help="Path to output directory")
    page_filter_parser.add_argument("--model", type=str, required=True, help="Model ID to use")
    page_filter_parser.add_argument("--max_len", type=int, required=True, help="Max input tokens per request")
    page_filter_parser.add_argument("--first_k_tokens", type=int, required=True, help="Only process the first K tokens")

    # Monitor job
    monitor_parser = subparsers.add_parser('monitor')
    monitor_parser.add_argument("job_id", type=str, help="If provided, don't launch any jobs, but just monitor the provided batched job ID")
    monitor_parser.add_argument("--output", required=True, help="Path to output directory")

    # Postprocess job
    postprocess_parser = subparsers.add_parser('postprocess')
    postprocess_input = postprocess_parser.add_mutually_exclusive_group(required=True)
    postprocess_input.add_argument("--file", help="Path to input file (if only processing one)")
    postprocess_input.add_argument("--filelist", help="Path to a file containing a list of input files")

    postprocess_parser.add_argument("--type", choices=['nameref', 'base'], type=str, help='Whether to postprocess base predictions of name/ref predictions.')
    postprocess_parser.add_argument("--predictions", required=True, help="Path to predictions file")
    postprocess_parser.add_argument("--output", required=True, help="Path to output directory")
    # fmt:on

    args = parser.parse_args()

    client = openai.OpenAI()
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini-2024-07-18")
    logging.basicConfig(level=args.log_level.upper())
    logger = logging.getLogger(__name__)

    # Enable logging to proper level
    logger.setLevel(logging.INFO)

    match args.command:
        case "process" | "batch":
            # Get the input file name(s) and tokenize them
            files = [l.strip() for l in open(args.filelist, "r").readlines()] if args.filelist else [args.file]
            mmds = load_and_tokenize_mmd_files(files)

            # Get an estimate total price and confirm it's ok. This is (for my case) an
            # overestimate, but better to be safe.
            total_price = 0.0
            for name, (mmd, tokens) in mmds.items():
                est_in = (len(tokens) / 1_000_000) * 0.150
                est_out = (len(tokens) / 1_000_000) * 0.600
                logger.info(f"Estimated price for {name}: ${est_in + est_out:0.2f}")
                total_price += est_in + est_out

            answer = input(f"Total estimated price: ${total_price:0.2f}. Is this ok? [y/N]")
            if answer not in {"y", "Y"}:
                exit(0)

            # Ensure the output directory exists, and then process the inputs in the chosen way
            Path(args.output).parent.mkdir(exist_ok=True, parents=True)
            if args.command == "batch":
                outputs = batch_process(mmds, model=args.model, max_len=args.max_len, output_dir=args.output)
            else:  # args.command == "process":
                outputs = process(mmds, model=args.model, max_len=args.max_len, output_dir=args.output)
        case "page_filter":
            # Get the input file name(s) and tokenize them
            files = [l.strip() for l in open(args.filelist, "r").readlines()] if args.filelist else [args.file]
            mmds = load_and_tokenize_mmd_files(files)
            Path(args.output).parent.mkdir(exist_ok=True, parents=True)
            outputs = batch_process_page_filter(
                mmds, model=args.model, max_len=args.max_len, output_dir=args.output, first_k_tokens=args.first_k_tokens
            )
        case "add_names":
            files = [l.strip() for l in open(args.filelist, "r").readlines()] if args.filelist else [args.file]
            annos = load_anno_files(files)
            Path(args.output).parent.mkdir(exist_ok=True, parents=True)
            outputs = batch_process_add_names(annos, model=args.model, max_len=args.max_len, output_dir=args.output)
        case "monitor":
            monitor(args.job_id, args.output)
        case "postprocess":
            # Get the input file name(s) and tokenize them
            files = [l.strip() for l in open(args.filelist, "r").readlines()] if args.filelist else [args.file]
            if args.type == "base":
                mmds = {}
                for fname in files:
                    with open(fname, "r") as f:
                        mmd = f.read()
                        mmds[Path(fname).name] = (mmd, None)

                # Ensure the output directory exists
                Path(args.output).parent.mkdir(exist_ok=True, parents=True)
                postprocess(args.predictions, mmds, args.output)
            else:
                annos = load_anno_files(files)
                # Ensure the output directory exists
                Path(args.output).parent.mkdir(exist_ok=True, parents=True)
                postprocess_names(args.predictions, annos, args.output)
