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
      --output OUTPUT      Path to output file
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
      --output OUTPUT      Path to output file
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
      --output OUTPUT  Path to output file
"""

import json
import logging
import os
import tempfile
import time
from argparse import ArgumentParser
from pathlib import Path

import openai
from openai.types import Batch
import pandas as pd
import tiktoken
from more_itertools import chunked
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
        item = dict(tag=row.tag, text=text, start=start, end=end)
        new.append(item)
        idx += 1

    return pd.DataFrame(new)


def tokenize_files(files: list[str]) -> dict[str, tuple[str, list[int]]]:
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
            tokens = tokenizer.encode(mmd)
            mmds[fname] = (mmd, tokens)
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
        ChatCompletionUserMessageParam(role="user", content=instruction + "\n\n" + input_text),
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
        Path to save output file

    Returns
    -------
    pd.DataFrame
        Processed output (see OpenAI batch API response docs)

    Examples
    --------
    >>> mmds = tokenize_files(input_files)
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
        Path to save output file

    Returns
    -------
    pd.DataFrame
        Postprocessed annotations (merged + cleaned)

    Examples
    --------
    >>> mmds = tokenize_files(input_files)
    >>> df = process(mmds, model='gpt-4o-mini', max_len=1024, output_dir=Path('./outputs'))
    """
    for name, (mmd, tokens) in mmds.items():
        df = process_file(tokens, model=model, max_len=max_len)
        logger.info(f"Finished generating for {name}, got {len(df)} annotations.")

        df = df[df["start"] != -1]
        logger.info(f"Dropped rows whose start location went unfound, left with {len(df)} annotations.")

        df = merge_neighbors(df.sort_values(["tag", "start"]))  # type:ignore
        logger.info(f"Merged neighbors, left with {len(df)} annotations.")

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
        Directory to save output file (in jsonl)

    Returns
    -------
    pd.DataFrame
        DataFrame of downloaded annotations
    """

    job_info = client.batches.retrieve(job_id)
    job_status = job_info.status

    logger.info("Waiting for batch to start...")
    while job_status != "in_progress":
        time.sleep(5)
    logger.info("Starting!")

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


if __name__ == "__main__":

    # fmt:off
    parser = ArgumentParser("texner")
    parser.usage = __doc__
    subparsers = parser.add_subparsers(help='command', dest='command')

    # Launch job and get results in real-time
    process_parser = subparsers.add_parser('process')
    process_input = process_parser.add_mutually_exclusive_group(required=True)
    process_input.add_argument("--file", help="Path to input file (if only processing one)")
    process_input.add_argument("--filelist", help="Path to a file containing a list of input files")

    process_parser.add_argument("--output", required=True, help="Path to output file")
    process_parser.add_argument("--model", type=str, required=True, help="Model ID to use")
    process_parser.add_argument("--max_len", type=int, required=True, help="Max input tokens per request")

    # Launch job via Batch API
    batch_parser = subparsers.add_parser('batch')
    batch_input = batch_parser.add_mutually_exclusive_group(required=True)
    batch_input.add_argument("--file", help="Path to input file (if only processing one)")
    batch_input.add_argument("--filelist", help="Path to a file containing a list of input files")

    batch_parser.add_argument("--output", required=True, help="Path to output file")
    batch_parser.add_argument("--model", type=str, required=True, help="Model ID to use")
    batch_parser.add_argument("--max_len", type=int, required=True, help="Max input tokens per request")

    # Monitor job
    monitor_parser = subparsers.add_parser('monitor')
    monitor_parser.add_argument("job_id", type=str, help="If provided, don't launch any jobs, but just monitor the provided batched job ID")
    monitor_parser.add_argument("--output", required=True, help="Path to output file")
    # fmt:on

    args = parser.parse_args()

    client = openai.OpenAI()
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini-2024-07-18")
    logger = logging.getLogger(__name__)

    # Enable logging to proper level
    logger.setLevel(logging.INFO)

    match args.command:
        case "process" | "batch":
            # Get the input file name(s) and tokenize them
            files = [l.strip() for l in open(args.filelist, "r").readlines()] if args.filelist else [args.file]
            mmds = tokenize_files(files)

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
        case "monitor":
            monitor(args.job_id, args.output)
