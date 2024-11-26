import pandas as pd

from sklearn.model_selection import train_test_split
from openai import OpenAI
import json
import tiktoken # for token counting
import numpy as np
from pathlib import Path
from pprint import pprint
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
from argparse import ArgumentParser

client = OpenAI()

def validate_data(dataset):
# Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
            
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
            
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
        
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
        return False
    else:
        print("No errors found")
        return True

def format_filter_prompt(file_content, label=None):
    system = "You are an expert mathematician fluent in parsing LaTeX and markdown."
    instruction = "Determine whether the following page from a textbook contains mathematical content ('math'), metadata about a textbook ('metadata') such as a table of contents or index, or is missing ('missing')."
    user = f"""{instruction}\n\n{file_content}""" 
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    if label:
        assistant = label.lower()
        messages.append({'role': 'assistant', 'content': assistant})
    return {'messages': messages}

parser = ArgumentParser('pagefilter')
parser.add_argument('cmd', choices=['train', 'test'])
parser.add_argument('--input_file', required=True, type=str)
parser.add_argument('--page_dir', required=True, type=str)
parser.add_argument('--output_file', required=False, default='./page_filter_preds.json', type=str)
parser.add_argument('--model', required=False, default="gpt-4o-mini", type=str)

args = parser.parse_args()

# Read/split data for training
# ============================
df = pd.read_csv(args.input_file)
df = df.drop_duplicates(subset=["File ID"]) # type:ignore

# Rename the labels
df['Annotation'] = df['Annotation'].apply(lambda label: 'math' if label == 'good' else 'metadata' if label == 'bad' else 'missing')
train, test = train_test_split(df, test_size=0.25, stratify=df['Annotation'])

# Format data into messages
# =========================
data_dir = Path(args.page_dir)
train_prompts = []
for i, row in train.iterrows():
    with open(data_dir / row['File ID'], 'r') as f:
        contents = f.read()
    train_prompts.append(format_filter_prompt(contents, row.Annotation))

test_prompts = []
for i, row in test.iterrows():
    with open(data_dir / row['File ID'], 'r') as f:
        contents = f.read()
    test_prompts.append(format_filter_prompt(contents, row.Annotation))

if not validate_data(train_prompts):
    exit(1)
if not validate_data(test_prompts):
    exit(1)


if args.cmd == 'train':
# Upload data to Open AI
# ======================
    train_path = "./data/openai/page-filter.train.jsonl"
    test_path = "./data/openai/page-filter.test.jsonl"
    pd.DataFrame.from_records(train_prompts).to_json(train_path, lines=True, orient='records')
    pd.DataFrame.from_records(test_prompts).to_json(test_path, lines=True, orient='records')

    train_file = client.files.create(file=open(train_path, "rb"), purpose="fine-tune")
    test_file = client.files.create(file=open(test_path, "rb"), purpose="fine-tune")

    job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        model="gpt-4o-mini-2024-07-18",
        validation_file=test_file.id,
        seed=1299874447,
    )
    print(job)

if args.cmd == 'test':
    records = []
    for msgs in tqdm(test_prompts):
        system, user, assistant = msgs['messages']
        completion = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:uc-santa-cruz-jlab-nlp::AXam829S",
            messages=[system, user],
            temperature=0.0,
        )
        pred = completion.choices[0].message.content
        if pred in {'bad', 'good', 'nougat fail'}:
            pred = 'math' if pred == 'good' else 'metadata' if pred == 'bad' else 'missing'
        record = dict(page=user['content'].split('# Page:')[1], label=assistant['content'], prediction=pred)
        records.append(record)

    output_df = pd.DataFrame.from_records(records)
    print('accuracy: ', accuracy_score(output_df['label'], output_df['prediction']))

    print(classification_report(y_true=output_df['label'], y_pred=output_df['prediction']))

    output_df.to_json(args.output_file)


