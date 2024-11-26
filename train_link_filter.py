import pandas as pd

from sklearn.model_selection import train_test_split
from openai import OpenAI
import json
import tiktoken # for token counting
import numpy as np
from pprint import pprint
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm

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

def format_filter_prompt(src_file, tgt_file, src_context, tgt_context, src_text, tgt_text, label=None):
    system = "You are an expert mathematician with an eye for detail."
    instruction = "Determine whether the following reference is referring to the given named enti ty or not. Be mathematically precise. Ensure the usage of the term is the same in both the refere nce context and the named entity. If the reference is correct, respond 'good'. Otherwise, respond 'bad'."
    user = f"""{instruction}\n\n# Reference: {src_text}\n# Reference Context: {src_context}\n\n# Entity Name: {tgt_text}\n# Entity Text: {tgt_context}""" 
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    if label:
        assistant = label.lower()
        messages.append({'role': 'assistant', 'content': assistant})
    return {'messages': messages}

# Read/split data for training
# ============================
df = pd.read_csv('./cross_document_links_annotated.csv')
df = df[df.Annotation.isin(('Good', 'Bad'))]
df = df.drop_duplicates(['Source Start', 'Source End', 'Target Start', 'Target End']) # type:ignore
train, test = train_test_split(df, test_size=0.25, stratify=df['Annotation'])

# Format data into messages
# =========================
train_prompts = []
for i, row in train.iterrows():
    train_prompts.append(format_filter_prompt(row['Source File ID'], row['Target File ID'], row['Source Context'], row['Target Context'], row['Source Text'], row['Target Text'], row['Annotation']))

test_prompts = []
for i, row in test.iterrows():
    test_prompts.append(format_filter_prompt(row['Source File ID'], row['Target File ID'], row['Source Context'], row['Target Context'], row['Source Text'], row['Target Text'], row['Annotation']))

if not validate_data(train_prompts):
    exit(1)
if not validate_data(test_prompts):
    exit(1)

# Upload data to Open AI
# ======================
train_path = "./data/openai/link-filter.train.jsonl"
test_path = "./data/openai/link-filter.test.jsonl"
pd.DataFrame.from_records(train_prompts).to_json(train_path, lines=True, orient='records')
pd.DataFrame.from_records(test_prompts).to_json(test_path, lines=True, orient='records')

# train_file = client.files.create(file=open(train_path, "rb"), purpose="fine-tune")
# test_file = client.files.create(file=open(test_path, "rb"), purpose="fine-tune")
# 
# job = client.fine_tuning.jobs.create(
#   training_file=train_file.id,
#   model="gpt-4o-mini-2024-07-18",
#   validation_file=test_file.id,
#   seed=1299874447,
# )
# print(job)

records = []
for i, row in tqdm(test.iterrows(), total=len(test)):
    msgs = format_filter_prompt(row['Source File ID'], row['Target File ID'], row['Source Context'], row['Target Context'], row['Source Text'], row['Target Text'], row['Annotation'])
    completion = client.chat.completions.create(
      model="ft:gpt-4o-mini-2024-07-18:uc-santa-cruz-jlab-nlp::AWUpupDE",
      messages=msgs['messages'][:-1],
      temperature=0.0,
    )
    pred = completion.choices[0].message.content
    record = dict(reference=row['Source Text'], context=row['Source Context'], name=row['Target Text'], definition=row['Target Context'], label=row['Annotation'].lower(), prediction=pred)
    records.append(record)

output_df = pd.DataFrame.from_records(records)
print('accuracy: ', accuracy_score(output_df['label'], output_df['prediction']))

(pg, pb), (rg, rb), (f1g, f1b), _ = precision_recall_fscore_support(output_df['label'], output_df['prediction'], average=None, labels=['good', 'bad'])

pprint(dict(good_precision=pg, good_recall=rg, good_f1=f1g, bad_precision=pb, bad_recall=rb, bad_f1=f1b))
output_df.to_json('./linker_predictions.json')


