import torch
from datasets import load_from_disk, Dataset
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125")

def prompt_formation_and_tokenize_rosetta(example):
    prompt = f"### Task description:\n{example['task_description']}\n"
    prompt += f"### Language name:\n{example['language_name']}\n"
    prompt += f"### Code:\n"

    prompt_tokens=tokenizer(prompt)
    response_tokens=tokenizer(example['code'])
    eos_token_id = tokenizer.eos_token_id

    input_ids = prompt_tokens['input_ids'] + response_tokens['input_ids'] + [eos_token_id]
    attention_mask = prompt_tokens['attention_mask'] + response_tokens['attention_mask'] + [1]
    labels=[-100]*len(prompt_tokens['input_ids']) + response_tokens['input_ids'] + [eos_token_id]

    max_length = 2048
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def prompt_formation_and_tokenize_meta_math(example):
    prompt = f"### Query:\n{example['query']}\n"
    prompt += f"### Response:\n"

    prompt_tokens=tokenizer(prompt)
    response_tokens=tokenizer(example['response'])
    eos_token_id = tokenizer.eos_token_id

    input_ids = prompt_tokens['input_ids'] + response_tokens['input_ids'] + [eos_token_id]
    attention_mask = prompt_tokens['attention_mask'] + response_tokens['attention_mask'] + [1]
    labels=[-100]*len(prompt_tokens['input_ids']) + response_tokens['input_ids'] + [eos_token_id]

    max_length = 2048
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def prompt_formation_and_tokenize_dolly15k(example):
    prompt = f"### Instruction:\n{example['instruction']}\n"
    prompt += f"### Context:\n{example['context']}\n"
    prompt += f"### Response:\n"

    prompt_tokens=tokenizer(prompt)
    response_tokens=tokenizer(example['response'])
    eos_token_id = tokenizer.eos_token_id

    input_ids = prompt_tokens['input_ids'] + response_tokens['input_ids'] + [eos_token_id]
    attention_mask = prompt_tokens['attention_mask'] + response_tokens['attention_mask'] + [1]
    labels=[-100]*len(prompt_tokens['input_ids']) + response_tokens['input_ids'] + [eos_token_id]

    max_length = 2048
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def prompt_formation_and_tokenize_code(example):
    prompt = f"### Instruction:\n{example['instruction']}\n"
    prompt += f"### Output:\n"

    prompt_tokens=tokenizer(prompt)
    response_tokens=tokenizer(example['output'])
    eos_token_id = tokenizer.eos_token_id

    input_ids = prompt_tokens['input_ids'] + response_tokens['input_ids'] + [eos_token_id]
    attention_mask = prompt_tokens['attention_mask'] + response_tokens['attention_mask'] + [1]
    labels=[-100]*len(prompt_tokens['input_ids']) + response_tokens['input_ids'] + [eos_token_id]

    max_length = 2048
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def prompt_formation_and_tokenize_gsm8k(batch):
    prompts = []
    for query in batch["question"]:
        prompt = f"### Query:\n{query}\n"
        prompt += f"### Response:\n"
        prompts.append(prompt)

    # Tokenize all prompts together → returns dict of tensors
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )
    return tokenized