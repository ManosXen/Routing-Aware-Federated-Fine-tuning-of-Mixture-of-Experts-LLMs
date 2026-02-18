import torch
from datasets import load_from_disk, Dataset
from transformers import DataCollatorForSeq2Seq, AutoTokenizer
import ast
import copy

# Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_and_mask_helper(instruction, response, tokenizer, max_length=2048):

    tokenized_response = tokenizer(
            response, 
            add_special_tokens=False, 
            truncation=True,
            max_length=256
        )
        
    instr_budget = max_length - len(tokenized_response['input_ids']) - 1
        
    tokenized_instruction = tokenizer(
            instruction, 
            add_special_tokens=True,  
            truncation=True,
            max_length=instr_budget
        )
    
    input_ids_instr = tokenized_instruction['input_ids']
    input_ids_resp = tokenized_response['input_ids']
    
    input_ids = input_ids_instr + input_ids_resp + [tokenizer.eos_token_id]
    
    attention_mask = [1] * len(input_ids)
    
    labels = ([-100] * len(input_ids_instr)) + input_ids_resp + [tokenizer.eos_token_id]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# ==========================================
# Dataset Specific Functions
# ==========================================

def prompt_formation_and_tokenize_nli(example):
    label_map = {0: "True", 1: "Neither", 2: "False"}
    
    hypothesis = example['hypothesis'].strip()
    if not hypothesis.endswith("."):
        hypothesis += "."
    
    instruction = (
        f"{example['premise']}\n"
        f"Question: {hypothesis} True, False or Neither?\n"
        f"Answer:"
    )
    
    response = " " + label_map[example['label']] # Add leading space for clean tokenization
    
    return tokenize_and_mask_helper(instruction, response, tokenizer)


def prompt_formation_and_tokenize_piqa(example):
    label_map = {0: example['sol1'], 1: example['sol2']}
    
    instruction = f"Question: {example['goal']}\nAnswer:"
    response = " " + label_map[example['label']]
    
    return tokenize_and_mask_helper(instruction, response, tokenizer)


def prompt_formation_and_tokenize_social_iqa(example):
    label = example['label']
    label_int = int(label)
    label_0indexed = label_int - 1
    
    answer_map = {0: example['answerA'], 1: example['answerB'], 2: example['answerC']}
    
    instruction = f"Q: {example['context']} {example['question']}\nA:"
    response = " " + answer_map[label_0indexed]
    
    return tokenize_and_mask_helper(instruction, response, tokenizer)


def prompt_formation_and_tokenize_winogrande(example):
    answer_to_num = {"1": 0, "2": 1}
    label_idx = answer_to_num[example["answer"]]
    
    sentence = example["sentence"]
    # Ensure underscore exists to avoid ValueError
    if "_" in sentence:
        idx = sentence.index("_")
        pre_underscore = sentence[:idx]
        post_underscore = sentence[idx + 1:].strip()
    else:
        # Fallback if data is malformed
        pre_underscore = sentence
        post_underscore = ""

    options = [example["option1"], example["option2"]]
    correct_option = options[label_idx]
    
    instruction = pre_underscore + correct_option
    response = " " + post_underscore
    
    return tokenize_and_mask_helper(instruction, response, tokenizer)


def prompt_formation_and_tokenize_commonsense_qa(example):
    question = example['question'].strip()
    choices = example['choices']['text']
    answer_key = example['answerKey']
    
    instruction = (
        f"Question: {question}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        f"E. {choices[4]}\n"
        f"Answer:"
    )
    
    response = " " + answer_key
    
    return tokenize_and_mask_helper(instruction, response, tokenizer)


def prompt_formation_and_tokenize_boolq(example):
    passage = example['passage']
    question = example['question']
    label = example['answer']  # 0 or 1
    
    instruction = f"{passage}\nQuestion: {question}?\nAnswer:"
    
    label_map = {0: "no", 1: "yes"}
    response = " " + label_map[label]
    
    return tokenize_and_mask_helper(instruction, response, tokenizer)


def prompt_formation_and_tokenize_race(example):
    problems = ast.literal_eval(example["problems"])
    last_problem = problems[-1]
    
    instruction = "Article: " + example["article"] + "\n\n"
    
    # Process context examples
    for problem in problems[:-1]:
        letter_to_num = {"A": 0, "B": 1, "C": 2, "D": 3}
        answer_idx = letter_to_num[problem["answer"]]
        correct_answer = problem["options"][answer_idx]
        
        if "  _  " in problem["question"]: # Safer check
            # Fill in blank style
            parts = problem["question"].split("  _  ")
            # Reconstruct with answer
            instruction += parts[0] + " " + correct_answer
            if len(parts) > 1:
                instruction += " " + parts[1]
            instruction += "\n"
        elif problem["question"].endswith("_"):
             instruction += problem["question"][:-1] + " " + correct_answer + "\n"
        else:
            # QA style
            question = "Question: " + problem["question"] + "\n"
            answer = "Answer: " + correct_answer + "\n"
            instruction += question + answer
    
    # Process the target (last) problem
    instruction += last_problem["question"]
    
    letter_to_num = {"A": 0, "B": 1, "C": 2, "D": 3}
    answer_idx = letter_to_num[last_problem["answer"]]
    response = " " + last_problem["options"][answer_idx]
    
    return tokenize_and_mask_helper(instruction, response, tokenizer)