import torch
from transformers import (
    AutoModelForCausalLM,
    CodeLlamaTokenizer,
    default_data_collator,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from contextlib import nullcontext
from tqdm import tqdm
import json
import copy
import datasets
from peft import LoraConfig, PeftConfig
from transformers import default_data_collator, Trainer
import os

def create_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def load_dataset(dataset_id, tokenizer, split,save_directory,processed_dataset_path):
    dataset = datasets.load_dataset(dataset_id,split=split,cache_dir=save_directory)
    def apply_prompt_template(sample):
            return {
                "prompt": sample["prompt"],
                "message": sample["completion"],
            }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    def tokenize_and_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False, max_length=200, truncation=True)
        message = tokenizer.encode(sample["message"] +  tokenizer.eos_token, max_length=400, truncation=True, add_special_tokens=False)
        max_length = 601 - len(prompt) - len(message)
        if max_length < 0:
            max_length = 0
        pad = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False, max_length=max_length, padding='max_length', truncation=True)
        sample = {
                "input_ids": prompt + message + pad,
                "attention_mask" : [1] * (len(prompt) + len(message) + len(pad)),
                "labels": [-100] * len(prompt) + message + [-100] * len(pad),
                }
    
        return sample
    
    dataset = dataset.map(tokenize_and_add_label, remove_columns=list(dataset.features))
    print(f"Saving processed dataset to {processed_dataset_path}")
    dataset.save_to_disk(processed_dataset_path)
    return dataset

