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

from Dataset import create_path, load_dataset
from model_setup import load_model, load_tokenizer, save_model, save_tokenizer, pre_trained_model_inference
from peft_config import create_peft_config

MODEL_PATH:str = 'MODEL PATH'
TOKENIZER_PATH:str = 'TOKENIZER PATH'
DATASET_PATH:str = 'DATA PATH'
PROCESSED_DATASET_PATH:str = 'PROCESSED DATA PATH'
FINE_TUNED_MODEL_PATH:str = 'FINAL MODEL PATH'
MODEL_NAME:str = "codellama/CodeLlama-7b-hf"
DATASET_ID:str = "HuggingFaceH4/CodeAlpaca_20K"

if __name__ =='__main__':
    create_path(TOKENIZER_PATH)
    create_path(DATASET_PATH)
    create_path(PROCESSED_DATASET_PATH)
    create_path(FINE_TUNED_MODEL_PATH)
    model = load_model(MODEL_NAME)
    tokenizer = load_tokenizer(MODEL_NAME)
    dataset = load_dataset(DATASET_ID,tokenizer,'train',DATASET_PATH,PROCESSED_DATASET_PATH)
    train_dataset = datasets.load_from_disk(PROCESSED_DATASET_PATH)
    model, config = create_peft_config(model)
    training_arguments = TrainingArguments(
        output_dir="logs",
        num_train_epochs=0.5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        bf16=False,
        group_by_length=True,
        logging_strategy="steps",
        save_strategy="no",
        gradient_checkpointing=False,
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    print(f'saving model to:{FINE_TUNED_MODEL_PATH}')
    trainer.model.save_pretrained(FINE_TUNED_MODEL_PATH)
