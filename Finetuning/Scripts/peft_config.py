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
    AutoModel,
)
from contextlib import nullcontext
from tqdm import tqdm
import json
import copy
import datasets
from peft import LoraConfig, PeftConfig
from transformers import default_data_collator, Trainer
import os
import transformers
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

def create_peft_config(model):
    peft_config = LoraConfig(
          task_type = TaskType.CAUSAL_LM,
          inference_mode=False,
          r=8,
          lora_alpha = 64,
          lora_dropout=0.1,
          target_modules = ["q_proj","v_proj"]
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model,peft_config