import os
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
import transformers

def load_model(model_name:str,):
        quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  
        )
        model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code = True,
        quantization_config = quantization_config
        )
        model.config.use_cache = False
        return model

def load_tokenizer(model_name:str):
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          trust_remote_code=True,
                                             )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

def save_model(model,model_path):
  model.save_pretrained(model_path)

def save_tokenizer(tokenizer,tokenizer_path):
  tokenizer.save_pretrained(tokenizer_path)


def pre_trained_model_inference(model,tokenizer):
    pipeline = transformers.pipeline(
        'text-generation',
        model = model,
        torch_dtype = torch.float64,
        device_map='auto',
        tokenizer=tokenizer
    )
    sequences = pipeline(
    'import torch\n\nclass MultiheadAttention(nn.Module):',
    do_sample=True,
    top_k=5,
    temperature=0.2,
    top_p=0.95,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=500,
    )
    for seq in sequences:
        print(f'inference: {seq['generated_text']}')


