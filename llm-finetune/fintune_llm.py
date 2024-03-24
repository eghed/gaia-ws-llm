from typing import Dict, Optional, List

import torch
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, SFTTrainer

# TODO, refs:
# https://huggingface.co/blog/pref-tuning
# https://huggingface.co/blog/dpo-trl

#
# Dataset ARGS:
model_name_or_path = "TODO"
training_args: Optional[TrainingArguments] = None

#
# Model Args:
model_name = "TinyPixel/Llama-2-7B-bf16-sharded"
# model_name = "meta-llama/Llama-2-7b-hf"

#
# SFT ARGS:
# TODO
lora_r = 8
lora_alpha = 8
lora_dropout = 0.0

#
# DPO ARGS:
dpo_beta: float = 0.1


#
# Setup dataset:
def return_prompts_and_responses(batch: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Restructure dataset to fit HF TRL

    Args:
        batch:

    Returns:

    """
    prompts = [f"Question: {question} \n\nAnswer: " for question in batch["question"]]

    return {
        'prompt': list(prompts),
        'chosen': list(batch["response_j"]),
        'rejected': list(batch["response_k"])
    }


print(" - Loading datasets")
# ds_name = "lvwerra/stack-exchange-paired"
ds_name = "MaestroDmitry/stack-exchange-paired-shorted"
dataset = load_dataset(
    ds_name,
    # data_dir="data/rl",
    cache_dir="data"
)

train_dataset = dataset['train']
test_dataset = dataset['test']

print(" - remapping datasets")
# TODO copy paste
original_columns = train_dataset.column_names
train_dataset.map(
    function=return_prompts_and_responses,
    batched=True,
    with_indices=False,
    remove_columns=original_columns
)
original_columns = test_dataset.column_names
test_dataset.map(
    function=return_prompts_and_responses,
    batched=True,
    with_indices=False,
    remove_columns=original_columns
)


#
# Load base model:
print(" - loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load the base model in 4-bit quantization
# TODO this only works on cuda
print(" - bits and bytes")
"""bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)"""

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    # use_auth_token=True,
)

base_model.config.use_cache = False

#
# Supervised Fine-Tuning:

# add LoRA layers on top of the quantized base model
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# ...
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    packing=True,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,  # HF Trainer arguments
)
trainer.train()

#
# DPO Training
model = AutoPeftModelForCausalLM.from_pretrained(
    model_name_or_path,  # location of saved SFT model
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    is_trainable=True,
)

model_ref = AutoPeftModelForCausalLM.from_pretrained(
    model_name_or_path,  # same model as the main one
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=dpo_beta,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)
dpo_trainer.train()
dpo_trainer.save_model()
