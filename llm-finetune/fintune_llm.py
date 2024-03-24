from time import sleep
from typing import Dict, Optional, List

import torch
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, SFTTrainer

# TODO, refs:
# https://huggingface.co/blog/pref-tuning
# https://huggingface.co/blog/dpo-trl

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

#
# Dataset ARGS:
# ds_name = "lvwerra/stack-exchange-paired"
ds_name = "MaestroDmitry/stack-exchange-paired-shorted"

#
# Model Args:
# model_path = "/media/kloek/ubu_data/Models/HuggingFace/gpt-neo-1.3B"
model_path = "/media/kloek/ubu_data/Models/HuggingFace/Prajna-gpt-neo-1.3B-fitbot"  # TODO TMP https://huggingface.co/Prajna1999/Prajna-gpt-neo-1.3B-fitbot
# model_path = "/media/kloek/ubu_data/Models/HuggingFace/Mistral-7B-Instruct-v0.2"
# model_path = "/media/kloek/ubu_data/Models/HuggingFace/Mistral-7B-v0.1"
# model_path = "TinyPixel/Llama-2-7B-bf16-sharded"
# model_path = "meta-llama/Llama-2-7b-hf"

# For saved sft model:
model_name_or_path = "TODO"

#
# Lora args:
lora_r = 8
lora_alpha = 8
lora_dropout = 0.0

#
# SFT ARGS:
sft_training_args: Optional[TrainingArguments] = TrainingArguments(
    output_dir="sft_train",
    # use_cpu=True,
    per_device_train_batch_size=4,
    per_gpu_eval_batch_size=4,
)

#
# DPO ARGS:
dpo_beta: float = 0.1
dpo_training_args: Optional[TrainingArguments] = TrainingArguments(
    output_dir="dpo_train",
    # use_cpu=True,
    per_device_train_batch_size=2,  # TODO DPO seems to use one model / gpu, so i can up this!
    per_gpu_eval_batch_size=2,
)


############################################################################################
# Setup dataset:
print("\n=== Setup === \n")
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


print("\n - Loading datasets")
dataset = load_dataset(
    ds_name,
    cache_dir="data"
)

train_dataset = dataset['train']
test_dataset = dataset['test']

print("\n - remapping datasets for DPO")
# TODO copy paste
dpo_train_dataset = train_dataset.map(
    function=return_prompts_and_responses,
    batched=True,
    with_indices=False,
    remove_columns=train_dataset.column_names
)

dpo_test_dataset = test_dataset.map(
    function=return_prompts_and_responses,
    batched=True,
    with_indices=False,
    remove_columns=test_dataset.column_names
)



#############################################################################################
print("\n=== Load Model === \n")
# Load base model:
print("\n - loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# load the base model in 4-bit quantization
# TODO this only works on cuda
print("\n - bits and bytes")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("\n - loading pretrained model")
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map='auto',  # {"": 0},
    trust_remote_code=True,
    # use_auth_token=True,
)

base_model.config.use_cache = False

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    base_model.resize_token_embeddings(len(tokenizer))

print(base_model)

#############################################################################################
# Supervised Fine-Tuning:
print("\n=== FineTuning (SFT) === \n")

def sft_formatting_func(example):
    """
{
        'qid': int,
        'question': str,
        'date': datestr,
        'metadata': [url_str],
        'response_j': str,
        'response_k': str
    }
    Args:
        example:

    Returns:
    """

    output_texts = []
    output_texts.append(f"### Question: {example['question']}\n ### Answer: {example['response_j']}")
    output_texts.append(f"### Question: {example['question']}\n ### Answer: {example['response_k']}")
    return output_texts

# add LoRA layers on top of the quantized base model
print("\n - LoRA config")
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
"""

print("\n - SFTTrainer")
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    packing=True,  # Used only in case `dataset_text_field` is passed. This argument is used by the `ConstantLengthDataset` to pack the sequences of the dataset.
    max_seq_length=None,  # The maximum sequence length to use for the `ConstantLengthDataset` and for automatically creating the Dataset. Defaults to `512`.
    formatting_func=sft_formatting_func,
    tokenizer=tokenizer,
    args=sft_training_args,  # HF Trainer arguments

)
print("\n trainer.train()")
trainer.train()"""

#############################################################################################
# DPO Training
print("\n=== Finetuning (DPO) === \n")

# The model to train, preferably an `AutoModelForSequenceClassification`.
dpo_model = "sft_train/checkpoint-1000"

# Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss.
# If no reference model is provided, the trainer will create a reference model with the same architecture as the model
# to be optimized.
dpo_model_ref = "sft_train/checkpoint-1000"

model = AutoPeftModelForCausalLM.from_pretrained(
    dpo_model,  # location of saved SFT model
    device_map='auto',  # {"": 0},
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    is_trainable=True,
)

model_ref = AutoPeftModelForCausalLM.from_pretrained(
    dpo_model_ref,  # same model as the main one
    device_map='auto',  # {"": 0},
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=dpo_training_args,
    beta=dpo_beta,
    train_dataset=dpo_train_dataset,
    eval_dataset=dpo_test_dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()
dpo_trainer.save_model()
