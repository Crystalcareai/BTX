import signal
import sys

# Signal handler function
def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Exiting...')
    sys.exit(0)

from datasets import load_dataset
from trl import SFTTrainer

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig


# Load jsonl data from HF or disk
dataset = load_dataset("Crystalcareai/Teknium-OpenHermes-2.5-250k-trl", split="train")

# Hugging Face model id
model_id = "Crystalcareai/GemMoE-Medium-Base"
tokenizer_id = "philschmid/gemma-tokenizer-chatml"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.padding_side = 'right' # to prevent warnings


peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.0,
        r=16,
        bias="none",
        target_modules="all-linear",  
        use_dora=False,  
)


max_seq_length = 2042

# Freeze the expert parameters
for layer in model.model.layers:
    for expert in layer.block_sparse_moe.experts:
        for param in expert.parameters():
            param.requires_grad = False

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="output/250k",
    num_train_epochs=2.5,
    per_device_train_batch_size=3,
    gradient_accumulation_steps=2,
    optim="adamw_torch_fused",
    logging_steps=2,                      
    save_strategy="steps",                   
    save_steps=1000, 
    learning_rate=1e-4,
    weight_decay=0.1,
    report_to="wandb",
    bf16=True,                              # use bfloat16 precision
    tf32=True,
    lr_scheduler_type="constant",
    warmup_steps=100,
    max_grad_norm=1.0,
    seed=42,
)

# Create the SFT trainer
trainer =SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    packing=True,

)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained()