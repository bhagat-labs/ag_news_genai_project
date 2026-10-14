# train_lora.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def train_model():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir = "./qwen_ag_news_adapter"

    print(f"Loading dataset from data/train.csv...")
    # Load dataset using Hugging Face datasets library
    dataset = load_dataset("csv", data_files={"train": "data/train.csv"})

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Important for training

    # Formatting function to map AG News columns into an instruction format
    def format_prompt(sample):
        # Adjust column names ("text" and "label") if your CSV uses different headers
        text = sample.get("text", "")
        label = sample.get("label", "")
        
        prompt = (
            "You are classifying a news article. Return valid JSON only.\n"
            f"Valid categories: World, Sports, Business, Sci/Tech.\n\n"
            f"Article: {text}\n"
            f"JSON: {{\"category\": \"{label}\", \"confidence_score\": 1.0, \"reasoning\": \"True dataset label.\"}}"
        )
        return {"text": prompt}

    # Map dataset formatting
    train_dataset = dataset["train"].map(format_prompt)

    print("Configuring 4-bit Quantization (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True,
    )

    # Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Define LoRA Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=1, # Adjust based on dataset size
        max_steps=200,      # Remove or increase for full dataset training
        optim="paged_adamw_8bit",
        fp16=not torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        save_strategy="steps",
        save_steps=100,
    )

    # Initialize SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting Training Loop...")
    trainer.train()

    print(f"Saving fine-tuned LoRA adapter to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_model()