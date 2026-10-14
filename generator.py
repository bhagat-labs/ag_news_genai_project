import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

class LLMGenerator:
    def __init__(self, model_name: str = "distilbert/distilgpt2"):
        """Initializes the tokenizer and pretrained causal language model."""
        print(f"Loading model and tokenizer: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def generate(self, prompt_messages: List[Dict[str, str]], max_new_tokens: int = 60) -> str:
        """
        1. Prompt Construction formatting conversion
        2. Tokenizer encoding
        3. Pretrained LLM inference
        4. Autoregressive generation & decoding
        """
        # Flatten prompt messages into a single text string for causal LLMs
        formatted_prompt = ""
        for msg in prompt_messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted_prompt += f"{role}: {content}\n"
        formatted_prompt += "Assistant: "

        # Step 2: Tokenizer -> Input Tensors
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )

        # Step 3 & 4: Pretrained LLM & Autoregressive Generation
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode generated token IDs back into text
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text