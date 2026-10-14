# step4_generate.py
import torch
from prompt import construct_prompt
from tokenizer import tokenize_prompt
from model import load_pretrained_llm

def autoregressive_generation(tokenized_inputs, model, tokenizer, device, max_new_tokens: int = 15):
    """Generates text autoregressively using the Qwen model."""
    input_ids = tokenized_inputs["input_ids"].to(device)
    attention_mask = tokenized_inputs["attention_mask"].to(device)
    
    print("Generating response autoregressively with Qwen...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False  # Deterministic greedy generation for classification
        )
        
    # Decode only the generated tokens (excluding the prompt)
    prompt_length = input_ids.shape[1]
    generated_tokens = output_ids[0][prompt_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response

if __name__ == "__main__":
    # 1. Prompt
    text = "SpaceX successfully launches new satellite payload into orbit."
    prompt = construct_prompt(text)
    
    # 2. Tokenize
    tokenizer, tokenized_inputs = tokenize_prompt(prompt)
    
    # 3. Load Model (Qwen 1.5B)
    model, device = load_pretrained_llm()
    
    # 4. Generate
    result = autoregressive_generation(tokenized_inputs, model, tokenizer, device, max_new_tokens=10)
    
    print("\n--- STEP 4: Final Autoregressive Output ---")
    print(f"Prompt:\n{prompt}")
    print(f"Generated Output: {result}")