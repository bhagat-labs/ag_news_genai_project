# step2_tokenizer.py
from transformers import AutoTokenizer

def tokenize_prompt(prompt: str, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Tokenizes the input prompt into tensor representations using Qwen tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure padding token is set if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    inputs = tokenizer(prompt, return_tensors="pt")
    return tokenizer, inputs

if __name__ == "__main__":
    from prompt import construct_prompt
    prompt = construct_prompt("Stock markets hit record highs today amid tech rally.")
    
    tokenizer, tokenized_inputs = tokenize_prompt(prompt)
    print("--- STEP 2: Tokenizer Output ---")
    print(f"Input Token IDs Shape: {tokenized_inputs['input_ids'].shape}")
    print(f"Decoded check: {tokenizer.decode(tokenized_inputs['input_ids'][0])}")