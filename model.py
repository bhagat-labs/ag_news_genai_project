# step3_model.py
import torch
from transformers import AutoModelForCausalLM

def load_pretrained_llm(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Loads the pretrained Qwen causal language model."""
    print(f"Loading pretrained model: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use bfloat16 if supported by GPU for better efficiency, else float32
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if not torch.cuda.is_available():
        model.to(device)
        
    model.eval()
    return model, device

if __name__ == "__main__":
    model, device = load_pretrained_llm()
    print("--- STEP 3: Model Loaded Successfully ---")
    print(f"Running on device: {device}")