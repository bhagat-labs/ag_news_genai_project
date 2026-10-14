# evaluate_model.py
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from structured_validator import extract_json_object

def evaluate_test_split():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_dir = "./qwen_ag_news_adapter"  # Path to your trained LoRA adapter (if trained)
    
    print("Loading test dataset from data/test.csv...")
    dataset = load_dataset("csv", data_files={"test": "data/test.csv"})
    test_samples = dataset["test"]

    print(f"Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    # Check if a fine-tuned adapter exists; if so, load it!
    if os.path.exists(adapter_dir):
        print(f"Loading fine-tuned LoRA weights from {adapter_dir}...")
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        print("No fine-tuned adapter found. Evaluating with base pretrained model...")
        model = base_model

    model.eval()

    correct_predictions = 0
    total_evaluated = 0

    print("\nRunning Evaluation on Test Split...")
    print("=" * 60)

    # Evaluate on a subset (e.g., first 50 test records) or whole set
    for idx, sample in enumerate(test_samples[:50]):
        text = sample.get("text", "")
        true_label = sample.get("label", "")

        # Construct prompt matching your generation pipeline
        prompt = (
            "You are classifying a news article. Return valid JSON only.\n"
            "Do not add markdown fences, explanations, or commentary.\n"
            "Valid categories: World, Sports, Business, Sci/Tech.\n"
            "Required JSON schema: {\"category\": \"<one of the valid categories>\", \"confidence_score\": 0.0-1.0, \"reasoning\": \"brief explanation\"}\n\n"
            f"Article: {text}\n"
            "JSON:"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = output_ids[0][prompt_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Parse and validate prediction
        predicted_category = "Unknown"
        try:
            parsed = extract_json_object(response)
            predicted_category = parsed.get("category", "").strip()
        except Exception:
            # Fallback text check if JSON parsing fails
            for cat in ["World", "Sports", "Business", "Sci/Tech"]:
                if cat.lower() in response.lower():
                    predicted_category = cat
                    break

        is_match = (predicted_category.lower() == true_label.lower())
        if is_match:
            correct_predictions += 1
        total_evaluated += 1

        print(f"Sample [{idx+1}] -> True: {true_label} | Predicted: {predicted_category} | Match: {'PASS' if is_match else 'FAIL'}")

    accuracy = (correct_predictions / total_evaluated) * 100
    print("=" * 60)
    print(f"EVALUATION COMPLETE:")
    print(f"Total Tested: {total_evaluated}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_test_split()