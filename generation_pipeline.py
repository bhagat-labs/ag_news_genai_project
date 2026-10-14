# generation_pipeline.py
import sys

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from structured_validator import NewsClassification, validate_and_parse_response


def load_generation_model(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads the tokenizer and model so callers can reuse them across predictions."""
    print(f"Loading model and tokenizer: {model_name}...")
    use_cuda = torch.cuda.is_available()
    torch_dtype = torch.bfloat16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if use_cuda else None,
    )
    if not use_cuda:
        model.to("cpu")
    model.eval()
    return model, tokenizer


def build_structured_prompt(text: str, valid_labels: tuple[str, ...] = ("World", "Sports", "Business", "Sci/Tech")) -> str:
    label_list = ", ".join(valid_labels)
    return (
        "You are classifying a news article. Return valid JSON only.\n"
        "Do not add markdown fences, explanations, or commentary.\n"
        f"Valid categories: {label_list}.\n"
        "Keep reasoning to one short sentence of at most 15 words.\n"
        "Required JSON schema: {\"category\": \"<one of the valid categories>\", \"confidence_score\": 0.0-1.0, \"reasoning\": \"brief explanation\"}\n\n"
        f"Article: {text}\n"
        "JSON:"
    )


def run_generation_pipeline(
    prompt: str,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    top_k: int = 20,
    top_p: float = 0.85,
    repetition_penalty: float = 1.1,
    do_sample: bool = False,
    valid_labels: tuple[str, ...] = ("World", "Sports", "Business", "Sci/Tech"),
    model_bundle: tuple[PreTrainedModel, PreTrainedTokenizerBase] | None = None,
) -> NewsClassification:
    """Generates classification output, validates it, and returns a structured Pydantic model."""
    model, tokenizer = model_bundle or load_generation_model(model_name)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    prompt_length = input_ids.shape[1]

    print("\nGenerating structured classification output...")
    generation_options = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
    }
    if do_sample:
        generation_options.update(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    with torch.no_grad():
        output_ids = model.generate(**generation_options)

    generated_tokens = output_ids[0][prompt_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    validated = validate_and_parse_response(response)
    return validated


if __name__ == "__main__":
    default_text = "Tech giant Apple unveils a revolutionary new chip architecture for AI data centers."
    user_text = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    article_text = user_text if user_text else input("Enter the news article text: ").strip() or default_text

    sample_prompt = build_structured_prompt(article_text)

    print(f"--- Prompt ---\n{sample_prompt}\n")

    structured_result = run_generation_pipeline(
        prompt=sample_prompt,
        max_new_tokens=128,
        temperature=0.1,
        top_k=20,
        top_p=0.85,
        repetition_penalty=1.1,
        do_sample=False,
    )

    print("\n--- Validated Structured Output ---")
    print(structured_result.model_dump_json(indent=2))