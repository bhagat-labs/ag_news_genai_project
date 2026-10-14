# structured_validator.py
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


class NewsClassification(BaseModel):
    category: Literal["World", "Sports", "Business", "Sci/Tech"] = Field(
        description="The predicted category of the news article."
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )
    reasoning: str = Field(
        description="Brief justification for the chosen category."
    )


def extract_json_object(raw_output: str | dict | Any) -> dict[str, Any]:
    """Extracts a JSON object from a raw model response, whether JSON, markdown, or mixed text."""
    if isinstance(raw_output, dict):
        return raw_output

    text = str(raw_output).strip()
    if not text:
        raise ValueError("Model output is empty.")

    decoder = json.JSONDecoder()
    fenced_blocks = re.findall(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    candidates = fenced_blocks + [text]

    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    raise ValueError("No valid JSON object found in model output.")


def validate_and_parse_response(raw_output: str | dict | Any) -> NewsClassification:
    """Parses and validates a raw output from generation_pipeline against the required schema."""
    print(f"Raw Model Output to Validate:\n{raw_output}\n")

    try:
        parsed_data = extract_json_object(raw_output)
        validated_output = NewsClassification(**parsed_data)
        print("Validation Successful!")
        return validated_output

    except (TypeError, ValueError, ValidationError) as exc:
        print(f"Schema Validation Error:\n{exc}")
        raise


def validate_generation_output(output: dict[str, Any]) -> NewsClassification:
    """Convenience wrapper for validating the structured output returned by generation_pipeline."""
    return validate_and_parse_response(output)


if __name__ == "__main__":
    valid_json_response = '''
    {
        "category": "Sci/Tech",
        "confidence_score": 0.95,
        "reasoning": "The article discusses tech giant Apple unveiling a new chip architecture for AI data centers."
    }
    '''

    result = validate_and_parse_response(valid_json_response)
    print("\nValidated Structured Object:")
    print(result.model_dump_json(indent=2))