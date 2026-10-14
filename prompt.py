# step1_prompt.py

def construct_prompt(text: str, label_description: str = "World, Sports, Business, Sci/Tech") -> str:
    """Constructs a strict prompt for AG News classification."""
    labels = [label.strip() for label in label_description.split(",")]
    label_list = ", ".join(labels)

    prompt = (
        "You are classifying a news article. Return exactly one label and nothing else.\n"
        f"Valid labels: {label_list}.\n\n"
        f"Article: {text}\n"
        "Label:"
    )
    return prompt


def normalize_label(raw_label: str, valid_labels: list[str] | tuple[str, ...] | None = None) -> str:
    """Extracts a valid label from model output and returns it normalized."""
    if raw_label is None:
        return ""

    cleaned = raw_label.strip().strip("`\"'")
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.split(" ")[0]

    if valid_labels is not None:
        for label in valid_labels:
            if cleaned.lower() == label.lower():
                return label
            if label.lower() in cleaned.lower():
                return label

    return cleaned


if __name__ == "__main__":
    sample_text = "Oracle announces new breakthrough cloud infrastructure services."
    prompt = construct_prompt(sample_text)
    print("--- STEP 1: Constructed Prompt ---")
    print(prompt)