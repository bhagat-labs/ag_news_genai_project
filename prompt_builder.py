from typing import List, Dict, Any
from dataset import RawSample

class PromptBuilder:
    def __init__(self, system_instruction: str):
        self.system_instruction = system_instruction

    def build_zero_shot_prompt(self, sample: RawSample) -> List[Dict[str, str]]:
        """Constructs a chat-style prompt format for a single sample."""
        return [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": f"Analyze the following text:\n\n{sample.text_content}"}
        ]

    def build_few_shot_prompt(self, few_shot_examples: List[Tuple[str, str]], target_sample: RawSample) -> List[Dict[str, str]]:
        """Constructs a few-shot prompt with context examples."""
        messages = [{"role": "system", "content": self.system_instruction}]
        
        for text, label in few_shot_examples:
            messages.append({"role": "user", "content": f"Text: {text}"})
            messages.append({"role": "assistant", "content": f"Label: {label}"})
            
        messages.append({"role": "user", "content": f"Text: {target_sample.text_content}"})
        return messages