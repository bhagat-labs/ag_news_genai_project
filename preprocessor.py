import re
from typing import List
from dataset import RawSample

class Preprocessor:
    def __init__(self, min_length: int = 10, max_length: int = 2000):
        self.min_length = min_length
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """Cleans whitespace, HTML tags, and special characters."""
        text = re.sub(r'<.*?>', '', text)  # Remove HTML
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text

    def filter_samples(self, samples: List[RawSample]) -> List[RawSample]:
        """Filters samples based on length constraints and validity."""
        valid_samples = []
        for sample in samples:
            cleaned = self.clean_text(sample.text_content)
            if self.min_length <= len(cleaned) <= self.max_length:
                sample.text_content = cleaned
                valid_samples.append(sample)
        return valid_samples

    def select_samples(self, samples: List[RawSample], limit: int = 50) -> List[RawSample]:
        """Selects a targeted subset of samples for prompt construction."""
        filtered = self.filter_samples(samples)
        # Implement sample selection strategy (e.g., first N or stratified)
        return filtered[:limit]