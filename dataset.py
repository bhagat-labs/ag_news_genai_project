import os
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class RawSample:
    sample_id: str
    text_content: str
    metadata: dict
    label: Optional[str] = None

class DatasetLoader:
    def __init__(self, data_dir: str = "data"):
        self.train_path = os.path.join(data_dir, "train.csv")
        self.test_path = os.path.join(data_dir, "test.csv")

    def _load_file(self, file_path: str, is_train: bool = True) -> List[RawSample]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find file at: {file_path}")
            
        df = pd.read_csv(file_path)
        samples = []
        
        for idx, row in df.iterrows():
            samples.append(
                RawSample(
                    sample_id=str(row.get("id", f"{'train' if is_train else 'test'}_{idx}")),
                    text_content=str(row.get("text", "")),
                    metadata={"split": "train" if is_train else "test"},
                    label=str(row.get("label")) if is_train and "label" in row else None
                )
            )
        return samples

    def load_splits(self) -> Tuple[List[RawSample], List[RawSample]]:
        """Loads both train and test splits from the data directory."""
        train_samples = self._load_file(self.train_path, is_train=True)
        test_samples = self._load_file(self.test_path, is_train=False)
        return train_samples, test_samples

    def display_sample_records(self, samples: List[RawSample], num_records: int = 5):
        """Displays a formatted sample of records from the loaded dataset."""
        print("=" * 70)
        print(f"SAMPLE OF {num_records} RECORDS:")
        print("=" * 70)
        
        for i, sample in enumerate(samples[:num_records], 1):
            print(f"Record {i}:")
            print(f"  - ID:     {sample.sample_id}")
            print(f"  - Label:  {sample.label}")
            # Truncate long texts for clean printing
            text_preview = sample.text_content[:150] + "..." if len(sample.text_content) > 150 else sample.text_content
            print(f"  - Text:   {text_preview}")
            print(f"  - Meta:   {sample.metadata}")
            print("-" * 70)

# Optional: Direct execution block for testing dataset.py independently
if __name__ == "__main__":
    loader = DatasetLoader(data_dir="data")
    train_samples, test_samples = loader.load_splits()
    
    print(f"Loaded {len(train_samples)} training samples and {len(test_samples)} test samples.\n")
    
    # Display 5 sample training records automatically
    loader.display_sample_records(train_samples, num_records=5)