from dataset import DatasetLoader
from preprocessor import Preprocessor
from prompt_builder import PromptBuilder
from generator import LLMGenerator

# 1. Load Dataset & Inspect 5 Samples
loader = DatasetLoader(data_dir="data")
train_samples, test_samples = loader.load_splits()
print(f"Loaded {len(train_samples)} training and {len(test_samples)} test samples.\n")

# 2. Preprocess & Select Samples
preprocessor = Preprocessor(min_length=10, max_length=1000)
processed_test = preprocessor.select_samples(test_samples, limit=3)

# 3. Construct Prompts
system_prompt = "Classify the following news article into categories like World, Sports, Business, or Sci/Tech."
prompt_builder = PromptBuilder(system_instruction=system_prompt)

# 4. Initialize Pretrained LLM Generator
llm_generator = LLMGenerator(model_name="distilbert/distilgpt2")

# 5. Run Autoregressive Generation on Test Samples
for sample in processed_test:
    # Build prompt
    prompt_messages = prompt_builder.build_zero_shot_prompt(sample)
    
    print("=" * 70)
    print(f"Processing Sample ID: {sample.sample_id}")
    
    # Generate text autoregressively
    result_text = llm_generator.generate(prompt_messages, max_new_tokens=40)
    
    print("\n--- Model Output ---")
    print(result_text)
    print("=" * 70)