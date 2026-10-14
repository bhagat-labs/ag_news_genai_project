# comparative_evaluation.py
from dataclasses import dataclass
from typing import List, Dict, Any

# 1. Define Evaluation Sample Structure
@dataclass
class EvalSample:
    article_id: str
    text: str
    true_label: str

# 2. Mock Baseline and Improved LLM Evaluators
def mock_baseline_llm_call(text: str) -> str:
    """Simulates a baseline model/prompt (less precise, prone to formatting issues)."""
    # Simple keyword heuristic as a weak baseline
    if "space" in text.lower() or "tech" in text.lower():
        return "Sci/Tech"
    elif "stock" in text.lower() or "market" in text.lower():
        return "Business"
    return "World"

def mock_improved_llm_call(text: str) -> str:
    """Simulates an improved system prompt with structured constraints and few-shot examples."""
    # Better simulated classification logic
    if any(k in text.lower() for k in ["space", "chip", "tech", "software"]):
        return "Sci/Tech"
    elif any(k in text.lower() for k in ["stock", "market", "economy", "bank"]):
        return "Business"
    elif any(k in text.lower() for k in ["game", "team", "championship", "player"]):
        return "Sports"
    return "World"

# 3. Comparative Evaluation Runner
def run_comparative_evaluation(eval_dataset: List[EvalSample]) -> Dict[str, Any]:
    """Runs evaluation on both baseline and improved setups, then compares metrics."""
    baseline_correct = 0
    improved_correct = 0
    total = len(eval_dataset)
    
    detailed_results = []

    print("Running Evaluation Pipeline...\n" + "="*60)
    
    for sample in eval_dataset:
        # Get predictions
        base_pred = mock_baseline_llm_call(sample.text)
        impr_pred = mock_improved_llm_call(sample.text)
        
        b_correct = (base_pred.strip().lower() == sample.true_label.strip().lower())
        i_correct = (impr_pred.strip().lower() == sample.true_label.strip().lower())
        
        if b_correct:
            baseline_correct += 1
        if i_correct:
            improved_correct += 1
            
        detailed_results.append({
            "id": sample.article_id,
            "true": sample.true_label,
            "baseline_pred": base_pred,
            "baseline_match": b_correct,
            "improved_pred": impr_pred,
            "improved_match": i_correct
        })

    # Calculate final accuracy metrics
    baseline_accuracy = (baseline_correct / total) * 100
    improved_accuracy = (improved_correct / total) * 100
    
    analysis_report = {
        "total_samples": total,
        "baseline_accuracy": baseline_accuracy,
        "improved_accuracy": improved_accuracy,
        "accuracy_delta": improved_accuracy - baseline_accuracy,
        "detailed_results": detailed_results
    }
    
    return analysis_report

if __name__ == "__main__":
    # Sample Evaluation Dataset (e.g., from your test.csv split)
    evaluation_data = [
        EvalSample("1", "SpaceX successfully launches a new batch of communication satellites.", "Sci/Tech"),
        EvalSample("2", "Central banks hint at potential interest rate cuts amid slowing inflation.", "Business"),
        EvalSample("3", "The national cricket team clinched a thrilling victory in the final over.", "Sports"),
        EvalSample("4", "Global leaders convene at the summit to address international trade agreements.", "World"),
        EvalSample("5", "Semiconductor manufacturers announce next-generation 2nm processor nodes.", "Sci/Tech")
    ]

    # Run the evaluation
    report = run_comparative_evaluation(evaluation_data)

    # 4. Print Comparative Analysis Report
    print("=" * 60)
    print("COMPARATIVE ANALYSIS REPORT")
    print("=" * 60)
    print(f"Total Evaluation Samples: {report['total_samples']}")
    print(f"Baseline Prompt Accuracy:  {report['baseline_accuracy']:.2f}%")
    print(f"Improved Prompt Accuracy:  {report['improved_accuracy']:.2f}%")
    print(f"Performance Delta (Gain):  +{report['accuracy_delta']:.2f}%")
    print("-" * 60)
    
    print("\nDetailed Sample-by-Sample Breakdown:")
    for res in report['detailed_results']:
        print(f"  [Sample ID: {res['id']}] True: {res['true']}")
        print(f"    -> Baseline: {res['baseline_pred']} ({'PASS' if res['baseline_match'] else 'FAIL'})")
        print(f"    -> Improved: {res['improved_pred']} ({'PASS' if res['improved_match'] else 'FAIL'})")
    print("=" * 60)