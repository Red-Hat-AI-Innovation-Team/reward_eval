"""
RewardBench: A benchmark for evaluating reward models.
This module provides utilities for preparing evaluation data and calculating benchmark scores.
"""

import os
import json
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from typing import Dict, List, Optional, Any


# Constants for benchmark evaluation
EXAMPLE_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 250,
    "xstest-should-respond": 154,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
    "helpful-base": 500,
    "harmless-base": 500,
    "summarize": 500,
    "hhh-base": 221,
}

SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}

def load_rewardbench_data(dataset_name: str = "allenai/reward-bench") -> Dataset:
    """
    Load the RewardBench dataset.
    
    Args:
        dataset_name: HuggingFace dataset name or path
        
    Returns:
        Dataset containing the evaluation examples
    """
    # Load core dataset
    core_ds = load_dataset(dataset_name, split='filtered', keep_in_memory=True)

    return core_ds

def calculate_accuracy_from_scores(example_ids: List[str], 
                                  example_subsets: List[str],
                                  scores: List[float]) -> pd.DataFrame:
    """
    Calculate accuracy from model scores.
    
    Args:
        example_ids: List of example IDs
        example_subsets: List of example subset names
        scores: List of model scores for each example
        
    Returns:
        DataFrame with accuracy results per example
    """
    # Create a mapping from ID to scores
    id_to_scores = {}
    id_to_subset = {}
    
    for i in range(0, len(scores), 2):
        if i+1 < len(scores):
            example_id = example_ids[i]
            subset = example_subsets[i]
            chosen_score = scores[i]
            rejected_score = scores[i+1]
            
            id_to_scores[example_id] = (chosen_score, rejected_score)
            id_to_subset[example_id] = subset
    
    # Calculate accuracy
    results = []
    for example_id, (chosen_score, rejected_score) in id_to_scores.items():
        if chosen_score > rejected_score:
            correct = 1.0
        elif chosen_score < rejected_score:
            correct = 0.0
        else:
            correct = 0.5
            
        results.append({
            'id': example_id,
            'subset': id_to_subset[example_id],
            'correct': correct
        })
    
    return pd.DataFrame(results)

def calculate_subset_accuracy(accuracy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate accuracy per subset.
    
    Args:
        accuracy_df: DataFrame with accuracy results
        
    Returns:
        DataFrame with accuracy per subset
    """
    subset_results = []
    
    for subset in accuracy_df['subset'].unique():
        subset_df = accuracy_df[accuracy_df['subset'] == subset]
        accuracy = subset_df['correct'].mean()
        count = len(subset_df)
        
        subset_results.append({
            'subset': subset,
            'accuracy': accuracy,
            'count': count
        })
    
    return pd.DataFrame(subset_results)

def calculate_category_scores(subset_accuracy_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate weighted scores for each category.
    
    Args:
        subset_accuracy_df: DataFrame with accuracy per subset
        
    Returns:
        Dictionary with category scores
    """
    category_scores = {}
    
    # Create mapping from subset to accuracy
    subset_to_accuracy = dict(zip(subset_accuracy_df['subset'], subset_accuracy_df['accuracy']))
    
    # Calculate weighted score for each category
    for category, subsets in SUBSET_MAPPING.items():
        total_weighted_score = 0
        total_examples = 0
        
        for subset in subsets:
            if subset in subset_to_accuracy:
                weight = EXAMPLE_COUNTS.get(subset, 0)
                total_weighted_score += subset_to_accuracy[subset] * weight
                total_examples += weight
        
        if total_examples > 0:
            category_scores[category] = round(100 * total_weighted_score / total_examples, 2)
        else:
            category_scores[category] = 0.0
    
    # Calculate overall scores
    rewardbench_categories = ["Chat", "Chat Hard", "Safety", "Reasoning"]
    rewardbench_score = np.mean([category_scores[cat] for cat in rewardbench_categories])
    
    category_scores["RewardBench"] = round(rewardbench_score, 1)
    
    return category_scores

def save_results(category_scores: Dict[str, float], 
                subset_accuracy: pd.DataFrame,
                output_dir: str,
                model_name: str) -> None:
    """
    Save evaluation results to files.
    
    Args:
        category_scores: Dictionary with category scores
        subset_accuracy: DataFrame with accuracy per subset
        output_dir: Directory to save results
        model_name: Name of the model being evaluated
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save category scores
    with open(os.path.join(output_dir, f"{model_name}_scores.json"), 'w') as f:
        json.dump(category_scores, f, indent=2)
    
    # Save subset accuracy
    subset_accuracy.to_csv(os.path.join(output_dir, f"{model_name}_subset_accuracy.csv"), index=False)




def evaluate_model(model, dataset, output_dir: str, model_name: str) -> Dict[str, float]:
    """
    Evaluate a reward model on the RewardBench dataset and save results.
    
    Args:
        model: The reward model to evaluate
        dataset: RewardBench dataset
        output_dir: Directory to save results
        model_name: Name of the model being evaluated
        
    Returns:
        Dictionary with category scores
    """
    example_ids = []
    example_subsets = []
    scores = []

    # Score chosen response
    chosen_scores = generate_score(model, [example['prompt'] for example in dataset], [example['chosen'] for example in dataset])
    
    # Score rejected response
    rejected_scores = generate_score(model, [example['prompt'] for example in dataset], [example['rejected'] for example in dataset])

    # Process each example in the dataset
    for i, example in enumerate(dataset):
        example_id = example['id']
        subset = example['subset']
        

        
        # Store results
        example_ids.extend([example_id, example_id])
        example_subsets.extend([subset, subset])
        scores.extend([chosen_scores[i], rejected_scores[i]])
    
    # Calculate accuracy
    accuracy_df = calculate_accuracy_from_scores(example_ids, example_subsets, scores)
    
    # Calculate subset accuracy
    subset_accuracy_df = calculate_subset_accuracy(accuracy_df)
    
    # Calculate category scores
    category_scores = calculate_category_scores(subset_accuracy_df)
    
    # Save results
    save_results(category_scores, subset_accuracy_df, output_dir, model_name)
    
    # Print results
    print(f"RewardBench Score: {category_scores['RewardBench']}")
    print("\nCategory Scores:")
    for category in SUBSET_MAPPING.keys():
        print(f"  {category}: {category_scores[category]}")
    
    print("\nTop 5 Best Performing Subsets:")
    top_subsets = subset_accuracy_df.sort_values('accuracy', ascending=False).head(5)
    for _, row in top_subsets.iterrows():
        print(f"  {row['subset']}: {row['accuracy']:.4f} ({row['count']} examples)")
    
    print("\nBottom 5 Worst Performing Subsets:")
    bottom_subsets = subset_accuracy_df.sort_values('accuracy').head(5)
    for _, row in bottom_subsets.iterrows():
        print(f"  {row['subset']}: {row['accuracy']:.4f} ({row['count']} examples)")
    
    return category_scores


from reward_hub import AutoRM
from reward_hub.drsow import DrSowConfig

def generate_score(reward_model, questions, answers) -> float:
    """
    Generate a score for a given question and answer.
    """
    messages_list = [[
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ] for question, answer in zip(questions, answers)]
    reward_results = reward_model.score(messages_list, return_raw_scores=True)
    scores = [x["drsow_reward"] for x in reward_results]
    return scores


if __name__ == "__main__":

    dataset = load_rewardbench_data()
    output_dir = "evals/rewardbench"

    drsow_config = DrSowConfig(
        strong_model_name="Qwen/Qwen2.5-32B-instruct",
        strong_port=8305,
        weak_model_name="Qwen/Qwen2.5-32B",
        weak_port=8306
    )
    model = AutoRM.load("drsow", load_method="openai", drsow_config=drsow_config)

    evaluate_model(model, dataset, output_dir, "drsow_qwen32b_instruct")
    breakpoint()

