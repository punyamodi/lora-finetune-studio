import math
from typing import List, Dict, Optional
import numpy as np


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            scores["rouge1"].append(result["rouge1"].fmeasure)
            scores["rouge2"].append(result["rouge2"].fmeasure)
            scores["rougeL"].append(result["rougeL"].fmeasure)
        return {k: round(np.mean(v), 4) for k, v in scores.items()}
    except ImportError:
        return {}


def compute_perplexity(loss: float) -> float:
    return round(math.exp(loss), 4)


def format_training_summary(metrics: dict) -> str:
    lines = []
    if "train_loss" in metrics:
        lines.append(f"Training Loss: {metrics['train_loss']:.4f}")
    if "train_loss" in metrics:
        lines.append(f"Perplexity: {compute_perplexity(metrics['train_loss']):.4f}")
    if "train_runtime" in metrics:
        runtime = metrics["train_runtime"]
        lines.append(f"Runtime: {int(runtime // 60)}m {int(runtime % 60)}s")
    if "train_samples_per_second" in metrics:
        lines.append(f"Samples/sec: {metrics['train_samples_per_second']:.2f}")
    return "\n".join(lines)
