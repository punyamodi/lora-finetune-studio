from typing import Optional, Callable
from datasets import Dataset
from transformers import PreTrainedTokenizer

from ..config import DataConfig


PROMPT_TEMPLATES = {
    "instruction": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    ),
    "chat": "<s>[INST] {instruction} [/INST] {response}</s>",
    "alpaca": "### Human: {instruction}\n### Assistant: {response}",
    "raw": "{text}",
}


class TextPreprocessor:
    def __init__(self, config: DataConfig, tokenizer: Optional[PreTrainedTokenizer] = None):
        self.config = config
        self.tokenizer = tokenizer

    def get_formatter(self, template: str = "instruction") -> Callable:
        template_str = PROMPT_TEMPLATES.get(template, PROMPT_TEMPLATES["raw"])

        def format_example(example):
            if template == "raw":
                text = example.get(self.config.text_column, "")
            elif self.config.prompt_column and self.config.response_column:
                text = template_str.format(
                    instruction=example.get(self.config.prompt_column, ""),
                    response=example.get(self.config.response_column, ""),
                )
            else:
                text = example.get(self.config.text_column, "")
            return {"text": text}

        return format_example

    def tokenize(self, example: dict) -> dict:
        result = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def compute_token_stats(self, dataset: Dataset) -> dict:
        if self.tokenizer is None:
            return {}
        lengths = [
            len(self.tokenizer(ex["text"]).input_ids)
            for ex in dataset
        ]
        return {
            "min_tokens": min(lengths),
            "max_tokens": max(lengths),
            "mean_tokens": round(sum(lengths) / len(lengths), 1),
            "over_max": sum(1 for l in lengths if l > self.config.max_seq_length),
        }
