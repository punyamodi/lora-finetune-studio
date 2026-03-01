import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datasets import Dataset, DatasetDict, load_dataset

from ..config import DataConfig


class DatasetLoader:
    def __init__(self, config: DataConfig):
        self.config = config

    def load_from_csv(self, file_path: str) -> Dataset:
        df = pd.read_csv(file_path)
        return Dataset.from_pandas(df)

    def load_from_hub(self, dataset_name: str, config: Optional[str] = None) -> DatasetDict:
        return load_dataset(dataset_name, config)

    def load(self, file_path: Optional[str] = None) -> DatasetDict:
        if file_path is not None:
            dataset = self.load_from_csv(file_path)
        elif self.config.dataset_name:
            dataset = self.load_from_hub(self.config.dataset_name, self.config.dataset_config)
            if isinstance(dataset, DatasetDict) and "train" in dataset:
                return dataset
        else:
            dataset = self.load_from_csv(self.config.train_file)

        split = dataset.train_test_split(test_size=self.config.val_split, seed=42)
        return DatasetDict({"train": split["train"], "validation": split["test"]})

    def format_dataset(self, dataset: DatasetDict, formatter) -> DatasetDict:
        return dataset.map(formatter, batched=False, remove_columns=dataset["train"].column_names)

    def get_dataset_stats(self, dataset: DatasetDict) -> Dict[str, Any]:
        stats = {}
        for split_name, split_data in dataset.items():
            stats[split_name] = {
                "num_examples": len(split_data),
                "columns": split_data.column_names,
            }
        return stats

    @staticmethod
    def preview_csv(file_path: str, n_rows: int = 5) -> pd.DataFrame:
        return pd.read_csv(file_path, nrows=n_rows)

    @staticmethod
    def get_csv_stats(file_path: str) -> Dict[str, Any]:
        df = pd.read_csv(file_path)
        return {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        }
