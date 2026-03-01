import pytest
import pandas as pd
import tempfile
import os
from src.lorakit.config import DataConfig
from src.lorakit.data.dataset import DatasetLoader


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "instruction": ["Write a poem about AI", "Explain neural networks", "What is LoRA?"],
        "output": ["AI dreams in data streams...", "Neural networks are...", "LoRA is a fine-tuning method..."],
        "text": ["Sample text 1", "Sample text 2", "Sample text 3"],
    })
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_load_from_csv(sample_csv):
    config = DataConfig(train_file=sample_csv)
    loader = DatasetLoader(config)
    dataset = loader.load_from_csv(sample_csv)
    assert len(dataset) == 3
    assert "instruction" in dataset.column_names


def test_load_creates_splits(sample_csv):
    config = DataConfig(train_file=sample_csv, val_split=0.33)
    loader = DatasetLoader(config)
    dataset = loader.load(file_path=sample_csv)
    assert "train" in dataset
    assert "validation" in dataset
    assert len(dataset["train"]) + len(dataset["validation"]) == 3


def test_csv_stats(sample_csv):
    stats = DatasetLoader.get_csv_stats(sample_csv)
    assert stats["num_rows"] == 3
    assert stats["num_columns"] == 3
    assert "instruction" in stats["columns"]


def test_preview_csv(sample_csv):
    preview = DatasetLoader.preview_csv(sample_csv, n_rows=2)
    assert isinstance(preview, pd.DataFrame)
    assert len(preview) == 2


def test_dataset_stats(sample_csv):
    config = DataConfig(train_file=sample_csv)
    loader = DatasetLoader(config)
    dataset = loader.load(file_path=sample_csv)
    stats = loader.get_dataset_stats(dataset)
    assert "train" in stats
    assert "num_examples" in stats["train"]
