import pytest
from src.lorakit.config import ModelConfig, LoRAConfig, DataConfig, TrainingConfig, FullConfig


def test_model_config_defaults():
    config = ModelConfig()
    assert config.model_name == "mistralai/Mistral-7B-v0.1"
    assert config.use_4bit is True
    assert config.use_8bit is False


def test_lora_config_defaults():
    config = LoRAConfig()
    assert config.r == 16
    assert config.lora_alpha == 32
    assert "q_proj" in config.target_modules


def test_data_config_defaults():
    config = DataConfig()
    assert config.max_seq_length == 512
    assert config.val_split == 0.1


def test_training_config_defaults():
    config = TrainingConfig()
    assert config.num_train_epochs == 3
    assert config.learning_rate == 2e-4
    assert config.per_device_train_batch_size == 4


def test_full_config():
    config = FullConfig()
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.lora, LoRAConfig)
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.training, TrainingConfig)


def test_custom_model_config():
    config = ModelConfig(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_4bit=False)
    assert config.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert config.use_4bit is False


def test_lora_config_custom():
    config = LoRAConfig(r=64, lora_alpha=128)
    assert config.r == 64
    assert config.lora_alpha == 128
