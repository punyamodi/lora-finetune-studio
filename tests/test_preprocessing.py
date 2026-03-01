import pytest
from src.lorakit.config import DataConfig
from src.lorakit.data.preprocessor import TextPreprocessor, PROMPT_TEMPLATES


def test_prompt_templates_exist():
    assert "instruction" in PROMPT_TEMPLATES
    assert "chat" in PROMPT_TEMPLATES
    assert "alpaca" in PROMPT_TEMPLATES
    assert "raw" in PROMPT_TEMPLATES


def test_formatter_raw():
    config = DataConfig(text_column="text")
    preprocessor = TextPreprocessor(config)
    formatter = preprocessor.get_formatter("raw")
    result = formatter({"text": "hello world"})
    assert result["text"] == "hello world"


def test_formatter_instruction():
    config = DataConfig(prompt_column="instruction", response_column="output")
    preprocessor = TextPreprocessor(config)
    formatter = preprocessor.get_formatter("instruction")
    result = formatter({"instruction": "What is AI?", "output": "AI is..."})
    assert "What is AI?" in result["text"]
    assert "AI is..." in result["text"]
    assert "### Instruction:" in result["text"]


def test_formatter_chat():
    config = DataConfig(prompt_column="question", response_column="answer")
    preprocessor = TextPreprocessor(config)
    formatter = preprocessor.get_formatter("chat")
    result = formatter({"question": "Hello", "answer": "Hi there"})
    assert "[INST]" in result["text"]
    assert "Hello" in result["text"]


def test_formatter_alpaca():
    config = DataConfig(prompt_column="input", response_column="output")
    preprocessor = TextPreprocessor(config)
    formatter = preprocessor.get_formatter("alpaca")
    result = formatter({"input": "Explain LoRA", "output": "LoRA is..."})
    assert "### Human:" in result["text"]
    assert "### Assistant:" in result["text"]
