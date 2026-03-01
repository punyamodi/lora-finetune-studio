from .config import TrainingConfig, ModelConfig, LoRAConfig, DataConfig

__version__ = "1.0.0"
__all__ = [
    "TrainingConfig",
    "ModelConfig",
    "LoRAConfig",
    "DataConfig",
]


def __getattr__(name):
    if name == "LoRATrainer":
        from .training.trainer import LoRATrainer
        return LoRATrainer
    if name == "TextGenerator":
        from .inference.generator import TextGenerator
        return TextGenerator
    if name == "ModelManager":
        from .models.manager import ModelManager
        return ModelManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
