from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    use_4bit: bool = True
    use_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    trust_remote_code: bool = False
    torch_dtype: str = "float16"


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DataConfig:
    train_file: str = "train.csv"
    text_column: str = "text"
    prompt_column: Optional[str] = None
    response_column: Optional[str] = None
    max_seq_length: int = 512
    val_split: float = 0.1
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    optimizer: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    max_steps: int = -1
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    fp16: bool = True
    bf16: bool = False
    packing: bool = False
    report_to: str = "none"
    seed: int = 42


@dataclass
class FullConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
