import os
from typing import Optional, Callable
from pathlib import Path

import torch
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from ..config import FullConfig, ModelConfig, LoRAConfig, TrainingConfig, DataConfig
from ..data.dataset import DatasetLoader
from ..data.preprocessor import TextPreprocessor
from .callbacks import ProgressCallback


class LoRATrainer:
    def __init__(self, config: FullConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _build_bnb_config(self) -> Optional[BitsAndBytesConfig]:
        mc = self.config.model
        if not mc.use_4bit and not mc.use_8bit:
            return None
        compute_dtype = getattr(torch, mc.bnb_4bit_compute_dtype)
        return BitsAndBytesConfig(
            load_in_4bit=mc.use_4bit,
            load_in_8bit=mc.use_8bit,
            bnb_4bit_quant_type=mc.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=mc.use_nested_quant,
        )

    def _load_model_and_tokenizer(self):
        mc = self.config.model
        bnb_config = self._build_bnb_config()
        torch_dtype = getattr(torch, mc.torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            mc.model_name, trust_remote_code=mc.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            mc.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=mc.trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

    def _apply_lora(self):
        lc = self.config.lora
        if self.config.model.use_4bit or self.config.model.use_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        peft_config = LoraConfig(
            r=lc.r,
            lora_alpha=lc.lora_alpha,
            target_modules=lc.target_modules,
            lora_dropout=lc.lora_dropout,
            bias=lc.bias,
            task_type=lc.task_type,
        )
        self.model = get_peft_model(self.model, peft_config)

    def _build_training_args(self) -> TrainingArguments:
        tc = self.config.training
        return TrainingArguments(
            output_dir=tc.output_dir,
            num_train_epochs=tc.num_train_epochs,
            per_device_train_batch_size=tc.per_device_train_batch_size,
            per_device_eval_batch_size=tc.per_device_eval_batch_size,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            learning_rate=tc.learning_rate,
            weight_decay=tc.weight_decay,
            optim=tc.optimizer,
            lr_scheduler_type=tc.lr_scheduler_type,
            max_grad_norm=tc.max_grad_norm,
            warmup_ratio=tc.warmup_ratio,
            save_steps=tc.save_steps,
            eval_steps=tc.eval_steps,
            logging_steps=tc.logging_steps,
            max_steps=tc.max_steps,
            fp16=tc.fp16,
            bf16=tc.bf16,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            report_to=tc.report_to,
            seed=tc.seed,
            push_to_hub=tc.push_to_hub,
            hub_model_id=tc.hub_model_id,
            hub_token=tc.hub_token,
        )

    def train(
        self,
        dataset: DatasetDict,
        prompt_template: str = "instruction",
        on_log: Optional[Callable] = None,
    ) -> dict:
        self._load_model_and_tokenizer()
        self._apply_lora()

        preprocessor = TextPreprocessor(self.config.data, self.tokenizer)
        formatter = preprocessor.get_formatter(prompt_template)
        formatted = dataset.map(formatter, batched=False)

        training_args = self._build_training_args()
        callback = ProgressCallback(on_log=on_log)

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=formatted["train"],
            eval_dataset=formatted.get("validation"),
            dataset_text_field="text",
            max_seq_length=self.config.data.max_seq_length,
            tokenizer=self.tokenizer,
            args=training_args,
            packing=self.config.training.packing,
            callbacks=[callback],
        )

        result = self.trainer.train()

        output = {
            "train_runtime": result.metrics.get("train_runtime", 0),
            "train_loss": result.metrics.get("train_loss", 0),
            "train_samples_per_second": result.metrics.get("train_samples_per_second", 0),
            "logs": callback.get_logs(),
            "output_dir": self.config.training.output_dir,
        }

        self.trainer.save_model()

        if self.config.training.push_to_hub and self.config.training.hub_model_id:
            self.trainer.push_to_hub()

        return output

    def get_trainable_params(self) -> dict:
        if self.model is None:
            return {}
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_percent": round(100 * trainable / total, 4),
        }
