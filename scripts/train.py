import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lorakit.config import FullConfig, ModelConfig, LoRAConfig, DataConfig, TrainingConfig
from lorakit.data.dataset import DatasetLoader
from lorakit.training.trainer import LoRATrainer
from lorakit.utils.logging import get_logger
from lorakit.utils.metrics import format_training_summary

logger = get_logger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an LLM with LoRA")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--data", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--output", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--use-4bit", action="store_true", default=True)
    parser.add_argument("--use-8bit", action="store_true", default=False)
    parser.add_argument("--template", type=str, default="instruction", choices=["raw", "instruction", "chat", "alpaca"])
    parser.add_argument("--prompt-col", type=str, default=None)
    parser.add_argument("--response-col", type=str, default=None)
    parser.add_argument("--push-to-hub", action="store_true", default=False)
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--hub-token", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = FullConfig(**config_dict)
    else:
        config = FullConfig(
            model=ModelConfig(model_name=args.model, use_4bit=args.use_4bit, use_8bit=args.use_8bit),
            lora=LoRAConfig(r=args.lora_r, lora_alpha=args.lora_alpha),
            data=DataConfig(
                max_seq_length=args.max_seq_len,
                prompt_column=args.prompt_col,
                response_column=args.response_col,
            ),
            training=TrainingConfig(
                output_dir=args.output,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                learning_rate=args.lr,
                push_to_hub=args.push_to_hub,
                hub_model_id=args.hub_model_id,
                hub_token=args.hub_token,
            ),
        )

    logger.info(f"Loading dataset from {args.data}")
    loader = DatasetLoader(config.data)
    dataset = loader.load(file_path=args.data)
    stats = loader.get_dataset_stats(dataset)
    logger.info(f"Dataset stats: {stats}")

    logger.info(f"Starting training with model: {config.model.model_name}")
    trainer = LoRATrainer(config)

    def on_log(entry):
        logger.info(f"Step {entry.get('step', 0)}: {entry}")

    result = trainer.train(dataset, prompt_template=args.template, on_log=on_log)

    summary = format_training_summary(result)
    logger.info(f"Training complete.\n{summary}")
    logger.info(f"Model saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
