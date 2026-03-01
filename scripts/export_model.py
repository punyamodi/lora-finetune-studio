import argparse
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lorakit.utils.logging import get_logger

logger = get_logger("export")


def merge_lora_weights(base_model: str, adapter_path: str, output_path: str, load_in_4bit: bool = False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    logger.info("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging weights...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to {output_path}...")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Merge complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Export and merge LoRA fine-tuned model")
    subparsers = parser.add_subparsers(dest="command")

    merge_parser = subparsers.add_parser("merge", help="Merge LoRA adapter with base model")
    merge_parser.add_argument("--base-model", type=str, required=True)
    merge_parser.add_argument("--adapter", type=str, required=True)
    merge_parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "merge":
        merge_lora_weights(args.base_model, args.adapter, args.output)
    else:
        logger.error("Unknown command. Use 'merge'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
