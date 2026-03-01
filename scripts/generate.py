import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lorakit.inference.generator import TextGenerator
from lorakit.utils.logging import get_logger

logger = get_logger("generate")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned model")
    parser.add_argument("--model", type=str, required=True, help="Path to fine-tuned model or HF model ID")
    parser.add_argument("--base-model", type=str, default=None, help="Base model if using LoRA adapter")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to generate from")
    parser.add_argument("--prompt-file", type=str, default=None, help="File with one prompt per line")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-sequences", type=int, default=1)
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Loading model from {args.model}")
    generator = TextGenerator(
        model_path=args.model,
        base_model=args.base_model,
        load_in_4bit=args.load_in_4bit,
    )
    generator.load()

    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        logger.error("Provide --prompt or --prompt-file")
        sys.exit(1)

    all_results = []
    for i, prompt in enumerate(prompts):
        logger.info(f"Generating for prompt {i + 1}/{len(prompts)}")
        outputs = generator.generate(
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_sequences,
        )
        for j, output in enumerate(outputs):
            result = {"prompt": prompt, "sequence": j + 1, "output": output}
            all_results.append(result)
            print(f"\nPrompt: {prompt}")
            print(f"Output [{j + 1}]: {output}")

    if args.output_file:
        import json
        with open(args.output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")

    generator.unload()


if __name__ == "__main__":
    main()
