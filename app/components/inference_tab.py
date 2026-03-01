import gradio as gr
from pathlib import Path
import os


_generator = None


def build_inference_tab():
    global _generator

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Load Model")
            model_source = gr.Radio(
                label="Model Source",
                choices=["Local Fine-tuned Model", "Hugging Face Hub"],
                value="Local Fine-tuned Model",
            )
            with gr.Group() as local_group:
                model_path = gr.Textbox(label="Model Path", placeholder="./outputs")
                base_model_path = gr.Textbox(
                    label="Base Model (if LoRA adapter)",
                    placeholder="mistralai/Mistral-7B-v0.1",
                )
            with gr.Group() as hub_group:
                hub_model_input = gr.Textbox(label="Hugging Face Model ID", placeholder="username/model-name")
                hf_token_infer = gr.Textbox(label="HF Token (for private models)", type="password")

            load_4bit = gr.Checkbox(label="Load in 4-bit", value=False)
            load_btn = gr.Button("Load Model", variant="primary")
            model_status = gr.Textbox(label="Status", value="No model loaded", interactive=False)
            unload_btn = gr.Button("Unload Model")

        with gr.Column(scale=2):
            gr.Markdown("### Generate Text")
            prompt_input = gr.Textbox(
                label="Prompt",
                lines=5,
                placeholder="Enter your prompt here...",
            )

            with gr.Accordion("Generation Parameters", open=False):
                with gr.Row():
                    max_new_tokens = gr.Slider(label="Max New Tokens", minimum=10, maximum=2048, value=256, step=10)
                    temperature = gr.Slider(label="Temperature", minimum=0.01, maximum=2.0, value=0.7, step=0.01)
                with gr.Row():
                    top_p = gr.Slider(label="Top P", minimum=0.1, maximum=1.0, value=0.9, step=0.01)
                    top_k = gr.Slider(label="Top K", minimum=1, maximum=200, value=50, step=1)
                with gr.Row():
                    rep_penalty = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=2.0, value=1.1, step=0.05)
                    num_sequences = gr.Slider(label="Number of Sequences", minimum=1, maximum=5, value=1, step=1)

            generate_btn = gr.Button("Generate", variant="primary")
            output_display = gr.Textbox(label="Generated Output", lines=10, interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Batch Inference")
            batch_file = gr.File(label="Upload CSV with prompts (column: 'prompt')", file_types=[".csv"])
            batch_generate_btn = gr.Button("Run Batch Inference")
            batch_output = gr.Dataframe(label="Batch Results", interactive=False)
            batch_download = gr.File(label="Download Results")

    def load_model(source, local_path, base_model, hub_id, token, four_bit):
        global _generator
        try:
            from lorakit.inference.generator import TextGenerator
            if _generator is not None:
                _generator.unload()

            if source == "Local Fine-tuned Model":
                path = local_path.strip()
                base = base_model.strip() if base_model.strip() else None
            else:
                path = hub_id.strip()
                base = None

            _generator = TextGenerator(model_path=path, base_model=base, load_in_4bit=four_bit)
            _generator.load()
            return "Model loaded successfully."
        except Exception as e:
            return f"Error loading model: {e}"

    def unload_model():
        global _generator
        if _generator is not None:
            _generator.unload()
            _generator = None
        return "Model unloaded."

    def generate_text(prompt, max_tok, temp, tp, tk, rep, num_seq):
        global _generator
        if _generator is None:
            return "No model loaded. Please load a model first."
        try:
            outputs = _generator.generate(
                prompt=prompt,
                max_new_tokens=int(max_tok),
                temperature=float(temp),
                top_p=float(tp),
                top_k=int(tk),
                repetition_penalty=float(rep),
                num_return_sequences=int(num_seq),
            )
            return "\n\n---\n\n".join(outputs)
        except Exception as e:
            return f"Generation error: {e}"

    def run_batch_inference(file, max_tok, temp, tp, tk, rep):
        global _generator
        if file is None:
            return [], None
        if _generator is None:
            return [], None
        try:
            import pandas as pd
            df = pd.read_csv(file.name)
            if "prompt" not in df.columns:
                return [], None
            results = []
            for prompt in df["prompt"].tolist():
                output = _generator.generate(
                    prompt=str(prompt),
                    max_new_tokens=int(max_tok),
                    temperature=float(temp),
                    top_p=float(tp),
                    top_k=int(tk),
                    repetition_penalty=float(rep),
                )
                results.append({"prompt": prompt, "output": output[0] if output else ""})
            result_df = pd.DataFrame(results)
            out_path = "/tmp/batch_results.csv"
            result_df.to_csv(out_path, index=False)
            return result_df, out_path
        except Exception as e:
            return [], None

    load_btn.click(
        load_model,
        inputs=[model_source, model_path, base_model_path, hub_model_input, hf_token_infer, load_4bit],
        outputs=[model_status],
    )

    unload_btn.click(unload_model, outputs=[model_status])

    generate_btn.click(
        generate_text,
        inputs=[prompt_input, max_new_tokens, temperature, top_p, top_k, rep_penalty, num_sequences],
        outputs=[output_display],
    )

    batch_generate_btn.click(
        run_batch_inference,
        inputs=[batch_file, max_new_tokens, temperature, top_p, top_k, rep_penalty],
        outputs=[batch_output, batch_download],
    )
