import gradio as gr
import threading
import pandas as pd
from typing import Optional


_training_logs = []
_training_active = False


def build_training_tab():
    global _training_logs, _training_active

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Model Configuration")
            model_dropdown = gr.Dropdown(
                label="Base Model",
                choices=[
                    "mistralai/Mistral-7B-v0.1",
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "microsoft/phi-2",
                    "meta-llama/Llama-2-7b-hf",
                    "meta-llama/Llama-2-7b-chat-hf",
                    "meta-llama/Meta-Llama-3-8B",
                    "google/gemma-2b",
                ],
                value="mistralai/Mistral-7B-v0.1",
            )
            with gr.Row():
                use_4bit = gr.Checkbox(label="4-bit Quantization", value=True)
                use_8bit = gr.Checkbox(label="8-bit Quantization", value=False)

            gr.Markdown("### LoRA Configuration")
            with gr.Row():
                lora_r = gr.Slider(label="LoRA Rank (r)", minimum=4, maximum=128, value=16, step=4)
                lora_alpha = gr.Slider(label="LoRA Alpha", minimum=8, maximum=256, value=32, step=8)
            lora_dropout = gr.Slider(label="LoRA Dropout", minimum=0.0, maximum=0.5, value=0.05, step=0.01)
            target_modules = gr.Textbox(
                label="Target Modules (comma separated)",
                value="q_proj,k_proj,v_proj,o_proj",
            )

        with gr.Column(scale=1):
            gr.Markdown("### Training Configuration")
            with gr.Row():
                epochs = gr.Slider(label="Epochs", minimum=1, maximum=20, value=3, step=1)
                batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=32, value=4, step=1)
            with gr.Row():
                learning_rate = gr.Number(label="Learning Rate", value=2e-4, precision=6)
                grad_accum = gr.Slider(label="Gradient Accumulation Steps", minimum=1, maximum=16, value=1, step=1)
            warmup_ratio = gr.Slider(label="Warmup Ratio", minimum=0.0, maximum=0.3, value=0.03, step=0.01)
            max_seq_len = gr.Slider(label="Max Sequence Length", minimum=128, maximum=4096, value=512, step=128)
            output_dir = gr.Textbox(label="Output Directory", value="./outputs")

            gr.Markdown("### Dataset")
            dataset_file = gr.File(label="Training CSV", file_types=[".csv"])
            template = gr.Dropdown(
                label="Prompt Template",
                choices=["raw", "instruction", "chat", "alpaca"],
                value="instruction",
            )
            with gr.Row():
                prompt_col = gr.Textbox(label="Prompt Column", value="", placeholder="instruction")
                response_col = gr.Textbox(label="Response Column", value="", placeholder="output")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Hugging Face Hub (optional)")
            hf_token = gr.Textbox(label="HF Token", type="password", placeholder="hf_...")
            hub_model_id = gr.Textbox(label="Hub Model ID", placeholder="username/model-name")
            push_to_hub = gr.Checkbox(label="Push to Hub after training", value=False)

        with gr.Column(scale=1):
            gr.Markdown("### Actions")
            with gr.Row():
                start_btn = gr.Button("Start Training", variant="primary", scale=2)
                stop_btn = gr.Button("Stop", variant="stop", scale=1)
            training_status = gr.Textbox(label="Status", value="Ready", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Training Progress")
            progress_df = gr.Dataframe(
                label="Training Logs",
                headers=["Step", "Epoch", "Loss", "Learning Rate"],
                interactive=False,
            )
            loss_plot = gr.LinePlot(
                label="Training Loss",
                x="Step",
                y="Loss",
                height=300,
            )

    with gr.Row():
        training_result = gr.JSON(label="Training Summary")

    def start_training(
        model_name, use_4bit_val, use_8bit_val,
        lora_r_val, lora_alpha_val, lora_dropout_val, target_modules_val,
        epochs_val, batch_size_val, lr_val, grad_accum_val, warmup_val,
        max_seq_val, output_dir_val,
        dataset_file_val, template_val, prompt_col_val, response_col_val,
        hf_token_val, hub_model_id_val, push_hub_val,
    ):
        global _training_logs, _training_active

        if dataset_file_val is None:
            yield "No dataset provided.", pd.DataFrame(), pd.DataFrame(), {}
            return

        _training_logs = []
        _training_active = True

        try:
            from lorakit.config import FullConfig, ModelConfig, LoRAConfig, DataConfig, TrainingConfig
            from lorakit.data.dataset import DatasetLoader
            from lorakit.training.trainer import LoRATrainer

            modules = [m.strip() for m in target_modules_val.split(",") if m.strip()]

            config = FullConfig(
                model=ModelConfig(
                    model_name=model_name,
                    use_4bit=use_4bit_val,
                    use_8bit=use_8bit_val,
                ),
                lora=LoRAConfig(
                    r=int(lora_r_val),
                    lora_alpha=int(lora_alpha_val),
                    lora_dropout=lora_dropout_val,
                    target_modules=modules,
                ),
                data=DataConfig(
                    text_column="text",
                    prompt_column=prompt_col_val if prompt_col_val else None,
                    response_column=response_col_val if response_col_val else None,
                    max_seq_length=int(max_seq_val),
                ),
                training=TrainingConfig(
                    output_dir=output_dir_val,
                    num_train_epochs=int(epochs_val),
                    per_device_train_batch_size=int(batch_size_val),
                    learning_rate=float(lr_val),
                    gradient_accumulation_steps=int(grad_accum_val),
                    warmup_ratio=warmup_val,
                    push_to_hub=push_hub_val,
                    hub_model_id=hub_model_id_val if hub_model_id_val else None,
                    hub_token=hf_token_val if hf_token_val else None,
                ),
            )

            loader = DatasetLoader(config.data)
            dataset = loader.load(file_path=dataset_file_val.name)

            logs_df = pd.DataFrame(columns=["Step", "Epoch", "Loss", "Learning Rate"])
            loss_data = pd.DataFrame(columns=["Step", "Loss"])

            yield "Training started...", logs_df, loss_data, {}

            trainer = LoRATrainer(config)

            def on_log(entry):
                _training_logs.append(entry)

            result = trainer.train(dataset, prompt_template=template_val, on_log=on_log)

            rows = []
            loss_rows = []
            for entry in _training_logs:
                row = {
                    "Step": entry.get("step", 0),
                    "Epoch": entry.get("epoch", 0),
                    "Loss": entry.get("loss", entry.get("train_loss", "")),
                    "Learning Rate": entry.get("learning_rate", ""),
                }
                rows.append(row)
                if "loss" in entry or "train_loss" in entry:
                    loss_rows.append({"Step": row["Step"], "Loss": row["Loss"]})

            logs_df = pd.DataFrame(rows)
            loss_data = pd.DataFrame(loss_rows)

            summary = {
                "train_loss": result.get("train_loss", 0),
                "train_runtime_seconds": result.get("train_runtime", 0),
                "output_dir": result.get("output_dir", ""),
                "total_logs": len(_training_logs),
            }

            _training_active = False
            yield "Training complete.", logs_df, loss_data, summary

        except Exception as e:
            _training_active = False
            yield f"Error: {e}", pd.DataFrame(), pd.DataFrame(), {"error": str(e)}

    start_btn.click(
        start_training,
        inputs=[
            model_dropdown, use_4bit, use_8bit,
            lora_r, lora_alpha, lora_dropout, target_modules,
            epochs, batch_size, learning_rate, grad_accum, warmup_ratio,
            max_seq_len, output_dir,
            dataset_file, template, prompt_col, response_col,
            hf_token, hub_model_id, push_to_hub,
        ],
        outputs=[training_status, progress_df, loss_plot, training_result],
    )
