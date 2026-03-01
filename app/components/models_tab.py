import gradio as gr
import json
from pathlib import Path


def build_models_tab():
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Local Models")
            models_dir = gr.Textbox(label="Models Directory", value="./outputs")
            refresh_btn = gr.Button("Refresh Model List")
            models_list = gr.JSON(label="Local Models")

        with gr.Column(scale=1):
            gr.Markdown("### Model Details")
            model_path_input = gr.Textbox(label="Model Path", placeholder="./outputs/my-model")
            inspect_btn = gr.Button("Inspect Model")
            model_details = gr.JSON(label="Model Details")
            delete_btn = gr.Button("Delete Model", variant="stop")
            delete_status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Push to Hugging Face Hub")
            with gr.Row():
                push_model_path = gr.Textbox(label="Local Model Path", placeholder="./outputs/my-model")
                push_repo_id = gr.Textbox(label="Hub Repository ID", placeholder="username/model-name")
            push_token = gr.Textbox(label="HF Write Token", type="password")
            push_private = gr.Checkbox(label="Private Repository", value=False)
            push_btn = gr.Button("Push to Hub", variant="primary")
            push_status = gr.Textbox(label="Push Status", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Download from Hugging Face Hub")
            with gr.Row():
                download_repo = gr.Textbox(label="Repository ID", placeholder="username/model-name")
                download_dir = gr.Textbox(label="Save Directory", value="./models")
            download_token = gr.Textbox(label="HF Token (optional)", type="password")
            download_btn = gr.Button("Download Model")
            download_status = gr.Textbox(label="Download Status", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Supported Base Models")
            from lorakit.models.manager import SUPPORTED_BASE_MODELS
            models_table_data = [[name, model_id] for name, model_id in SUPPORTED_BASE_MODELS.items()]
            gr.Dataframe(
                value=models_table_data,
                headers=["Model Name", "Hub ID"],
                interactive=False,
                label="Available Base Models",
            )

    def refresh_models(directory):
        try:
            from lorakit.models.manager import ModelManager
            manager = ModelManager(models_dir=directory)
            return manager.list_local_models()
        except Exception as e:
            return {"error": str(e)}

    def inspect_model(path):
        try:
            from lorakit.models.manager import ModelManager
            manager = ModelManager()
            return manager.get_model_info(path)
        except Exception as e:
            return {"error": str(e)}

    def delete_model(path):
        try:
            from lorakit.models.manager import ModelManager
            manager = ModelManager()
            manager.delete_model(path)
            return f"Deleted: {path}"
        except Exception as e:
            return f"Error: {e}"

    def push_model(path, repo_id, token, private):
        try:
            from lorakit.models.manager import ModelManager
            manager = ModelManager()
            manager.push_to_hub(path, repo_id, token, private)
            return f"Successfully pushed to {repo_id}"
        except Exception as e:
            return f"Error: {e}"

    def download_model(repo_id, save_dir, token):
        try:
            from lorakit.models.manager import ModelManager
            manager = ModelManager(models_dir=save_dir)
            dest = manager.download_adapter(repo_id, token=token if token else None)
            return f"Downloaded to: {dest}"
        except Exception as e:
            return f"Error: {e}"

    refresh_btn.click(refresh_models, inputs=[models_dir], outputs=[models_list])
    inspect_btn.click(inspect_model, inputs=[model_path_input], outputs=[model_details])
    delete_btn.click(delete_model, inputs=[model_path_input], outputs=[delete_status])
    push_btn.click(push_model, inputs=[push_model_path, push_repo_id, push_token, push_private], outputs=[push_status])
    download_btn.click(download_model, inputs=[download_repo, download_dir, download_token], outputs=[download_status])
