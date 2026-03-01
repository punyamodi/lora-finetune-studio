import gradio as gr
from .components.dataset_tab import build_dataset_tab
from .components.training_tab import build_training_tab
from .components.inference_tab import build_inference_tab
from .components.models_tab import build_models_tab


CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
}
.tab-nav button {
    font-size: 16px !important;
    font-weight: 600 !important;
}
#header {
    text-align: center;
    padding: 20px 0 10px;
}
#header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1a1a2e;
}
#header p {
    font-size: 1rem;
    color: #555;
}
.metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="LoRA Finetune Studio",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
        css=CUSTOM_CSS,
    ) as app:
        gr.HTML(
            """
            <div id="header">
                <h1>LoRA Finetune Studio</h1>
                <p>Fine-tune large language models with LoRA and QLoRA. Upload your dataset, configure training, and deploy to Hugging Face Hub.</p>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("Dataset"):
                build_dataset_tab()
            with gr.Tab("Training"):
                build_training_tab()
            with gr.Tab("Inference"):
                build_inference_tab()
            with gr.Tab("Models"):
                build_models_tab()

    return app


def main():
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
    )


if __name__ == "__main__":
    main()
