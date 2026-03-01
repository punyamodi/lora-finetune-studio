import gradio as gr
import pandas as pd
from pathlib import Path
from typing import Optional


def build_dataset_tab():
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Dataset")
            file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
            url_input = gr.Textbox(label="Or enter Hugging Face Dataset ID", placeholder="username/dataset-name")
            load_btn = gr.Button("Load Dataset", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Dataset Preview")
            stats_display = gr.JSON(label="Dataset Statistics")
            preview_df = gr.Dataframe(label="Preview (first 10 rows)", interactive=False)
            column_selector = gr.Dropdown(label="Text Column", choices=[], interactive=True)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Column Mapping")
            with gr.Row():
                prompt_col = gr.Dropdown(label="Prompt Column (optional)", choices=[], interactive=True)
                response_col = gr.Dropdown(label="Response Column (optional)", choices=[], interactive=True)
            template_choice = gr.Radio(
                label="Prompt Template",
                choices=["raw", "instruction", "chat", "alpaca"],
                value="instruction",
            )
            preview_formatted_btn = gr.Button("Preview Formatted Sample")
            formatted_preview = gr.Textbox(label="Formatted Sample", lines=8, interactive=False)

    def load_dataset_file(file, hf_dataset_id):
        if file is None and not hf_dataset_id:
            return {}, pd.DataFrame(), [], [], []

        if file is not None:
            file_path = file.name
        else:
            return {}, pd.DataFrame(), [], [], []

        try:
            df = pd.read_csv(file_path)
            stats = {
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            }
            preview = df.head(10)
            cols = df.columns.tolist()
            return stats, preview, gr.update(choices=cols, value=cols[0] if cols else None), gr.update(choices=[""] + cols), gr.update(choices=[""] + cols)
        except Exception as e:
            return {"error": str(e)}, pd.DataFrame(), [], [], []

    def preview_formatted(file, text_col, prompt_col_val, response_col_val, template):
        if file is None:
            return "No dataset loaded."
        try:
            df = pd.read_csv(file.name)
            if len(df) == 0:
                return "Empty dataset."
            sample = df.iloc[0]

            from lorakit.data.preprocessor import PROMPT_TEMPLATES
            template_str = PROMPT_TEMPLATES.get(template, PROMPT_TEMPLATES["raw"])

            if template == "raw" or (not prompt_col_val and not response_col_val):
                text = sample.get(text_col, "") if text_col else str(sample.iloc[0])
            else:
                instruction = sample.get(prompt_col_val, "") if prompt_col_val else ""
                response = sample.get(response_col_val, "") if response_col_val else ""
                text = template_str.format(instruction=instruction, response=response)
            return text
        except Exception as e:
            return f"Error: {e}"

    load_btn.click(
        load_dataset_file,
        inputs=[file_input, url_input],
        outputs=[stats_display, preview_df, column_selector, prompt_col, response_col],
    )

    preview_formatted_btn.click(
        preview_formatted,
        inputs=[file_input, column_selector, prompt_col, response_col, template_choice],
        outputs=[formatted_preview],
    )

    return file_input, column_selector, prompt_col, response_col, template_choice
