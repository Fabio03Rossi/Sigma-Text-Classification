import gradio as gr

from zero_shot_classification import classify_text

interface = gr.Interface(
    fn=classify_text,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter text here.")
    ],
    outputs=gr.Label(num_top_classes=3),
    title="Closing Notes Classification"
)

interface.launch()