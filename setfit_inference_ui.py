import gradio as gr

from setfit_inference_test import execute_prediction

interface = gr.Interface(
    fn=execute_prediction,
    inputs=[
        gr.Textbox(lines=5, placeholder="Inserisci qui le stringhe."),
    ],
    outputs=[
        gr.Label(label="Prediction Model:"),
        gr.Label(label="Prediction Model Probability:"),
        #gr.Label(label="Prediction Model 3:")
    ],
    title="Closing Notes Classification"
)

interface.launch()