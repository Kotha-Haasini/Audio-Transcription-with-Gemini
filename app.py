import os
import gradio as gr
from google import genai

# Get API key from Render Environment Variables
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=API_KEY)


def transcribe(audio_path):
    if audio_path is None:
        return "Please record some audio."

    try:
        # Upload audio file
        audio_file = client.files.upload(file=audio_path)

        # Generate transcription
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Transcribe this audio exactly as spoken. Do not summarize. Include punctuation.",
                audio_file,
            ],
        )

        return response.text

    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks(title="Gemini Audio Transcription") as demo:
    gr.Markdown("# 🎤 Gemini Audio Transcription")
    gr.Markdown("Record your voice and click **Transcribe**.")

    audio = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Record Your Voice"
    )

    output = gr.Textbox(
        label="Transcript",
        lines=10
    )

    btn = gr.Button("Transcribe")

    btn.click(
        fn=transcribe,
        inputs=audio,
        outputs=output
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
