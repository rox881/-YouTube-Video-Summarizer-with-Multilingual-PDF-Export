# Hindi : https://youtu.be/Ma7G842vHlc?si=-kaX5HxIv7rNfU_4
# Eng Without Subtitle : https://www.youtube.com/watch?v=nT6Be1Bqfoc
# Eng with Subtitle : https://www.youtube.com/watch?v=4s7rlRkwC0U
# 6 tips Ted Talk  https://www.youtube.com/watch?v=eHJnEHyyN1Y
# marathi video : https://www.youtube.com/watch?v=EL9Inf0lnn0
# marathi song : https://www.youtube.com/watch?v=gxcJ8EPrMaE
# .venv\Scripts\activate
import os
import yt_dlp
import whisper
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import torch
from concurrent.futures import ThreadPoolExecutor
from textwrap import wrap
import multiprocessing
import time
import subprocess
import threading
from threading import Event  # For async Whisper loading control
import google.generativeai as genai
genai.configure(api_key="AIzaSyBm8KxuZjSA2yUfjCEnPwM5CXKmJQCtBAM")
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from fpdf import FPDF
from bidi.algorithm import get_display  # For right-to-left languages like Hindi/Marathi

def get_video_title(video_url):
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            return info_dict.get('title', 'Title not found')
    except Exception as e:
        print(f"Error: {e}")
        return None

def refine_text_with_gemini(text,title,language):
    prompt = f"""
    Do the follwoing things on the given content :
    1. Print the Main title heading < {title} >
    2. Understand What is the Content about and what all does it include
    3. Structure the content in headings for better readability, try including all the provided content.
    4. Highlighting key points using **bold** text.
    NOTE* : Do no print any of your replies, direct print the final output in {language}. 


    Text: {text}
    """

    model = genai.GenerativeModel("gemini-2.0-flash")

    response = model.generate_content(prompt)

    if response.candidates:
        return response.candidates[0].content.parts[0].text
    else:
        return text

# Whisper Model Setup with Async Loading
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
model_loaded = Event()  # Event to signal model loading completion

def load_whisper_model():
    global model
    model = whisper.load_model("base", device=device, download_root="./models").to(device)
    if device == "cuda":
        model.half()
    model_loaded.set()  # Signal that model loading is complete

# Start loading Whisper asynchronously
threading.Thread(target=load_whisper_model, daemon=True).start()

# Summarization Pipeline with Batch Processing
summarizer = pipeline('summarization', model="sshleifer/distilbart-cnn-12-6", batch_size=8)

def get_transcript(video_url):
    video_id = video_url.split("v=")[1]

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        result = " ".join([i['text'] for i in transcript])
        return result
    except TranscriptsDisabled:
        print("No subtitles found, using Whisper AI for transcription...")
        return transcribe_audio(download_audio(video_url))

def download_audio(video_url):
    os.system(f'yt-dlp -f "bestaudio" --extract-audio --audio-format mp3 --quiet --output "audio.mp3" {video_url}')
    return "audio.mp3"

def transcribe_audio(audio_path):
    model_loaded.wait()  # Wait for Whisper model to load

    try:
        result = model.transcribe(audio_path, beam_size=1, temperature=0, best_of=1)
        return result["text"]
    except FileNotFoundError:
        print("⚠️ Audio conversion failed. Trying manual conversion via ffmpeg...")
        subprocess.run(['ffmpeg', '-i', audio_path, 'converted_audio.wav'], check=True)
        result = model.transcribe('converted_audio.wav', beam_size=1, temperature=0, best_of=1)
        return result["text"]

def summarize_text(text):
    MAX_TOKENS = 1500  # Reduced for faster chunk processing
    chunks = wrap(text, MAX_TOKENS)

    summaries = summarizer(chunks, max_length=250, min_length=80, do_sample=False)

    return " ".join([summary['summary_text'] for summary in summaries])


if __name__ == "__main__":
    st.title("🎥 YouTube Video Summarizer")

    # Input Box for URL
    video_url = st.text_input("Enter YouTube Video URL:")
    video_url = video_url.strip().replace("\\", "")

    if video_url:
        # Display Video Thumbnail
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            thumbnail_url = info_dict.get('thumbnail')
            if thumbnail_url:
                response = requests.get(thumbnail_url)
                image = Image.open(BytesIO(response.content))
                st.image(image, caption="Video Thumbnail", use_container_width=True)

        # Language Selection
        language = st.selectbox("Select Language for Summary:", ["English", "Hindi", "Marathi"], index=0)

        # Summarization Button
        if st.button("Summarize the Video"):
            start_time = time.time()

            spinner = st.empty()
            spinner.text("Summarizing the video... Please wait.")
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                title = get_video_title(video_url)
                future_transcript = executor.submit(get_transcript, video_url)
                transcript = future_transcript.result()

            if transcript:
                summary = summarize_text(transcript)
                refined_text = refine_text_with_gemini(summary, title, language)

                # Store summarized text in session state
                st.session_state['refined_text'] = refined_text
                st.session_state['title'] = title

                spinner.empty()

        # Show PDF Button Only If Summary Exists
        if 'refined_text' in st.session_state:
            class CustomPDF(FPDF):
                def chapter_title(self, title,language):
                    self.set_font("NotoSansDevanagari", "B", 14)
                    self.cell(0, 10, title, ln=True)
                    self.ln(5)

                def chapter_body(self, body,language):
                    self.set_font("NotoSansDevanagari", "", 12)
                    try:
                        self.multi_cell(0, 10, body.encode('utf-8').decode('utf-8'))
                    except UnicodeEncodeError:
                        self.multi_cell(0, 10, body)  # Fallback if encoding issues persist
                    self.ln()

            def save_to_pdf(text, filename="summary.pdf"):
                pdf = CustomPDF()
                pdf.add_page()

                pdf.add_font('NotoSansDevanagari', '', 'NotoSansDevanagari-Regular.ttf', uni=True)  # Add Unicode Font
                pdf.add_font('NotoSansDevanagari', 'B', 'NotoSansDevanagari-Bold.ttf', uni=True)

                pdf.chapter_title(st.session_state['title'],language)

                formatted_text = text.replace("**", "").replace("#", "")
                pdf.chapter_body(formatted_text,language)
                pdf.output(filename,"F")

            # Display Summary
            if 'refined_text' in st.session_state:
                st.subheader("🔍 Refined Summary")
                st.markdown(st.session_state['refined_text']) 

                # Auto-generate PDF and download directly
                save_to_pdf(st.session_state['refined_text'])
                with open("summary.pdf", "rb") as file:
                    st.download_button(
                        label="📄 Download PDF",
                        data=file,
                        file_name="summary.pdf",
                        mime="application/pdf"
                    )
           