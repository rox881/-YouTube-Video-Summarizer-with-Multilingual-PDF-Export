

# 🎥 YouTube Video Summarizer with Multilingual PDF Export

This is a Python application built with Streamlit that allows users to generate concise summaries of YouTube videos. The tool can either use existing transcripts or perform AI-based audio transcription. The final summary is refined by Google's Gemini model and can be exported as a neatly formatted PDF, with support for English, Hindi, and Marathi.

## ✨ Features

  * **Video Summarization**: Input any public YouTube video URL to get a summary.
  * **Automatic Transcription**: Automatically fetches existing English transcripts using the `youtube-transcript-api`.
  * **AI Audio Transcription**: If no transcript is available, it downloads the video's audio and uses OpenAI's `Whisper` model to generate a transcript from scratch.
  * **AI-Powered Summarization**: Uses the `distilbart-cnn-12-6` model from the `transformers` library to create a concise initial summary.
  * **Gemini-Powered Refinement**: Leverages the Google Gemini API to structure the summary with clear headings and highlights for better readability.
  * **Multilingual Support**: Provides the final refined summary in English, Hindi, or Marathi, as selected by the user.
  * **PDF Export**: Generates a downloadable PDF of the summary, with proper font support for Devanagari scripts (Hindi/Marathi).
  * **User-Friendly Web UI**: An interactive and easy-to-use web interface built with Streamlit.

## 🛠️ Tech Stack

  * **Web Framework**: Streamlit
  * **AI & Machine Learning**:
      * `Google Generative AI` (for summary refinement)
      * `OpenAI Whisper` (for audio transcription)
      * `Hugging Face Transformers` (for summarization)
      * `PyTorch` (as the backend for ML models)
  * **YouTube Interaction**: `yt-dlp`, `youtube-transcript-api`
  * **PDF Generation**: `fpdf2`, `python-bidi`

## ⚙️ Setup and Installation

### 1\. Prerequisites

Before you begin, ensure you have the following installed on your system:

  * **Python 3.8+**
  * **FFmpeg**: Required by `Whisper` for audio processing. Download it from the [official FFmpeg website](https://ffmpeg.org/download.html) and ensure it's accessible from your system's PATH.
  * **Noto Sans Devanagari Font**: Required for generating PDFs in Hindi or Marathi. You can download and install the font from [Google Fonts](https://fonts.google.com/noto/specimen/Noto+Sans+Devanagari).

### 2\. Installation Steps

1.  **Clone the Repository (or download the files):**

    ```sh
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and Activate a Virtual Environment:**

    ```sh
    # For Windows
    python -m venv .venv
    .venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install PyTorch:**
    Install `torch` separately based on your system's hardware for the best performance.

      * **CPU-Only:**
        ```sh
        pip install torch torchvision torchaudio
        ```
      * **NVIDIA GPU (CUDA):**
        Visit the [PyTorch website](https://pytorch.org/get-started/locally/) to get the specific command for your version of CUDA.

4.  **Install All Other Dependencies:**
    Use the provided `requirements.txt` file to install the remaining packages.

    ```sh
    pip install -r requirements.txt
    ```

### 3\. Configuration

The application requires a **Google Gemini API Key** to function.

1.  Obtain your API key from the [Google AI Studio](https://makersuite.google.com/app/apikey).
2.  Locate the following line in the `ui.py` script:
    ```python
    genai.configure(api_key="AIzaSyBm8KxuZjSA2yUfjCEnPwM5CXKmJQCtBAM")
    ```
3.  Replace the placeholder key with your actual Gemini API key.

> **Security Warning**: For production or shared environments, it is highly recommended to use environment variables to store your API key instead of hardcoding it directly in the script.

## 🚀 How to Run the Application

After completing the setup and configuration, run the following command in your terminal:

```sh
streamlit run ui.py
```

Your web browser will open a new tab with the application running.

### Usage

1.  Paste the full URL of a YouTube video into the input box.
2.  Select the desired language for the final summary (English, Hindi, or Marathi).
3.  Click the **"Summarize the Video"** button.
4.  Wait for the processing to complete. The refined summary will be displayed on the page.
5.  Click the **"Download PDF"** button to save the summary as a PDF file.

## flowchart TD

```
A[User Inputs YouTube URL] --> B{Get Transcript};
B -->|Subtitles Exist| C[Use YouTubeTranscriptApi];
B -->|No Subtitles| D[Download Audio with yt-dlp];
D --> E[Transcribe with Whisper AI];
C --> F[Full Transcript];
E --> F;
F --> G[Summarize Text with Transformers];
G --> H[Refine Summary with Gemini API];
H --> I[Display in Streamlit UI];
I --> J[Generate & Download PDF];
```

## ⚠️ Important Notes

  * **First-Time Run**: The first time you run the application, it will need to download the `Whisper` and `distilbart` models, which may take some time depending on your internet connection. These models will be cached for subsequent runs.
  * **API Usage**: This application makes calls to the Google Gemini API, which may be subject to usage quotas and costs based on your Google AI Platform account.
