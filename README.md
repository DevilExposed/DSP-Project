# Multilingual Speech-Based Translator

A desktop application that allows users to speak in one language and receive translated text in another language using fine-tuned Whisper and MarianMT models.


## Features

- Speech-to-text using a fine-tuned Whisper model
- Text-to-text translation using MarianMT (with fallback via English)
- Manual text input for typed translation
- Responsive GUI built with Tkinter
- Baseline vs fine-tuned WER evaluation scripts


## Install dependencies:

datasets<br>
playsound==1.2.2<br>
transformers>=4.35<br>
torch>=2.0<br>
torchaudio>=2.0<br>
sounddevice<br>
soundfile<br>
jiwer<br>
sacremoses<br>
gTTS<br>


## Configuration

- **Hugging Face API Token:** Required for downloading models. Set as an environment variable.
- **Weights and Biases API Token:** Required for improving the model. When prompted, input "c36bed5f601a686c39565a9197ff4d174b96865f"


## Usage

- Run the main application:
  ```
  python main.py
  ```
  or press

  ctrl+alt+p
- Evaluate ASR:
  ```
  python evaluate_asr.py
  ```
  or press

  ctrl+alt+p
- Run the notebook in Google Colab or Jupyter.


MIT — For educational use as part of the UWE Digital Systems Project.