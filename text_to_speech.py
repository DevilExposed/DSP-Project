# Generating speech from translated text using Google Text-to-Speech (gTTS)
from gtts import gTTS
import os
from pathlib import Path
from playsound import playsound
import tempfile
from uuid import uuid4

# Generating a spoken audio file from text
def text_to_speech(text: str, lang_code: str) -> Path:
    """
    Converting text to speech and saving it as a temporary MP3 file.

    Args:
        text (str): The text to convert
        lang_code (str): ISO 639-1 language code (e.g. 'en', 'it', 'bg')

    Returns:
        Path: Path to the temporary MP3 file
    """
    # Creating the TTS object with target language
    tts = gTTS(text=text, lang=lang_code)

    # Generating a unique name for the file
    unique_name = f"tts_{uuid4().hex}_{lang_code}.mp3"
    
    # Saving to a temporary file
    output_path = Path(tempfile.gettempdir()) / unique_name
    tts.save(output_path.as_posix())

    return output_path

# Playing the generated speech
def play_speech(mp3_path: Path):
    """
    Playing the generated MP3 file.
    """
    playsound(mp3_path.as_posix())
