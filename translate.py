# Handling multilingual text-to-text translation using Hugging Face MarianMT

from transformers import MarianMTModel, MarianTokenizer, pipeline
from typing import Optional
from transformers import AutoConfig

# Checking if a MarianMT model exists by trying to load the config
def model_exists(src: str, tgt: str) -> bool:
    """
    Checking if a MarianMT model exists for the given language pair.
    """
    model_id = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    try:
        _ = AutoConfig.from_pretrained(model_id)
        return True
    except Exception:
        return False


# Creating a dictionary to cache loaded translation pipelines
translation_cache = {}

# Defining a helper function to normalise input text before translation
def normalise_text(text: str) -> str:
    """
    Adding basic punctuation to short input if missing,
    so that MarianMT can produce more reliable output.
    """
    import re

    # Stripping leading and trailing whitespace
    text = text.strip()

    # Adding a period if the sentence ends without punctuation
    if text and text[-1].isalnum():
        text += "."

    # Replacing excessive spacing with single space
    text = re.sub(r"\s+", " ", text)

    return text


def get_translation_pipeline(src_lang: str, tgt_lang: str):
    """
    Loading and caching a MarianMT translation pipeline based on language pair.

    Returns:
        pipeline: A Hugging Face translation pipeline
    """
    model_id = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

    # Returning cached pipeline if it already exists
    if model_id in translation_cache:
        return translation_cache[model_id]

    # Loading the model
    try:
        print(f"Loading MarianMT model: {model_id}")
        # Creating the translation pipeline
        pipe = pipeline("translation", model=model_id, tokenizer=model_id)
        translation_cache[model_id] = pipe
        return pipe
    except Exception as e:
        raise ValueError(f"Failed to load model '{model_id}': {str(e)}")

def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translating text between any two supported languages using MarianMT.
    Falls back to pivoting through English if direct pair is unavailable.
    """
    if not text.strip():
        return ""

    # Normalising the input text
    text = normalise_text(text)

    # Case A – direct model exists
    if model_exists(src_lang, tgt_lang):
        translator = get_translation_pipeline(src_lang, tgt_lang)
        result = translator(text, max_length=512)
        return result[0]["translation_text"]

    # Case B – pivot through English
    print(f"No direct model for {src_lang} -> {tgt_lang}. Pivoting through English...")

    if not model_exists(src_lang, "en") or not model_exists("en", tgt_lang):
        if not model_exists(src_lang, "uk") and not model_exists("uk", tgt_lang):
            raise ValueError(f"No supported pivot route available for {src_lang} -> {tgt_lang}")
        else:
            # First hop: source to Ukrainian
            first_stage = get_translation_pipeline(src_lang, "uk")
            interim = first_stage(text, max_length=512)[0]["translation_text"]

            # Second hop: Ukrainian to target
            second_stage = get_translation_pipeline("uk", tgt_lang)
            final = second_stage(interim, max_length=512)[0]["translation_text"]
            return final
    else:
        # First hop: source to English
        first_stage = get_translation_pipeline(src_lang, "en")
        interim = first_stage(text, max_length=512)[0]["translation_text"]

        # Second hop: English to target
        second_stage = get_translation_pipeline("en", tgt_lang)
        final = second_stage(interim, max_length=512)[0]["translation_text"]

        return final

