# handling transcription using the fine-tuned Whisper model

# importing required libraries
from pathlib import Path
import torch, torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# defining the device (GPU if available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# defining the path to the fine-tuned model directory
MODEL_DIR = Path(__file__).parent / "models" / "whisperFineTune"

# loading the fine-tuned Whisper model and sending it to the appropriate device
print(f"Loading model from: {MODEL_DIR}")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)

# loading the processor which handles feature extraction and decoding
processor = WhisperProcessor.from_pretrained(MODEL_DIR)

# defining the main function to transcribe a .wav file
def transcribe(wav_path: Path, language: str | None = None) -> str:
    """
    Transcribes a 16 kHz mono WAV file using the fine-tuned Whisper model.

    Parameters:
    - wav_path: Path to the .wav file
    - language: ISO 639-1 code for language (e.g. "en", "bg")

    Returns:
    - str: the transcribed text
    """
    print(f"Transcribing file: {wav_path}")

    # loading and processing the audio input
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != 16000:
        print(f"Resampling from {sample_rate} Hz to 16000 Hz")
        waveform = torch.nn.functional.interpolate(waveform.unsqueeze(0), size=(1, int(waveform.size(1) * 16000 / sample_rate)), mode='linear').squeeze(0)

    # preparing input features using the processor
    print("Extracting input features...")
    inputs = processor(
        waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
    )

    # sending features to the same device as the model
    input_features = inputs.input_features.to(DEVICE)

    # preparing forced decoder ids if language is specified
    print(f"Setting language: {language}")
    decoder_prompt = processor.get_decoder_prompt_ids(language=language, task="transcribe") if language else None

    # generating the output ids using the model
    print("Generating transcription...")
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=decoder_prompt
        )

    # decoding the output ids into human-readable text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print(f"Transcription result: {transcription}")
    return transcription.strip()
