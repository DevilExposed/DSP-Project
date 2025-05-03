import csv
import pathlib
import jiwer
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torchaudio.transforms as T

print("Script started: Loading necessary libraries...")
# Defining the languages to evaluate
LANGUAGES = ["en", "bg", "uk", "ru", "ar", "it", "pl", "pt"]
# Defining the model name
MODEL_NAME = "openai/whisper-base"
print(f"Using Hugging Face baseline model: {MODEL_NAME}")

# Defining the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# Loading the baseline model & processor
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
print("Model and processor loaded successfully")

# Defining the target sampling rate
target_sample_rate = 16000

# Iterating over each language
for lang in LANGUAGES:
    print(f"\n{'='*50}")
    print(f"Evaluating language: {lang}")
    print(f"{'='*50}")

    # Defining the paths for the current language
    TEST_DIR = pathlib.Path(f"tests/{lang}_gold_wavs")
    CSV_PATH = pathlib.Path(f"tests/{lang}_gold.csv")
    print(f"Looking for test files in: {TEST_DIR}")
    print(f"Using reference transcripts from: {CSV_PATH}")

    # Checking if the test data exists
    if not TEST_DIR.exists() or not CSV_PATH.exists():
        print(f"Skipping {lang} — missing test data.")
        continue

    # Loading the reference transcripts for the current language
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        references = {row["file"]: row["text"] for row in reader}
    print(f"Loaded {len(references)} reference transcripts")

    # Initialising lists for references and hypotheses
    ref_texts, hyp_texts = [], []
    wav_files = list(TEST_DIR.glob("*.wav"))
    print(f"Found {len(wav_files)} audio files to process")

    # Iterating over the test audio files for the current language
    for wav_path in wav_files:
        try:
            # Loading the audio file
            waveform, sample_rate = torchaudio.load(wav_path)
            
            # Resampling the audio file if necessary
            if sample_rate != target_sample_rate:
                waveform = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)

            # Extracting features from the audio file
            audio = processor.feature_extractor(waveform.squeeze().numpy(), sampling_rate=target_sample_rate, return_tensors="pt", return_attention_mask=True)
            input_feats = audio.input_features.to(device)
            attn_mask = audio.attention_mask.to(device)

            # Generating the transcript
            prompt_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe")
            
            with torch.no_grad():
                predicted_ids = model.generate(input_feats, attention_mask=attn_mask, forced_decoder_ids=prompt_ids, use_cache=True)
            
            # Decoding the generated IDs to get the transcription
            transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Appending reference and hypothesis
            ref_texts.append(references[wav_path.name])
            hyp_texts.append(transcription)
            
        # Error handling
        except Exception as e:
            print(f" -> ERROR: {str(e)}")
            continue

    # If there are no successful transcriptions, skip WER calculation
    if len(hyp_texts) == 0:
        print(f"No valid predictions for {lang}.")
        continue

    # Computing and printing WER for the current language
    wer = jiwer.wer(ref_texts, hyp_texts)
    print(f"WER (baseline) for {lang}: {wer:.2%}")

    # Saving the results to a file
    with open(f"results/baseline_results_{lang}.txt", "w", encoding="utf-8") as f:
        f.write(f"Word Error Rate (WER) for {lang}: {wer:.5%}\n\n")
        for i, (ref, hyp) in enumerate(zip(ref_texts, hyp_texts)):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Reference: {ref}\n")
            f.write(f"Hypothesis: {hyp}\n\n")

print("\nBaseline evaluation completed for all languages!")
print("\nAll detailed results have been saved in the 'results' folder.")
