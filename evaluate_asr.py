import csv
import pathlib
import jiwer
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torchaudio.transforms as T

print("Script started: Loading necessary libraries...")

# Define languages
LANGUAGES = ["en", "bg", "uk", "ru", "ar", "it", "pl", "pt"]
print(f"Will evaluate these languages: {', '.join(LANGUAGES)}")

# Defining the model directory
MODEL_DIR = pathlib.Path("models/whisperFineTune")
print(f"Using model from: {MODEL_DIR}")

# Defining the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Loading the fine-tuned model and processor
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
processor = WhisperProcessor.from_pretrained(MODEL_DIR)
print("Model and processor loaded successfully")

# Defining the target sampling rate
target_sample_rate = 16000
print(f"Target sampling rate: {target_sample_rate}Hz")

# Iterate over each language
for lang in LANGUAGES:
    print(f"\n{'='*50}")
    print(f"Evaluating language: {lang}")
    print(f"{'='*50}")
    
    # Defining the paths for the current language
    TEST_DIR = pathlib.Path(f"tests/{lang}_gold_wavs")
    CSV_PATH = pathlib.Path(f"tests/{lang}_gold.csv")
    print(f"Looking for test files in: {TEST_DIR}")
    print(f"Using reference transcripts from: {CSV_PATH}")

    # Loading the reference transcripts for the current language
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        references = {row["file"]: row["text"] for row in reader}
    print(f"Loaded {len(references)} reference transcripts")

    # Initialising lists for references and hypotheses
    ref_texts = []
    hyp_texts = []
    
    # Counting the total number of audio files to process
    wav_files = list(TEST_DIR.glob("*.wav"))
    print(f"Found {len(wav_files)} audio files to process")

    # Iterating over the test audio files for the current language
    for i, wav_path in enumerate(TEST_DIR.glob("*.wav")):
        
        try:
            # Loading the audio file
            waveform, sample_rate = torchaudio.load(wav_path)
            
            # Resampling the audio file if necessary
            if sample_rate != target_sample_rate:
                
                resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                waveform = resampler(waveform)
                
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

            # Append reference and hypothesis
            ref_texts.append(references[wav_path.name])
            hyp_texts.append(transcription)
            
        # Error handling
        except Exception as e:
            print(f" -> ERROR: {str(e)}")
            continue

    # Show progress summary
    print(f"\nProcessed {len(hyp_texts)}/{len(wav_files)} files successfully for {lang}")
    
    # If there are no successful transcriptions, skip WER calculation
    if len(hyp_texts) == 0:
        print(f"No successful transcriptions for {lang}. Skipping WER calculation.")
        continue

    # Normalising the reference and hypothesis texts
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip()
    ])
    
    # Normalising the reference and hypothesis texts
    ref_texts_norm = [transform(r) for r in ref_texts]
    hyp_texts_norm = [transform(h) for h in hyp_texts]
    
    # Computing and printing WER for the current language
    print("Computing Word Error Rate...")
    wer = jiwer.wer(ref_texts, hyp_texts)
    print(f"Word Error Rate (WER) for {lang}: {wer:.5%}")
    
    # Saving the results to a file
    result_file = f"results_{lang}.txt"
    print(f"Saving detailed results to {result_file}")
    with open(f"results/{result_file}", "w", encoding="utf-8") as f:
        f.write(f"Word Error Rate (WER) for {lang}: {wer:.5%}\n\n")
        for i, (ref, hyp) in enumerate(zip(ref_texts, hyp_texts)):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Reference: {ref}\n")
            f.write(f"Hypothesis: {hyp}\n\n")

print("\nFine-tuned ASR evaluation completed for all languages!")
print("\nAll detailed results have been saved in the 'results' folder.")
