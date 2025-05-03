# Necessary imports for the GUI and threading
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

from concurrent.futures import ThreadPoolExecutor
import audioIO
import tempfile
from pathlib import Path, PurePath
from uuid import uuid4
import speech_to_text as stt
import translate
import text_to_speech as tts
import atexit


# Multilingual Translation class
class MultilingualTranslation:
    def __init__(self, root):
        # Setting the root window
        self.root = root
        self.root.title("Automated Multilingual Speech based Translator")
        self.root.geometry("800x600")  # Increased window size
        self.root.configure(bg="#f0f0f0")  # Light gray background
        self.last_tts_audio = None  # Storing path to last generated speech
        self.last_tts_text = "" # Storing the last text that was converted to speech
        
        # Single thread pool for the background tasks   
        self.executor = ThreadPoolExecutor(max_workers=2)        
        self.recorder: audioIO.Recorder | None = None
        
        # Main frame with padding and style
        self.main_frame = ttk.Frame(self.root, padding="20 20 20 20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create styles
        self.style = ttk.Style()
        self.style.configure("Title.TLabel", font=("Arial", 16, "bold"))
        self.style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        self.style.configure("Hint.TLabel", font=("Arial", 9, "italic"), foreground="gray")
        
        # Language selection frame
        self.lang_frame = ttk.LabelFrame(self.main_frame, text="Language Selection", padding="10 10 10 10")
        self.lang_frame.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        
        # Dictionary of language codes
        self.lang_code = {
            "English": "en", "Bulgarian": "bg", "Ukrainian": "uk", 
            "Russian": "ru", "Arabic": "ar", "Italian": "it", 
            "Polish": "pl", "Portuguese": "pt"
        }
        
        # Source language selection
        self.source_language_label = ttk.Label(self.lang_frame, text="Source Language:", style="Header.TLabel")
        self.source_language_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.source_language_dropdown = ttk.Combobox(
            self.lang_frame,
            values=list(self.lang_code.keys()),
            width=30,
            state="readonly"
        )
        self.source_language_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.source_language_dropdown.current(0)
        
        # Target language selection
        self.target_language_label = ttk.Label(self.lang_frame, text="Target Language:", style="Header.TLabel")
        self.target_language_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        self.target_language_dropdown = ttk.Combobox(
            self.lang_frame,
            values=list(self.lang_code.keys()),
            width=30,
            state="readonly"
        )
        self.target_language_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.target_language_dropdown.current(0)
        
        # Control buttons frame
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=10)
        
        # Recording control buttons
        self.record_button = ttk.Button(self.control_frame, text="Start Recording", command=self.start_recording, width=20)
        self.record_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(self.control_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED, width=20)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = ttk.Button(self.control_frame, text="Cancel", command=self.cancel_recording, state=tk.DISABLED, width=15)
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Text areas frame
        self.text_frame = ttk.Frame(self.main_frame)
        self.text_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        
        # Transcribed text section
        self.transcribed_text_label = ttk.Label(self.text_frame, text="Transcribed Text:", style="Header.TLabel")
        self.transcribed_text_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        self.transcribed_text = tk.Text(self.text_frame, height=8, width=70, wrap=tk.WORD)
        self.transcribed_text.grid(row=1, column=0, sticky="nsew", pady=(0, 5))
        
        # Punctuation hint
        self.punctuation_hint = ttk.Label(
            self.text_frame,
            text="Hint: Adding punctuation will improve translation accuracy",
            style="Hint.TLabel"
        )
        self.punctuation_hint.grid(row=2, column=0, sticky="w", pady=(0, 10))
        
        # Translated text section
        self.translated_text_label = ttk.Label(self.text_frame, text="Translated Text:", style="Header.TLabel")
        self.translated_text_label.grid(row=3, column=0, sticky="w", pady=(0, 5))
        
        self.translated_text = tk.Text(self.text_frame, height=8, width=70, wrap=tk.WORD)
        self.translated_text.grid(row=4, column=0, sticky="nsew", pady=(0, 5))
        
        # Bottom controls frame
        self.bottom_frame = ttk.Frame(self.main_frame)
        self.bottom_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=10)
        
        # Audio control buttons
        self.play_button = ttk.Button(self.bottom_frame, text="Play Audio", command=self.play_audio, width=20)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        # Clear text button
        self.clear_button = ttk.Button(self.main_frame, text="Clear", command=self.clear_text)
        self.clear_button.grid(row=7, column=2, pady=5)

        self.manual_translate_button = ttk.Button(
            self.bottom_frame,
            text="Translate Written Text",
            command=self.translate_written_text,
            width=20
        )
        self.manual_translate_button.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready.")
        self.status_bar = ttk.Label(
            self.main_frame,
            textvariable=self.status_var,
            style="Hint.TLabel",
            anchor="w"
        )
        self.status_bar.grid(row=4, column=0, columnspan=3, sticky="ew", padx=5, pady=(10, 0))
        
        # Configure grid weights
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(2, weight=1)
        self.text_frame.columnconfigure(0, weight=1)
        self.text_frame.rowconfigure(1, weight=1)
        self.text_frame.rowconfigure(4, weight=1)
        
        # Closing the application
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Closing the application forcefully
        self.root.bind("<Destroy>", self.on_close_event)
        
        atexit.register(self.cleanup_audio_file)

        
        
    # Starts the recording without freezing the GUI
    def start_recording(self):
        self.record_button.config(state=tk.DISABLED) # Disabling the button to avoid double clicks
        self.stop_button.config(state=tk.NORMAL) # Allowing the user to stop the recording
        self.cancel_button.config(state=tk.NORMAL) # Allowing the user to cancel the recording
        self.transcribed_text.delete("1.0", tk.END) # Clear previous text
        self.translated_text.delete("1.0", tk.END)

        # Creating and start the recorder (runs inside PortAudio's own thread)
        self.recorder = audioIO.Recorder()
        self.recorder.start()

        self.status_var.set("Recording… click Stop when finished.")

    # When user is ready stop and save the recording
    def stop_recording(self):
        if self.recorder is None:
            return

        # Disabling the buttons to avoid double clicks
        self.stop_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.DISABLED)
        self.status_var.set("Saving audio…")

        # Defining worker function to handle the recording process asynchronously
        def _worker(rec):
            try:
                # Creating a unique temporary WAV file path using UUID
                wav_path = (Path(tempfile.gettempdir()) / f"stt_{uuid4().hex}.wav")
                
                # Stopping the recording and saving it to the temporary WAV file
                rec.stop_and_save(wav_path)
                
                # Scheduling the post-recording processing in the main GUI thread
                self.root.after(0, self._after_record(wav_path))
            # Empty recording
            except RuntimeError as err:
                messagebox.showerror("Error", f"Runtime error: {err}")
                self.root.after(0, self._on_record_error(str(err)))
            # Other errors
            except Exception as err:
                messagebox.showerror("Error", f"Unexpected error: {err}")
                self.root.after(0, self._on_record_error(f"Unexpected error: {err}"))        
            finally:
                # Clearing the reference to the recorder
                self.recorder = None

        self.executor.submit(_worker, self.recorder)
    
    # Cancelling the recording, do not save to file and resetting the buttons
    def cancel_recording(self):
        if self.recorder is not None:
            self.recorder._stream.stop()
            self.recorder._stream.close()
            self.recorder = None
        self._reset_buttons()
        self.status_var.set("Recording cancelled.")
    
    # Defining what to do after the recording is finished and saved
    def _after_record(self, wav_path):
        # Resetting the buttons
        self._reset_buttons()

        # Setting status for user feedback
        self.status_var.set("Transcribing…")

        # Getting the selected source language from the dropdown
        source_lang = self.lang_code[self.source_language_dropdown.get()]

        # Transcribing the saved audio using the fine-tuned model
        try:
            #Transcribing the saved audio
            text = stt.transcribe(wav_path, language=source_lang)

            # Displaying the transcribed text in the GUI
            self.transcribed_text.delete("1.0", tk.END)
            self.transcribed_text.insert(tk.END, text)

            # Getting the selected target language
            target_lang = self.lang_code[self.target_language_dropdown.get()]

            # Checking if source and target language are the same (no translation needed)
            if source_lang == target_lang:
                self.translated_text.delete("1.0", tk.END)
                self.translated_text.insert(tk.END, text)
                self.status_var.set("Source and target languages are the same. Translation skipped.")
                return

            # Translating the transcribed text
            self.status_var.set("Translating...")
            print(f"Translating from {source_lang} -> {target_lang}...")
            translated = translate.translate_text(text, source_lang, target_lang)

            # Displaying the translated text in the GUI
            self.translated_text.delete("1.0", tk.END)
            self.translated_text.insert(tk.END, translated)
            self.status_var.set("Translation complete.")
            
        except Exception as e:
            messagebox.showerror("Translation model not found", f"This translation direction is not supported.")
            print(f"Error during transcription: {e}")
            self.status_var.set("Transcription failed.")   
        
    # Error handling for the recording process
    def _on_record_error(self, msg: str):
        self._reset_buttons()
        self.status_var.set("Ready.")
        messagebox.showwarning("Recording cancelled", msg)
    
    # Playing the translated speech using gTTS
    def play_audio(self):
        # Retrieving the translated text
        text = self.translated_text.get("1.0", tk.END).strip()

        if not text:
            messagebox.showwarning("No text", "Please translate something first.")
            return

        # Getting the language code from dropdown
        target_lang = self.lang_code[self.target_language_dropdown.get()]

        try:
            # Checking if the current text matches the last spoken text
            if text == self.last_tts_text and self.last_tts_audio and self.last_tts_audio.exists():
                print("Replaying cached audio.")
                tts.play_speech(self.last_tts_audio)
                self.status_var.set("Replaying previous audio.")
                return


            # If the text is new, generate new speech file
            print("Generating new speech file...")
            self.status_var.set("Generating speech...")
            mp3_path = tts.text_to_speech(text, target_lang)
            
            # Playing the new speech
            tts.play_speech(mp3_path)
            
            # Deleting old audio file
            if self.last_tts_audio and self.last_tts_audio.exists():
                try:
                    print(f"Deleting old audio file: {self.last_tts_audio}")
                    self.last_tts_audio.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete old audio file. {e}")
                
            

            # Updating cache
            self.last_tts_audio = mp3_path
            self.last_tts_text = text

            self.status_var.set("Playback complete.")
            
        except Exception as e:
            print(f"Error during TTS: {e}")
            messagebox.showerror("TTS failed", f"Error during TTS: {e}")
            self.status_var.set("TTS failed.")


        
    # Resetting the buttons
    def _reset_buttons(self):
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.DISABLED)
        
    # Defining what to do when manually translating the written transcription text
    def translate_written_text(self):
        # Retrieving the text from the transcription box
        input_text = self.transcribed_text.get("1.0", tk.END).strip()

        # Getting source and target languages from the dropdown
        source_lang = self.lang_code[self.source_language_dropdown.get()]
        target_lang = self.lang_code[self.target_language_dropdown.get()]

        # Checking if there's any text to translate
        if not input_text:
            messagebox.showwarning("No Input", "Please enter some text to translate.")
            return

        # Checking if source and target are the same
        if source_lang == target_lang:
            self.translated_text.delete("1.0", tk.END)
            self.translated_text.insert(tk.END, input_text)
            self.status_var.set("Same source and target language. Translation skipped.")
            return

        # Performing the translation using MarianMT
        try:
            self.status_var.set("Translating typed text...")
            print(f"Manually translating: {source_lang} -> {target_lang}")
            translated = translate.translate_text(input_text, source_lang, target_lang)

            self.translated_text.delete("1.0", tk.END)
            self.translated_text.insert(tk.END, translated)
            self.status_var.set("Translation complete.")
        except Exception as e:
            messagebox.showerror("Translation model not found", f"This translation direction is not supported.")
            print(f"Error during manual translation: {e}")
            self.status_var.set("Manual translation failed.")

    # Clearing transcription and translation text areas
    def clear_text(self):
        # Clearing transcribed text box
        self.transcribed_text.delete("1.0", tk.END)

        # Clearing translated text box
        self.translated_text.delete("1.0", tk.END)

        # Resetting the TTS cache
        self.last_tts_audio = None
        self.last_tts_text = ""

        self.status_var.set("Cleared.")

    # Handling app close and cleaning up files
    def on_close(self):
        print("[CLOSE] Closing application...")

        # Shutting down executor to allow Python to exit properly
        self.executor.shutdown(wait=False)

        # Deleting last audio file if exists
        if self.last_tts_audio and self.last_tts_audio.exists():
            try:
                print(f"[CLOSE] Deleting audio file: {self.last_tts_audio}")
                self.last_tts_audio.unlink()
            except Exception as e:
                messagebox.showerror("Error", f"Could not delete audio file. {e}")
                print(f"[CLOSE] Warning: Could not delete audio file. {e}")

        # Destroying the GUI
        self.root.destroy()
    
    # Handling the close event forcefully
    def on_close_event(self, event):
        if event.widget == self.root:
            print("[DESTROY] Destroying GUI...")
            
             # Shutting down executor
            self.executor.shutdown(wait=False)
        
            if self.last_tts_audio and self.last_tts_audio.exists():
                try:
                    print(f"Deleting audio file on exit (destroy): {self.last_tts_audio}")
                    self.last_tts_audio.unlink()
                except Exception as e:
                    messagebox.showerror("Error", f"Could not delete audio file. {e}")
                    print(f"Warning: Could not delete audio file on exit. {e}")
                    
    # Cleaning up audio file when application exits
    def cleanup_audio_file(self):
        if self.last_tts_audio and self.last_tts_audio.exists():
            try:
                print(f"[EXIT] Deleting audio file: {self.last_tts_audio}")
                self.last_tts_audio.unlink()
            except Exception as e:
                messagebox.showerror("Error", f"Could not delete audio file. {e}")
                print(f"[EXIT] Warning: Could not delete audio file. {e}")


# Main function
if __name__ == "__main__":
    root = tk.Tk()
    app = MultilingualTranslation(root)
    root.mainloop()
