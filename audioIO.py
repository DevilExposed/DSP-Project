# mono Wav 16kHz

import sounddevice as sd
import soundfile as sf
from pathlib import Path
import numpy as np
import tempfile

# Recording at 16kHz mono
# This is the default setting for most speech recognition libraries
SAMPLE_RATE = 16_000      
CHANNELS = 1           
DTYPE = "float32"

# Recording audio from the system’s default input device and writing it to a WAV file.
class Recorder:

    def __init__(self):
        self._frames: list[np.ndarray] = []
        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=self._callback)
        

    def start(self) -> None:
        self._frames.clear()
        self._stream.start()
        
    def stop_and_save(self, wav_path: Path) -> Path:
        self._stream.stop()
        self._stream.close()
        
        # Checking if nothing has been recorded
        if not self._frames:
            raise RuntimeError("Empty recording")
        
        # Join the chunks of audio data
        audio = np.concatenate(self._frames, axis=0)
        # Save the audio file
        sf.write(wav_path, audio, SAMPLE_RATE)
        return wav_path

    def _callback(self, indata, frames, time, status):
        if status:
            print("Recorder status:", status, flush=True)
        self._frames.append(indata.copy())