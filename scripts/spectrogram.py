# code for spectogram testing(take file input->check whether it is wav or not(webm)->if webm->convert to wav using moviepy->else generate spectrogram and return mel array)
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import librosa

def generate_spectrogram(file_path):
    file_path =  "data/genres_original/" + file_path[:file_path.find(".")] + "/" + file_path
    # some file in data set are corrupted -_-
    try:
        print(file_path)
        audio = AudioSegment.from_file(file_path)
        target_ms = 30 * 1000
        if len(audio) < target_ms:
            audio = audio * (target_ms // len(audio) + 1)
        if len(audio) > target_ms:
            best_start = 0
            max_rms = 0
            for i in range(0, len(audio) - target_ms + 1, 1000):
                chunk = audio[i:i+target_ms]
                if chunk.rms > max_rms:
                    max_rms = chunk.rms
                    best_start = i
            audio = audio[best_start:best_start+target_ms]
        audio = audio[:target_ms]
        audio.export("tmp_train_audio.wav", format="wav")
        y, sr = librosa.load("tmp_train_audio.wav") #sr sample rate
        # Step 2: do fft over the audio file and convert output to mel scale
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512)
        # # Convert amplitude to decibels so that difference in loudness is distinguishable
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        # getting all values between 0 and 1
        mel_spect = (mel_spect - mel_spect.min()) / (mel_spect.max() - mel_spect.min())
        return mel_spect
    except Exception:
        print("Audio corrupt.. skip")
        return None