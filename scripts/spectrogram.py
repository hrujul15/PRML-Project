# code for spectogram testing(take file input->check whether it is wav or not(webm)->if webm->convert to wav using moviepy->else generate spectrogram and return mel array)
import numpy as np
import matplotlib.pyplot as plt
from moviepy import VideoFileClip
import librosa

def generate_spectrogram(file_path):
    if(".wav" not in file_path):
        # Step 1: convert to wav if not in the format
        clip = VideoFileClip(file_path)
        clip.audio.write_audiofile("out_audio.wav")
        file_path="out_audio.wav"
    y, sr = librosa.load(file_path) # sr = sample rate
    y = y[:int(sr*30)]
    # Step 2: do fft over the audio file and convert output to mel scale
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    # # Convert amplitude to decibels so that difference in loudness is distinguishable
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    # getting all values between 0 and 1
    mel_spect = (mel_spect - mel_spect.min()) / (mel_spect.max() - mel_spect.min())
    return mel_spect
