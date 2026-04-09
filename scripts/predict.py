import torch
from model_def import CNN
import sys
import numpy as np
from moviepy import AudioFileClip
import librosa
import os
from train import train

# get file path of music from command line argument
file = sys.argv[-1]

# if no music file provided exit
if len(sys.argv) == 1:
    print("No file given in input")
else:
    # load model
    loaded_model = CNN()

    # if model not found train model
    if not os.path.exists("model.pth"):
        train()

    # load model
    loaded_model.load_state_dict(
        torch.load("model.pth", map_location=torch.device("cpu"), weights_only=True)
    )

    loaded_model.eval()

    # number to genre string mapping
    mapping = {
        0: np.str_("blues"),
        1: np.str_("classical"),
        2: np.str_("country"),
        3: np.str_("disco"),
        4: np.str_("hiphop"),
        5: np.str_("jazz"),
        6: np.str_("metal"),
        7: np.str_("pop"),
        8: np.str_("reggae"),
        9: np.str_("rock"),
    }

    # if file is not .wav convert to correct format
    if ".wav" not in file:
        clip = AudioFileClip(file)
        clip.write_audiofile("out_audio.wav")
        file = "out_audio.wav"

    y, sr = librosa.load(file)  # sr = sample rate
    y = y[: int(sr * 30)]

    # do fft over the audio file and convert output to mel scale
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512)

    # COnvert amplitude to decibels so that difference in loudness is distinguishable
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    # getting all values between 0 and 1
    mel_spect = (mel_spect - mel_spect.min()) / (mel_spect.max() - mel_spect.min())

    # convert to format torch expects
    mel_spect = mel_spect[np.newaxis, np.newaxis, :, :]

    with torch.no_grad():
        answer = loaded_model(torch.tensor(mel_spect, dtype=torch.float32))

    # get scores by softmax
    scores = torch.softmax(answer, dim=1).detach().numpy()[0]

    # sort scores
    scores = [(mapping[i], scores[i]) for i in range(len(mapping))]
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)

    for label, score in scores_sorted:
        print(f"{label} : {score:.4f}")
    pred_index = answer.argmax().item()

    print("PREDICTED:", mapping[pred_index])
