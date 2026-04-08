import os
from spectrogram import generate_spectrogram
from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess():
    data = "data/genres_original"
    x = []
    y = []

    # loop over folder and sub-folder
    for genre_folder in os.scandir(data):
        for files in os.scandir(genre_folder):
            # get sample
            sample = files.name
            spectogram = generate_spectrogram(sample)
            # if corrupt then dont add
            if spectogram is not None:
                x.append(spectogram)
                y.append(sample[:sample.find(".")])
    
    # get string to num map
    mapping = LabelEncoder()
    y = mapping.fit_transform(y)
    mapping = dict(zip(range(len(mapping.classes_)), mapping.classes_))
    x = np.array(x)
    print(x.shape)
    return x,y,mapping
