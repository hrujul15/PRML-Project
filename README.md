# PRML-Project

PRML Project Spring Semester 2025-26

---

# Music Genre Classifier

From audio files predict the genre of the music.  
This model works best for english songs only!
---

# Team Members
- Arin Nain (B24EE1096)
- Hrujul Mendhe (B24CM1077)
- Priyam Maheshbhai Patel (B24CS1058)
- Tushar Verma (B24CM1070)

---

# Running the Project

## 1. Create a virtual environment

python -m venv venv

> On Windows, if `python` doesn’t work, try:
py -m venv venv

## 2. Activate the virtual environment

### Linux / macOS
source venv/bin/activate

### Windows (PowerShell)
venv\Scripts\Activate.ps1

> If you get a permission error, please run this once:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run the program
python scripts/predict.py "Path_to_music_file"

---

## Resource used for spectogram generation
- Spectrogram using Librosa: https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53
- STFT, FFT explanation image source: https://www.mathworks.com/help/dsp/ref/dsp.stft.html

---

## Resnet Architecture
- Resnet wiki: https://en.wikipedia.org/wiki/Residual_neural_network

---
## MobileNet V2
- https://www.geeksforgeeks.org/computer-vision/what-is-mobilenet-v2/