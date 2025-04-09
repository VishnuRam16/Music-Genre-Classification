# Music Genre Classification App

This application uses machine learning to classify music genres. It provides a user-friendly interface through Streamlit where users can upload audio files and get genre predictions along with confidence scores.

## Features

- Upload audio files (MP3 or WAV format)
- Real-time genre prediction
- Confidence scores for each genre
- Visual representation of predictions
- Support for multiple music genres

## Installation

1. Clone this repository:
```bash
git clone https://github.com/VishnuRam16/Music-Genre-Classification.git
cd Music-Genre-Classification
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload an audio file using the file uploader

4. Click the "Predict Genre" button to get the genre prediction and confidence scores

## How it Works

The application:
1. Extracts audio features using librosa:
   - MFCC (Mel-frequency cepstral coefficients)
   - Spectral Centroid
   - Chroma features

2. Uses a Random Forest Classifier to predict the genre

3. Provides confidence scores for each possible genre

## Requirements

- Python 3.7+
- Dependencies listed in requirements.txt

## License

This project is licensed under the MIT License - see the LICENSE file for details. 