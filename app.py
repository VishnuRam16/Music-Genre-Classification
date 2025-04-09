import streamlit as st
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import joblib
import os
import tempfile
import soundfile as sf

# Set page config
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="centered"
)

# Title and description
st.title("🎵 Music Genre Classifier")
st.write("Upload an audio file to predict its genre!")

# Function to extract features from audio file
def extract_features(audio_path):
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, duration=30)
        
        # Calculate length
        length = len(y)
        
        # Extract features in the same order as training data
        # Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_stft_mean = np.mean(chroma_stft)
        chroma_stft_var = np.var(chroma_stft)
        
        # RMS
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)
        
        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_var = np.var(spectral_centroid)
        
        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_var = np.var(spectral_bandwidth)
        
        # Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_var = np.var(rolloff)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_var = np.var(zcr)
        
        # Harmony and Perceptr
        harmony = librosa.effects.harmonic(y)
        harmony_mean = np.mean(harmony)
        harmony_var = np.var(harmony)
        
        perceptr = librosa.effects.preemphasis(y)
        perceptr_mean = np.mean(perceptr)
        perceptr_var = np.var(perceptr)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = np.mean(mfcc, axis=1)
        mfcc_vars = np.var(mfcc, axis=1)
        
        # Create features dictionary with exact column names from training data
        features = {
            'length': length,
            'chroma_stft_mean': chroma_stft_mean,
            'chroma_stft_var': chroma_stft_var,
            'rms_mean': rms_mean,
            'rms_var': rms_var,
            'spectral_centroid_mean': spectral_centroid_mean,
            'spectral_centroid_var': spectral_centroid_var,
            'spectral_bandwidth_mean': spectral_bandwidth_mean,
            'spectral_bandwidth_var': spectral_bandwidth_var,
            'rolloff_mean': rolloff_mean,
            'rolloff_var': rolloff_var,
            'zero_crossing_rate_mean': zcr_mean,
            'zero_crossing_rate_var': zcr_var,
            'harmony_mean': harmony_mean,
            'harmony_var': harmony_var,
            'perceptr_mean': perceptr_mean,
            'perceptr_var': perceptr_var,
            'tempo': tempo
        }
        
        # Add MFCC features
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = mfcc_means[i]
            features[f'mfcc{i+1}_var'] = mfcc_vars[i]
        
        # Create DataFrame with single row
        features_df = pd.DataFrame([features])
        
        return features_df
        
    except Exception as e:
        st.error(f"Error processing audio file: {str(e)}")
        return None

# Function to train the model
def train_model():
    # Load the features dataset
    df = pd.read_csv('features_30_sec.csv')
    
    # Get feature columns in correct order (excluding filename and label)
    feature_columns = [col for col in df.columns if col not in ['filename', 'label']]
    
    # Prepare features and target
    X = df[feature_columns]
    y = df['label']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models to try
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    # Find the best model using cross-validation
    best_score = 0
    best_model = None
    
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=5)
        avg_score = scores.mean()
        st.write(f"{name} Cross-validation score: {avg_score:.3f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_model = model
    
    # Train the best model on the full dataset
    best_model.fit(X_scaled, y)
    st.success(f"Selected {best_model.__class__.__name__} as the best model with score: {best_score:.3f}")
    
    return best_model, scaler, feature_columns

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav'])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    # Display audio player
    st.audio(uploaded_file)
    
    # Add a button to trigger prediction
    if st.button("Predict Genre"):
        with st.spinner("Processing audio and making prediction..."):
            # Extract features
            features_df = extract_features(temp_path)
            
            if features_df is not None:
                # Train model (in production, you would load a pre-trained model)
                model, scaler, feature_columns = train_model()
                
                # Ensure features are in the same order as training data
                features_df = features_df[feature_columns]
                
                # Scale features
                features_scaled = scaler.transform(features_df)
                
                # Make prediction
                prediction = model.predict(features_scaled)[0]
                probabilities = model.predict_proba(features_scaled)[0]
                
                # Display results
                st.success(f"Predicted Genre: {prediction}")
                
                # Display confidence scores
                st.write("Confidence Scores:")
                for genre, prob in zip(model.classes_, probabilities):
                    st.write(f"{genre}: {prob:.2%}")
                
                # Create a bar chart of confidence scores
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 5))
                plt.bar(model.classes_, probabilities)
                plt.xticks(rotation=45)
                plt.title("Genre Prediction Confidence Scores")
                st.pyplot(plt)
    
    # Clean up temporary file
    os.unlink(temp_path)

# Add some information about the app
st.markdown("""
### About
This app uses machine learning to classify music genres. It extracts various audio features including:
- Chroma STFT
- RMS (Root Mean Square)
- Spectral Centroid
- Spectral Bandwidth
- Rolloff
- Zero Crossing Rate
- Harmony
- Perceptr
- Tempo
- MFCC (Mel-frequency cepstral coefficients)

The model is trained on a dataset of music samples from different genres and uses cross-validation to select the best performing model among Random Forest, KNN, and SVM.
""") 