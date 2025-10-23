# EACA - Emotion-Aware Conversational Agent

## Project Overview
Building a more natural, empathetic, and responsive conversational agent by enabling it to understand user emotions in real time.

Our focus: A text and audio-based Emotion Recognition in Conversation (ERC) system that can be integrated into a chatbot.

![logo](logo.png)

## Project Description
This project implements a multimodal emotion recognition system that analyzes both text and audio inputs to detect user emotions during conversations. The system is designed to be integrated into chatbot applications to enable more empathetic and context-aware responses.

## Features
- **Multimodal Emotion Recognition**: Combines text and audio analysis
- **MELD Dataset Support**: Built using the Multimodal EmotionLines Dataset
- **Pre-trained Models**: Utilizes DistilBERT for text features and Librosa for audio

## Dataset
We use the **MELD (Multimodal EmotionLines Dataset)** which contains:
- Text dialogues from TV series
- Audio recordings of conversations
- Emotion labels (anger, disgust, fear, joy, neutral, sadness, surprise)
- Sentiment labels (positive, negative, neutral)

## Installation

### Prerequisites
- Python 3.8+
- FFmpeg (for audio processing)
- PyTorch
- Hugging Face Transformers

### Setup
```
1. Install dependencies:
pip install -r requirements.txt
```

2. Download FFmpeg and place it in the `ffmpeg/` directory

3. Download the MELD dataset and place it in the `data/meld/` directory

## Project Structure
```
eaca-emotion-recognition/
├── src/
│   ├── __init__.py
│   ├── contractions.py          # Text contraction handling
│   ├── preprocessor.py          # Text and audio preprocessing
│   ├── feature_extractor.py     # Feature extraction classes
│   ├── data_loader.py           # Dataset loading utilities
│   └── batch_feature_extractor.py # Batch processing
├── main.py                    # Testing and demonstration
├── config.py                    # Configuration settings
├── extract_features.py          # Feature extraction script
├── requirements.txt             # Python dependencies
└── README.md
```

## Usage

### Basic Testing
Run the main testing script to verify all components:
```bash
python main.py
```

### Feature Extraction
Extract features from the MELD dataset:
```bash
python extract_features.py --split all --format pickle --verbose
```

## Model Architecture
1. **Text Processing**:
   - Contraction expansion
   - Lowercasing and punctuation removal
   - DistilBERT embeddings
   - Mean pooling for sentence representations

2. **Audio Processing**:
   - MP4 to WAV conversion (if needed)
   - Audio normalization and silence removal
   - MFCC feature extraction
   - Prosodic features (pitch, energy)
   - Statistical aggregation

3. **Multimodal Fusion**:
   - Concatenation of text and audio features
   - 936-dimensional combined feature vector

## Configuration
Modify `config.py` to adjust:
- File paths and directories
- Model parameters (batch size, learning rate)
- Feature dimensions
- Audio processing settings

## Dependencies
Key Python packages:
- torch >= 1.9.0
- transformers >= 4.20.0
- librosa >= 0.9.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- tqdm >= 4.60.0

This repository and all of the code contained within it are the result of the collective efforts of a group of students enrolled in the Purdue University course CSCI 49500 - Explorations in Applied Computing. The project represents our work for the course curriculum and is part of our academic requirements.
