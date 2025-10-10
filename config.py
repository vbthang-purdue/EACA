# config.py
import os

class Config:
    # Data paths
    DATA_DIR = "data/meld"
    RAW_DATA_DIR = "data/meld/raw"
    PROCESSED_DATA_DIR = "data/processed"
    
    # CSV files
    TRAIN_CSV = os.path.join(DATA_DIR, "train_sent_emo.csv")
    DEV_CSV = os.path.join(DATA_DIR, "dev_sent_emo.csv")
    TEST_CSV = os.path.join(DATA_DIR, "test_sent_emo.csv")
    
    # Video directories
    TRAIN_VIDEO_DIR = os.path.join(DATA_DIR, "train_splits")
    DEV_VIDEO_DIR = os.path.join(DATA_DIR, "dev_splits_complete")
    TEST_VIDEO_DIR = os.path.join(DATA_DIR, "output_repeated_splits_test")
    
    # Audio directories
    AUDIO_DIR = os.path.join(PROCESSED_DATA_DIR, "audio")
    
    # Model parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    
    # Feature extraction
    TEXT_EMBEDDING_DIM = 768
    AUDIO_FEATURE_DIM = 168
    MULTIMODAL_FEATURE_DIM = 936  # 768 + 168
    
    # Audio parameters
    TARGET_SAMPLE_RATE = 16000
    N_MFCC = 40
    
    # Emotion labels
    EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    EMOTION_MAP = {emotion: idx for idx, emotion in enumerate(EMOTION_LABELS)}
    
    # Sentiment labels
    SENTIMENT_LABELS = ['positive', 'negative', 'neutral']
    SENTIMENT_MAP = {sentiment: idx for idx, sentiment in enumerate(SENTIMENT_LABELS)}