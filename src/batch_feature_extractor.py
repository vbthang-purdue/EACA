# src/batch_feature_extractor.py
import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from typing import Dict, List, Tuple
from .data_loader import MELDDataset
from .feature_extractor import MultimodalFeatureExtractor
from .preprocessor import DataPreprocessor
from config import Config

class BatchFeatureExtractor:
    """
    Batch processor for extracting multimodal features from the entire MELD dataset.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_extractor = MultimodalFeatureExtractor()
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for storing features."""
        os.makedirs(self.config.TEXT_FEATURES_DIR, exist_ok=True)
        os.makedirs(self.config.AUDIO_FEATURES_DIR, exist_ok=True)
        os.makedirs(self.config.MULTIMODAL_FEATURES_DIR, exist_ok=True)
        os.makedirs(self.config.PROCESSED_DATA_DIR, exist_ok=True)
        
    def extract_features_for_split(self, split: str) -> Dict:
        """
        Extract features for a specific dataset split (train, dev, test).
        
        Args:
            split: Dataset split ('train', 'dev', 'test')
            
        Returns:
            Dictionary containing extracted features and metadata
        """
        print(f"\n🎯 Processing {split} split...")
        
        # Load dataset
        dataset = MELDDataset(self.config.DATA_DIR, split=split)
        
        # Initialize storage
        features = {
            'text_features': [],
            'audio_features': [],
            'multimodal_features': [],
            'emotions': [],
            'sentiments': [],
            'dialogue_ids': [],
            'utterance_ids': [],
            'speakers': [],
            'original_texts': []
        }
        
        successful = 0
        failed = 0
        
        # Process each sample
        for idx in tqdm(range(len(dataset)), desc=f"Extracting {split} features"):
            try:
                sample = dataset[idx]
                
                # Preprocess sample
                processed_sample = DataPreprocessor.preprocess_sample(sample)
                
                # Extract features
                sample_features = self.feature_extractor.extract_features(processed_sample)
                
                # Store features and metadata
                features['text_features'].append(sample_features['text'])
                features['audio_features'].append(sample_features['audio'])
                features['multimodal_features'].append(sample_features['multimodal'])
                features['emotions'].append(sample['emotion'])
                features['sentiments'].append(sample['sentiment'])
                features['dialogue_ids'].append(sample['dialogue_id'])
                features['utterance_ids'].append(sample['utterance_id'])
                features['speakers'].append(sample['speaker'])
                features['original_texts'].append(sample['text'])
                
                successful += 1
                
            except Exception as e:
                if self.config.VERBOSE:
                    print(f"❌ Failed to process sample {idx}: {e}")
                failed += 1
                continue
        
        # Convert to numpy arrays
        for key in ['text_features', 'audio_features', 'multimodal_features']:
            features[key] = np.array(features[key])
        
        print(f"✅ {split}: {successful} successful, {failed} failed")
        
        return features
    
    def save_features(self, features: Dict, split: str):
        """
        Save extracted features to disk.
        
        Args:
            features: Dictionary of features to save
            split: Dataset split name
        """
        base_path = self.config.FEATURES_DIR
        
        if self.config.SAVE_FORMAT == "pickle":
            # Save as pickle
            with open(os.path.join(base_path, f"{split}_features.pkl"), 'wb') as f:
                pickle.dump(features, f)
                
        elif self.config.SAVE_FORMAT == "numpy":
            # Save individual numpy arrays
            np.save(os.path.join(self.config.TEXT_FEATURES_DIR, f"{split}_text.npy"), features['text_features'])
            np.save(os.path.join(self.config.AUDIO_FEATURES_DIR, f"{split}_audio.npy"), features['audio_features'])
            np.save(os.path.join(self.config.MULTIMODAL_FEATURES_DIR, f"{split}_multimodal.npy"), features['multimodal_features'])
            
            # Save metadata
            metadata = {
                'emotions': features['emotions'],
                'sentiments': features['sentiments'],
                'dialogue_ids': features['dialogue_ids'],
                'utterance_ids': features['utterance_ids'],
                'speakers': features['speakers'],
                'original_texts': features['original_texts']
            }
            with open(os.path.join(base_path, f"{split}_metadata.pkl"), 'wb') as f:
                pickle.dump(metadata, f)
        
        print(f"💾 Saved {split} features to disk")
    
    def load_features(self, split: str) -> Dict:
        """
        Load previously extracted features.
        
        Args:
            split: Dataset split name
            
        Returns:
            Dictionary of loaded features
        """
        if self.config.SAVE_FORMAT == "pickle":
            with open(os.path.join(self.config.FEATURES_DIR, f"{split}_features.pkl"), 'rb') as f:
                return pickle.load(f)
        else:
            features = {}
            features['text_features'] = np.load(os.path.join(self.config.TEXT_FEATURES_DIR, f"{split}_text.npy"))
            features['audio_features'] = np.load(os.path.join(self.config.AUDIO_FEATURES_DIR, f"{split}_audio.npy"))
            features['multimodal_features'] = np.load(os.path.join(self.config.MULTIMODAL_FEATURES_DIR, f"{split}_multimodal.npy"))
            
            with open(os.path.join(self.config.FEATURES_DIR, f"{split}_metadata.pkl"), 'rb') as f:
                metadata = pickle.load(f)
                features.update(metadata)
            
            return features
    
    def extract_all_features(self):
        """
        Extract features for all dataset splits (train, dev, test).
        """
        print("🚀 Starting complete feature extraction for MELD dataset...")
        
        for split in ['train', 'dev', 'test']:
            try:
                # Extract features
                features = self.extract_features_for_split(split)
                
                # Save features
                self.save_features(features, split)
                
                # Print summary
                self.print_feature_summary(features, split)
                
            except Exception as e:
                print(f"❌ Error processing {split} split: {e}")
                continue
    
    def print_feature_summary(self, features: Dict, split: str):
        """
        Print summary of extracted features.
        """
        print(f"\n📊 {split.upper()} FEATURE SUMMARY:")
        print(f"   Text features: {features['text_features'].shape}")
        print(f"   Audio features: {features['audio_features'].shape}")
        print(f"   Multimodal features: {features['multimodal_features'].shape}")
        print(f"   Total samples: {len(features['emotions'])}")
        
        # Emotion distribution
        emotion_counts = {}
        for emotion in features['emotions']:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"   Emotion distribution:")
        for emotion, count in emotion_counts.items():
            print(f"     {emotion}: {count}")