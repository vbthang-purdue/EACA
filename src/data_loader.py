# src/data_loader.py
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from pathlib import Path

class MELDDataset(Dataset):
    """
    Custom PyTorch Dataset for the MELD (Multimodal EmotionLines Dataset).

    This class handles loading and preprocessing of MELD data, which includes
    text, emotion labels, sentiment, and audio paths for each utterance.
    It assumes the dataset is structured with separate folders for train/dev/test
    and corresponding CSV metadata files.
    """

    def __init__(self, data_dir, split='train', transform=None):
        """
        Initialize the MELD dataset.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Attempt to locate the correct CSV file for the given split
        csv_path = os.path.join(data_dir, f'{split}_sent_emo.csv')
        if not os.path.exists(csv_path):
            # Some MELD releases use different naming conventions
            csv_path = os.path.join(data_dir, f'{split}sent_emo.csv')
        
        # Load metadata CSV into a DataFrame
        self.df = pd.read_csv(csv_path)
        
        # Map split names to corresponding audio directories
        self.audio_dirs = {
            'train': os.path.join(data_dir, 'train_splits'),
            'dev': os.path.join(data_dir, 'dev_splits_complete'), 
            'test': os.path.join(data_dir, 'output_repeated_splits_test')
        }
        
        # Get the directory containing audio for the current split
        self.audio_dir = self.audio_dirs.get(split)
        
    def __len__(self):
        """Return total number of samples in this split."""
        return len(self.df)
    
    def find_audio_file(self, dialogue_id, utterance_id):
        """
        Attempt to locate the audio file corresponding to a dialogue and utterance.

        The MELD dataset may have inconsistent file naming conventions across splits,
        so multiple possible patterns are checked.
        """
        audio_dir = self.audio_dir
        
        # File naming patterns in MELD audio files
        patterns = [
            f"dia{dialogue_id}_utt{utterance_id}.mp4",
            f"dia{dialogue_id}_utt{utterance_id}.wav",
            f"{dialogue_id}_{utterance_id}.mp4",
            f"{dialogue_id}_{utterance_id}.wav",
        ]
        
        # Check for an exact match
        for pattern in patterns:
            audio_path = os.path.join(audio_dir, pattern)
            if os.path.exists(audio_path):
                return audio_path
                
        # If no exact match, perform a fuzzy search within the directory
        if os.path.exists(audio_dir):
            for file in os.listdir(audio_dir):
                if f"dia{dialogue_id}_utt{utterance_id}" in file:
                    return os.path.join(audio_dir, file)
                    
        # Return None if no audio file found (some utterances may not have audio)
        return None
    
    def __getitem__(self, idx):
        """
        Retrieve a single data sample from the dataset.
        """
        # Extract the row corresponding to the given index
        row = self.df.iloc[idx]
        
        # Extract textual and label information from CSV columns
        text = row['Utterance']
        speaker = row['Speaker']
        emotion = row['Emotion']
        sentiment = row['Sentiment']
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        
        # Locate the corresponding audio file on disk
        audio_path = self.find_audio_file(dialogue_id, utterance_id)
        
        # Assemble the sample dictionary
        sample = {
            'text': text,
            'audio_path': audio_path,
            'emotion': emotion,
            'sentiment': sentiment,
            'speaker': speaker,
            'dialogue_id': dialogue_id,
            'utterance_id': utterance_id
        }
        
        # Apply optional transformation (e.g., feature extraction or tokenization)
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class MELDDataLoader:
    """
    Wrapper class for generating DataLoaders for MELD dataset splits.

    This simplifies loading train/dev/test splits with consistent settings
    such as batch size, shuffling, and parallelism.
    """

    def __init__(self, data_dir, batch_size=32, num_workers=4):
        """
        Initialize the data loader manager.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def get_dataloaders(self):
        """
        Create PyTorch DataLoader objects for train, dev, and test splits.

        Returns:
            tuple: (train_loader, dev_loader, test_loader)
        """
        # Create Dataset instances for each split
        train_dataset = MELDDataset(self.data_dir, 'train')
        dev_dataset = MELDDataset(self.data_dir, 'dev')
        test_dataset = MELDDataset(self.data_dir, 'test')
        
        # Create DataLoader for training set with shuffling
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
        # Create DataLoader for development/validation set (no shuffling)
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        # Create DataLoader for test set (no shuffling)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        return train_loader, dev_loader, test_loader