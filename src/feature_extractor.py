# src/feature_extractor.py
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
import librosa
from typing import Optional, List, Dict, Tuple

class TextFeatureExtractor:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        self.model.eval()

    def get_utterance_embedding(self, text):
        # Handle list input
        if isinstance(text, list):
            text = " ".join(text)
            
        # tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )

        # embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state

        # mean pooling
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings.squeeze().numpy()
    
    def get_contextual_embeddings(self, current_utterance: str, previous_utterances: List[str], context_window: int = 3) -> np.ndarray:
        # Handle list input for current_utterance
        if isinstance(current_utterance, list):
            current_utterance = " ".join(current_utterance)
        
        # select last n utterances as context
        context_utterances = previous_utterances[-context_window:]
        
        # concatenate context with [SEP] token
        context_text = " ".join(context_utterances)
        
        # combine context and current utterance
        full_text = f"{context_text} [SEP] {current_utterance}" if context_text else current_utterance
        
        # embedding for the combined sequence
        return self.get_utterance_embedding(full_text)

class AudioFeatureExtractor:
    def __init__(self, n_mfcc: int = 40, frame_length: int = 25, hop_length: int = 10):
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.hop_length = hop_length

    def extract_mfcc(self, audio, sr):
        # ms -> # of samples
        frame_length_samples = int(self.frame_length * sr / 1000)
        hop_length_samples = int(self.hop_length * sr / 1000)
        
        # computing MFCCs using librosa
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=frame_length_samples,
            hop_length=hop_length_samples
        )
        return mfccs
    
    def extract_prosodic_features(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        # Pitch (F0) using pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )

        # RMS energy
        frame_length_samples = int(25 * sr / 1000)
        hop_length_samples = int(10 * sr / 1000)

        # energy = RMS amplitude per frame
        rms = librosa.feature.rms(  # Fixed variable name (was ms)
            y=audio,
            frame_length=frame_length_samples,
            hop_length=hop_length_samples
        )
        
        return f0, rms.squeeze()
    
    def get_statistical_features(self, features: np.ndarray) -> np.ndarray:
        if features is None or len(features) == 0:
            return np.zeros(4)
            
        # replace NaN values with zeros
        features = np.nan_to_num(features, nan=0.0)
        
        # compute statistical moments across time dimension
        stats = np.array([
            np.mean(features),
            np.std(features),
            np.max(features),
            np.min(features)
        ])
        
        return stats
    
    def extract_audio_features(self, audio: Optional[np.ndarray], sr: int) -> np.ndarray:
        # NO FILE
        if audio is None or len(audio) == 0:
            return np.zeros(self.n_mfcc * 4 + 8)
        
        # extract MFCCs
        mfccs = self.extract_mfcc(audio, sr)
        
        # extract prosodic features
        pitch, energy = self.extract_prosodic_features(audio, sr)
        
        # aggregate MFCC stats across all coefficients
        mfcc_stats = []
        for i in range(self.n_mfcc):
            mfcc_stats.extend(self.get_statistical_features(mfccs[i]))
        
        # aggregate prosodic feature statistics
        pitch_stats = self.get_statistical_features(pitch)
        energy_stats = self.get_statistical_features(energy)
        
        # concatenate all audio features into single vector
        audio_features = np.concatenate([mfcc_stats, pitch_stats, energy_stats])
        
        return audio_features

class MultimodalFeatureExtractor:
    def __init__(self):
        self.text_extractor = TextFeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()

    def extract_features(self, sample: Dict, dialogue_history: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        features = {}
        
        print(f"🔍 Processing utterance: {sample.get('text', '')[:50]}...")

        # TEXT FEATURE EXTRACTION
        text_input = sample.get('tokenized_text')
        if text_input is None:
            text_input = sample.get('cleaned_text', sample.get('text', ''))
        
        if isinstance(text_input, list):
            text_input = " ".join(text_input)

        print(f"📝 Extracting text features...")
        features['text'] = self.text_extractor.get_utterance_embedding(text_input)
        print(f"   Text features: {len(features['text'])} dimensions")

        # AUDIO FEATURE EXTRACTION
        processed_audio = sample.get('processed_audio')
        
        if processed_audio is not None:
            if isinstance(processed_audio, tuple) and len(processed_audio) == 2:
                audio, sr = processed_audio
                print(f"🎵 Extracting audio features from {len(audio)} samples...")
                features['audio'] = self.audio_extractor.extract_audio_features(audio, sr)
                print(f"   Audio features: {len(features['audio'])} dimensions")
            else:
                features['audio'] = np.zeros(168)
        else:
            features['audio'] = np.zeros(168)
            print("   No audio data, using zero vector")

        # COMBINE FEATURES
        features['multimodal'] = np.concatenate([features['text'], features['audio']])
        print(f"🌐 Combined multimodal features: {len(features['multimodal'])} dimensions")

        return features