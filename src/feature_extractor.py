from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
import librosa

class TextFeatureExtractor:
    # Text feature extraction via DistilBERT

    def __init__(self, model_name='distillbert-base-uncased'):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        self.model.eval()

    def get_utterance_embedding(self, text):
        # tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt', # return pytorch tensors
            padding=True,        # pad to longest sequence in batch
            truncation=True,     #trancate sequences longer then max_length
            max_length=128
        )

        # embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # shape: [batch_size=1, sequence_length, hidden_size=768]
            last_hidden_state = outputs.last_hidden_state

        # mean pooling application
        attention_mask = inputs['attention_mask'] #ensuring padding dokens don't affect the mean
        # expand mask to match embedding dimensions where [1, seq_len] -> [1, seq_len, 768]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1) # sum of all token embeddings
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9) # num of non-padded tokens per seq

        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings.squeeze().numpy()
    

    def get_contextual_embeddings(self, current_utterance: str, previous_utterances: List[str], context_window: int = 3) -> np.ndarray:
        # handle list input for current_utterance
        if isinstance(current_utterance, list):
            current_utterance = " ".join(current_utterance)
        
        # select last n utterances as context
        context_utterances = previous_utterances[-context_window:]
        
        # concatenate context with [SEP] token
        context_text = " ".join(context_utterances)
        
        # combine context and current utterance
        # in format "context_utt1 context_utt2 [SEP] current_utterance"
        full_text = f"{context_text} [SEP] {current_utterance}" if context_text else current_utterance
        
        # embedding for the combined sequence
        return self.get_utterance_embedding(full_text)
    

class AudioFeatureExtractor:
    # Audio feature extraction via MFCC and prosodic features

    def __init__(self, n_mfcc: int = 40, frame_length: int = 25, hop_length: int = 10):
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length  # Window size for FFT
        self.hop_length = hop_length      # Step size between frames


    def extract_mfcc(self, audio, sr):
        # ms -> # of samples
        frame_length_samples = int(self.frame_length * sr / 1000)
        hop_length_samples = int(self.hop_length * sr / 1000)
        
        # computing MFCCs using librosa
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,           # Number of coefficients
            n_fft=frame_length_samples,    # FFT window size
            hop_length=hop_length_samples  # Stride between frames
        )

        return mfccs
    
    
    def extract_prosodic_features(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        # Pitch (F0) using pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'), # 65.4 Hz
            fmax=librosa.note_to_hz('C7'), # 2093 Hz i.e human speech
            sr=sr
        )

        # rms frame parameters per mfcc framing
        frame_length_samples = int(25 * sr / 1000)
        hop_length_samples = int(10 * sr / 1000)

        # energy = RMS amplitude per frame
        ms = librosa.feature.rms(
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
            np.std(features),   # var
            np.max(features),
            np.min(features)
        ])
        
        return stats
    

    def extract_audio_features(self, audio: Optional[np.ndarray], sr: int) -> np.ndarray:
        # NO FILE
        if audio is None or len(audio) == 0:
            return np.zeros(self.n_mfcc * 4 + 8)  # 40*4 (MFCC) + 4 (pitch) + 4 (energy)
        
        # extract MFCCs
        mfccs = self.extract_mfcc(audio, sr)
        
        # extract prosodic features
        pitch, energy = self.extract_prosodic_features(audio, sr)
        
        # aggregate MFCC stats across all coefficients
        mfcc_stats = []
        for i in range(self.n_mfcc):
            # get stats for each MFCC coefficient across time
            mfcc_stats.extend(self.get_statistical_features(mfccs[i]))
        
        # aggregate prosodic feature statistics
        pitch_stats = self.get_statistical_features(pitch)
        energy_stats = self.get_statistical_features(energy)
        
        # concatenate all audio features into single vector
        audio_features = np.concatenate([mfcc_stats, pitch_stats, energy_stats])
        
        return audio_features


class MultimodalFeatureExtractor:
    # Main feature extraction combining text and audio

    def extract_features(self, sample, dialogue_history=None):
        features = {}

        return features
