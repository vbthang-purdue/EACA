from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np

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

    def extract_mfcc(self, audio, sr):

        return mfccs
    
    def extract_prosodic_features(self, audio, sr):


class MultimodalFeatureExtractor:
    # Main feature extraction combining text and audio

    def extract_features(self, sample, dialogue_history=None):
        features = {}

        return features
