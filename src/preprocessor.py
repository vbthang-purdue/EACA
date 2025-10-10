import contractions
import string
import librosa
from os import path
from types import Optional

class TextPreprocessor:
    def cleanText(text):
        text = contractions.fix(text.lower())
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    def tokenize(text):
        return text.split()
    
class AudioPreprocessor:
    def loadAudio(audioPath, desiredSR : Optional[int]):
        desiredSR = desiredSR or 16000
        try:
            audio = librosa.load(audioPath, sr=desiredSR)
            return audio
        except Exception as e:
            print(audioPath + " failed. Error: "+ e)
            return None
    def cleanAudio(audio, topDB : Optional[int]):
        topDB = topDB or 20
        audio = librosa.util.normalize(audio)
        if len(audio) == 0:
            print("um dude... your audio has no length.....")
            return audio
        trimmedAudio = librosa.effects.trim(audio, top_db=topDB)
        if len(trimmedAudio) == 0:
            return audio
        return trimmedAudio
    def preprocessAudioGivenPath(audioPath, desiredSR, topDB):
        if path.exists(audioPath):
            audio = AudioPreprocessor.loadAudio(audio, desiredSR)
            if audio:
                audio = AudioPreprocessor.cleanAudio(audio, topDB)
            
            return audio
        return None

class DataPreprocessor:
    def preprocessSample(sample):
        tokenizedText = TextPreprocessor.tokenize(TextPreprocessor.cleanText(sample['text']))
        audio = AudioPreprocessor.preprocessAudioGivenPath(sample["audio_path"])
        if tokenizedText and tokenizedText != '': #im actually not sure if this is even possible lol
            tokenizedText = None
        sample['tokenized_text'] = tokenizedText
        sample['processed_audio'] = audio

        return sample
            