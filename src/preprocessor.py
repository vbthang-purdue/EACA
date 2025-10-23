# src/preprocessor.py
import string
import librosa
import os
import numpy as np
import subprocess
import tempfile
from typing import Optional
from .contractions import expand_contractions

class TextPreprocessor:
    @staticmethod
    def clean_text(text):
        """Clean and preprocess text."""
        if not isinstance(text, str):
            return ""
        text = expand_contractions(text.lower())
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.strip()
    
    @staticmethod
    def tokenize(text):
        """Tokenize text into words."""
        if not text:
            return []
        return text.split()

class AudioPreprocessor:
    @staticmethod
    def get_ffmpeg_path():
        """Get FFmpeg path using absolute path to project."""
        # Absolute path to ffmpeg.exe
        ffmpeg_path = r"C:\Users\User\Documents\emotion_erc\ffmpeg\bin\ffmpeg.exe"
        
        if os.path.exists(ffmpeg_path):
            return ffmpeg_path
        else:
            print(f"❌ FFmpeg not found at: {ffmpeg_path}")
            # Try to find it automatically
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            auto_path = os.path.join(project_root, 'ffmpeg', 'bin', 'ffmpeg.exe')
            if os.path.exists(auto_path):
                print(f"✅ Found FFmpeg at: {auto_path}")
                return auto_path
            return None

    @staticmethod
    def check_ffmpeg_available():
        """Check if FFmpeg is available."""
        try:
            ffmpeg_path = AudioPreprocessor.get_ffmpeg_path()
            if not ffmpeg_path:
                return False
                
            # Hide the command window on Windows
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            result = subprocess.run(
                [ffmpeg_path, '-version'], 
                capture_output=True, 
                text=True,
                startupinfo=startupinfo
            )
            available = result.returncode == 0
            if available:
                print("✅ FFmpeg is available!")
            return available
        except Exception as e:
            print(f"❌ FFmpeg check failed: {e}")
            return False

    @staticmethod
    def extract_audio_from_mp4(mp4_path: str) -> str:
        """Extract audio from MP4 and return path to temporary WAV file."""
        ffmpeg_path = AudioPreprocessor.get_ffmpeg_path()
        if not ffmpeg_path:
            print("❌ FFmpeg not available")
            return None
        
        try:
            # Create temporary WAV file
            temp_dir = tempfile.gettempdir()
            base_name = os.path.basename(mp4_path).replace('.mp4', '.wav')
            output_wav = os.path.join(temp_dir, base_name)
            
            cmd = [
                ffmpeg_path,
                '-i', mp4_path,      # Input file
                '-ac', '1',          # Mono audio
                '-ar', '16000',      # 16kHz sample rate
                '-vn',               # No video
                '-y',                # Overwrite output
                output_wav
            ]
            
            # Hide command window on Windows
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            print(f"🔧 Converting MP4 to WAV: {os.path.basename(mp4_path)}")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                startupinfo=startupinfo
            )
            
            if result.returncode == 0 and os.path.exists(output_wav):
                print(f"✅ Successfully converted: {os.path.basename(mp4_path)}")
                return output_wav
            else:
                print(f"❌ FFmpeg conversion failed: {result.stderr[:200]}...")
                return None
                
        except Exception as e:
            print(f"❌ FFmpeg extraction error: {e}")
            return None

    @staticmethod
    def load_audio(audio_path: str, desired_sr: int = 16000):
        """Load audio file with FFmpeg fallback for MP4."""
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            return None
        
        # For MP4 files, use FFmpeg extraction
        if audio_path.lower().endswith('.mp4'):
            if AudioPreprocessor.check_ffmpeg_available():
                wav_path = AudioPreprocessor.extract_audio_from_mp4(audio_path)
                
                if wav_path and os.path.exists(wav_path):
                    try:
                        # Load the extracted WAV file
                        audio, sr = librosa.load(wav_path, sr=desired_sr)
                        
                        # Clean up temporary file
                        try:
                            os.remove(wav_path)
                        except:
                            pass
                        
                        return audio, sr
                        
                    except Exception as e:
                        print(f"❌ Failed to load extracted WAV: {e}")
                        # Clean up on failure
                        try:
                            os.remove(wav_path)
                        except:
                            pass
            else:
                print(f"❌ FFmpeg not available for MP4: {audio_path}")
                return None
        
        # For non-MP4 files, try direct loading
        try:
            audio, sr = librosa.load(audio_path, sr=desired_sr)
            return audio, sr
        except Exception as e:
            print(f"❌ Direct loading failed: {e}")
            return None

    @staticmethod
    def clean_audio(audio, top_db: int = 20):
        """Clean and normalize audio."""
        if audio is None or len(audio) == 0:
            return audio
            
        try:
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Trim silence
            trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
            
            return trimmed_audio if len(trimmed_audio) > 0 else audio
            
        except Exception as e:
            print(f"Audio cleaning failed: {e}")
            return audio

    @staticmethod
    def preprocess_audio_given_path(audio_path: str, desired_sr: int = 16000, top_db: int = 20):
        """Complete audio preprocessing pipeline."""
        if not audio_path or not os.path.exists(audio_path):
            print(f"❌ Audio path invalid: {audio_path}")
            return None
            
        result = AudioPreprocessor.load_audio(audio_path, desired_sr)
        if result is None:
            return None
            
        audio, sr = result
        if audio is not None:
            audio = AudioPreprocessor.clean_audio(audio, top_db)
        
        return audio, sr


    # Added to AudioPreprocessor class
    @staticmethod
    def analyze_audio(audio, sr):
        """Analyze audio and return detailed information."""
        if audio is None or len(audio) == 0:
            return "No audio data"
        
        duration = len(audio) / sr
        analysis = {
            'duration_seconds': f"{duration:.2f}",
            'sample_rate': sr,
            'total_samples': len(audio),
            'amplitude_range': f"{np.min(audio):.3f} to {np.max(audio):.3f}",
            'rms_energy': f"{np.sqrt(np.mean(audio**2)):.4f}"
        }
        
        # Try to extract some basic audio features for analysis
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            analysis['spectral_centroid_mean'] = f"{np.mean(spectral_centroid):.1f} Hz"
        except:
            analysis['spectral_centroid_mean'] = "N/A"
        
        return analysis
    
class DataPreprocessor:
    @staticmethod
    def preprocess_sample(sample):
        """Preprocess a single data sample."""
        # Text preprocessing
        cleaned_text = TextPreprocessor.clean_text(sample['text'])
        tokenized_text = TextPreprocessor.tokenize(cleaned_text)
        
        # Audio preprocessing
        audio_path = sample.get('audio_path')
        processed_audio = None
        if audio_path:
            processed_audio = AudioPreprocessor.preprocess_audio_given_path(audio_path)
        
        # Update sample
        sample['cleaned_text'] = cleaned_text
        sample['tokenized_text'] = tokenized_text
        sample['processed_audio'] = processed_audio

        return sample