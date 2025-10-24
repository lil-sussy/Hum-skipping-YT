import librosa
import numpy as np
from typing import List, Dict, Tuple
import json
from pathlib import Path

class AudioClassifier:
    def __init__(self):
        # Parameters tuned for speech/hum/silence classification
        self.frame_size = 2048
        self.hop_length = 512
        self.sr = 16000
        
        # Thresholds for classification
        self.silence_threshold = -30  # dB
        self.hum_threshold = -20      # dB
        
    def extract_features(self, audio: np.ndarray, frame_size: int, hop_length: int) -> Dict[str, float]:
        """Extract relevant features from audio segment."""
        # Calculate energy
        energy = np.mean(np.abs(audio))
        
        # Calculate spectral features
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=frame_size,
            hop_length=hop_length,
            n_mels=128
        )
        
        # Calculate spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(
            S=mel_spectrogram,
            sr=self.sr
        )[0].mean()
        
        # Calculate zero-crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0].mean()
        
        return {
            'energy': energy,
            'spectral_rolloff': rolloff,
            'zero_crossing_rate': zero_crossing_rate
        }
    
    def classify_segment(self, features: Dict[str, float]) -> str:
        """Classify audio segment based on extracted features."""
        # Convert energy to dB scale
        energy_db = 20 * np.log10(features['energy'] + 1e-10)
        
        # Classify based on thresholds
        if energy_db < self.silence_threshold:
            return 'silence'
        elif energy_db > self.hum_threshold and features['zero_crossing_rate'] > 0.05:
            return 'speech'
        else:
            return 'hum'
    
    def process_audio_file(self, audio_path: str, segment_duration: float = 0.5) -> List[Dict]:
        """
        Process audio file and return classifications for each segment.
        
        Args:
            audio_path: Path to audio file
            segment_duration: Duration of each segment in seconds
            
        Returns:
            List of dictionaries containing segment information
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sr)
        
        # Calculate segment size
        segment_size = int(segment_duration * self.sr)
        
        # Initialize results
        results = []
        
        # Process segments
        for i in range(0, len(audio), segment_size):
            segment = audio[i:i + segment_size]
            
            if len(segment) < segment_size:
                break
                
            # Extract features and classify
            features = self.extract_features(segment, self.frame_size, self.hop_length)
            classification = self.classify_segment(features)
            
            # Calculate timestamps
            start_time = i / sr
            end_time = (i + segment_size) / sr
            
            results.append({
                'start_time': float(start_time),
                'end_time': float(end_time),
                'classification': classification,
                'features': features
            })
        
        return results
    
    def save_json(self, results: List[Dict], output_path: str):
        """Save classification results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)