import librosa
import numpy as np
from typing import Dict, Any

class AcousticProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        # Load audio: sr=None keeps native sampling rate, mono=True for analysis
        self.y, self.sr = librosa.load(file_path, sr=None, mono=True)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)

    def analyze(self) -> Dict[str, Any]:
        """Runs the full acoustic analysis suite."""
        return {
            "speaking_rate": self.calculate_speaking_rate(),
            "pause_duration": self.calculate_pauses(),
            "jitter": self.calculate_jitter(),
            "shimmer": self.calculate_shimmer(),
            "duration": round(self.duration, 2)
        }

    def calculate_speaking_rate(self) -> Dict[str, Any]:
        """
        Estimates speaking rate. 
        Since we don't have the transcript yet, we estimate 'active speech' 
        time vs total time. In a full system, this would use the transcript word count.
        """
        # Detect non-silent intervals
        intervals = librosa.effects.split(self.y, top_db=20)
        active_duration = sum([end - start for start, end in intervals]) / self.sr
        
        # Heuristic: Average English speaker is ~130-150 WpM
        # We estimate WpM based on the ratio of speech to silence
        # Actual WpM = (Words / Active Duration) * 60
        # For the prototype, we return the speech-to-silence ratio as a proxy
        return {
            "active_speech_seconds": round(active_duration, 2),
            "silence_seconds": round(self.duration - active_duration, 2),
            "speech_ratio": round(active_duration / self.duration, 2) if self.duration > 0 else 0
        }

    def calculate_pauses(self) -> Dict[str, Any]:
        """Detects and measures pause durations."""
        intervals = librosa.effects.split(self.y, top_db=20)
        pauses = []
        
        for i in range(len(intervals) - 1):
            pause_start = intervals[i][1]
            pause_end = intervals[i+1][0]
            duration = (pause_end - pause_start) / self.sr
            pauses.append(duration)
            
        return {
            "total_pauses": len(pauses),
            "average_pause_duration": round(np.mean(pauses), 3) if pauses else 0,
            "max_pause_duration": round(np.max(pauses), 3) if pauses else 0,
            "pause_list": [round(p, 3) for p in pauses]
        }

    def calculate_jitter(self) -> float:
        """
        Calculates Jitter (local).
        Jitter is the absolute difference between consecutive periods.
        """
        # Extract fundamental frequency (F0) using pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            self.y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=self.sr
        )
        
        # Filter for voiced regions only
        f0_voiced = f0[~np.isnan(f0)]
        
        if len(f0_voiced) < 2:
            return 0.0
            
        # Period T = 1/f
        periods = 1.0 / f0_voiced
        
        # Local Jitter = average of |T(i) - T(i-1)| / average(T)
        diffs = np.abs(np.diff(periods))
        jitter = np.mean(diffs) / np.mean(periods)
        
        return round(jitter * 100, 3) # Return as percentage

    def calculate_shimmer(self) -> float:
        """
        Calculates Shimmer (local).
        Shimmer is the amplitude variation between consecutive periods.
        """
        # Use the same voicing mask as jitter
        f0, voiced_flag, _ = librosa.pyin(
            self.y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=self.sr
        )
        
        # Get amplitudes of voiced segments
        # For a simple prototype, we use the RMS energy of the windows
        # corresponding to the voiced regions.
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=self.y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Map F0 frames to RMS frames
        # pyin uses a specific hop length (usually 128)
        # We interpolate or align them. For this prototype, we'll use the raw RMS diffs.
        if len(rms) < 2:
            return 0.0
            
        diffs = np.abs(np.diff(rms))
        shimmer = np.mean(diffs) / np.mean(rms)
        
        return round(shimmer * 100, 3) # Return as percentage
