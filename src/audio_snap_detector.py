"""Audio-based snap gesture detector using microphone input."""

import threading
import time
from collections import deque
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class AudioSnapDetector:
    """
    Detects finger snap sounds using microphone input and spectral analysis.

    Snaps produce distinctive high-frequency transient sounds (2-12kHz).
    This detector uses FFT to analyze the frequency spectrum and detect
    energy spikes in the snap frequency range.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        chunk_size: int = 2048,
        snap_freq_low: int = 2000,
        snap_freq_high: int = 12000,
        energy_threshold: float = 5.0,
        debounce_time: float = 0.5,
    ):
        """
        Initialize the audio snap detector.

        Args:
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of samples per audio chunk (~46ms at 44.1kHz)
            snap_freq_low: Lower bound of snap frequency range (Hz)
            snap_freq_high: Upper bound of snap frequency range (Hz)
            energy_threshold: Multiplier over noise floor to trigger snap
            debounce_time: Minimum time between snap detections (seconds)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.snap_freq_low = snap_freq_low
        self.snap_freq_high = snap_freq_high
        self.energy_threshold = energy_threshold
        self.debounce_time = debounce_time

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._snap_detected = False
        self._snap_lock = threading.Lock()
        self._last_snap_time = 0.0

        # Noise floor estimation (rolling average of recent energy levels)
        self._noise_history: deque = deque(maxlen=50)
        self._noise_floor = 0.0

        # Audio stream
        self._stream: Optional[sd.InputStream] = None

    def start(self) -> bool:
        """
        Start the audio listener in a background thread.

        Returns:
            True if started successfully, False if sounddevice unavailable
        """
        if not SOUNDDEVICE_AVAILABLE:
            print("Warning: sounddevice not available. Audio snap detection disabled.")
            return False

        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Stop the audio listener."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def check_snap(self) -> bool:
        """
        Check if a snap was detected since last check.

        Returns:
            True if snap detected, False otherwise
        """
        with self._snap_lock:
            if self._snap_detected:
                self._snap_detected = False
                return True
            return False

    def _audio_loop(self):
        """Background thread that listens for audio and detects snaps."""
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size,
            )
            self._stream.start()

            while self._running:
                # Read audio chunk
                audio_data, overflowed = self._stream.read(self.chunk_size)
                if overflowed:
                    continue

                # Flatten to 1D
                audio = audio_data.flatten()

                # Analyze for snap
                if self._detect_snap(audio):
                    current_time = time.time()
                    # Debounce
                    if current_time - self._last_snap_time >= self.debounce_time:
                        with self._snap_lock:
                            self._snap_detected = True
                        self._last_snap_time = current_time

        except Exception as e:
            print(f"Audio snap detector error: {e}")
        finally:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

    def _detect_snap(self, audio: np.ndarray) -> bool:
        """
        Analyze audio chunk for snap sound using spectral analysis.

        Args:
            audio: Audio samples (float32, -1 to 1)

        Returns:
            True if snap detected
        """
        # Compute FFT
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)
        magnitudes = np.abs(fft)

        # Find frequency bins in snap range
        snap_mask = (freqs >= self.snap_freq_low) & (freqs <= self.snap_freq_high)
        snap_energy = np.sum(magnitudes[snap_mask] ** 2)

        # Total energy for comparison
        total_energy = np.sum(magnitudes ** 2)

        # Update noise floor (use total energy as baseline)
        self._noise_history.append(total_energy)
        if len(self._noise_history) >= 10:
            # Use median for robustness against outliers
            self._noise_floor = np.median(list(self._noise_history))

        # Skip if noise floor not established
        if self._noise_floor <= 0:
            return False

        # Check if snap frequency energy exceeds threshold
        # Snap sounds have high energy ratio in the snap frequency band
        if total_energy > 0:
            snap_ratio = snap_energy / total_energy
            energy_ratio = total_energy / self._noise_floor

            # Snap criteria:
            # 1. Overall energy spike (loud transient)
            # 2. High proportion of energy in snap frequency range
            if energy_ratio > self.energy_threshold and snap_ratio > 0.3:
                return True

        return False

    def get_debug_info(self) -> dict:
        """Get debug information about detector state."""
        return {
            "running": self._running,
            "noise_floor": self._noise_floor,
            "last_snap_time": self._last_snap_time,
            "sounddevice_available": SOUNDDEVICE_AVAILABLE,
        }
