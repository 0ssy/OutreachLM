from __future__ import annotations

import numpy as np


class FrontendAudioQuantizer:
    """Map a raw mono waveform to discrete token ids.

    Each `stride_ms` slice of audio becomes one token. The frame is converted
    to a log-magnitude spectrum (via rFFT) rather than being fed as raw
    amplitudes, because raw time-domain samples are dominated by phase and
    would hash near-randomly for perceptually identical sounds. The spectrum
    is then reduced with the same sign-based LSH scheme used by the visual
    quantizer, so both modalities share one well-understood mechanism.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        stride_ms: int = 20,
        codebook_bits: int = 12,
        base_vocab_offset: int = 8096,
        seed: int = 1337,
    ) -> None:
        self.sample_rate = sample_rate
        self.stride_ms = stride_ms
        self.frame_length = int(sample_rate * stride_ms / 1000)
        self.codebook_bits = codebook_bits
        self.base_vocab_offset = base_vocab_offset
        self.codebook_size = 2**codebook_bits

        spectrum_bins = self.frame_length // 2 + 1
        rng = np.random.default_rng(seed)
        self.audio_projection = rng.standard_normal(
            (spectrum_bins, codebook_bits)
        ).astype(np.float32) * 0.02
        self._bit_weights = (1 << np.arange(codebook_bits)).astype(np.int64)
        self._window = np.hanning(self.frame_length).astype(np.float32)

    @property
    def vocab_band(self) -> tuple[int, int]:
        return (self.base_vocab_offset, self.base_vocab_offset + self.codebook_size - 1)

    def process_audio_to_tokens(self, waveform: np.ndarray) -> list[int]:
        """Slice a 1-D waveform into frames and emit one token id per frame."""
        if waveform.ndim != 1:
            raise ValueError("expected a 1-D mono waveform")

        n = self.frame_length
        frame_count = waveform.shape[0] // n
        if frame_count == 0:
            return []

        frames = waveform[: frame_count * n].reshape(frame_count, n).astype(np.float32)
        frames = frames * self._window

        spectrum = np.abs(np.fft.rfft(frames, axis=1))
        spectrum = np.log1p(spectrum)
        spectrum -= spectrum.mean(axis=1, keepdims=True)

        projected = spectrum @ self.audio_projection
        bits = (projected > 0).astype(np.int64)
        codes = bits @ self._bit_weights
        return (codes + self.base_vocab_offset).tolist()
