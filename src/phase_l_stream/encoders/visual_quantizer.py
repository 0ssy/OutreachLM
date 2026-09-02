from __future__ import annotations

import numpy as np


class FrontendVisualQuantizer:
    """Map raw image pixels to discrete token ids before they reach the model.

    Design note on a defect in the original blueprint implementation:

        quantized_id = int(np.argmax(latent_vector) % 1000) + base_vocab_offset

    `np.argmax` over a length-`embedding_dim` (128) vector always returns a
    value in [0, 127], so `% 1000` never wraps and the quantizer could only
    ever emit 128 distinct ids -- not the ~1000 the expression implies. Worse,
    argmax of a random projection is dominated by whichever projection column
    happens to have the largest norm, so the mapping is close to constant
    across different images: nearly every patch collapses onto the same few
    ids, destroying the visual signal entirely.

    This implementation instead uses sign-based random projection to a
    Locality-Sensitive Hash (LSH). Each patch is projected onto `codebook_bits`
    random hyperplanes and the sign pattern forms the code. That yields
    2**codebook_bits distinct buckets, uses the whole projection (not just its
    argmax), and is locality sensitive: visually similar patches land in the
    same or nearby buckets, which is the property a quantizer actually needs.
    """

    def __init__(
        self,
        patch_dim: int = 16,
        codebook_bits: int = 12,
        base_vocab_offset: int = 4000,
        seed: int = 1337,
    ) -> None:
        self.patch_dim = patch_dim
        self.codebook_bits = codebook_bits
        self.base_vocab_offset = base_vocab_offset
        self.codebook_size = 2**codebook_bits
        rng = np.random.default_rng(seed)
        # Hyperplanes for sign-based LSH over a flattened RGB patch.
        self.visual_projection = rng.standard_normal(
            (patch_dim * patch_dim * 3, codebook_bits)
        ).astype(np.float32) * 0.02
        self._bit_weights = (1 << np.arange(codebook_bits)).astype(np.int64)

    @property
    def vocab_band(self) -> tuple[int, int]:
        return (self.base_vocab_offset, self.base_vocab_offset + self.codebook_size - 1)

    def process_image_to_tokens(self, raw_image_pixel_matrix: np.ndarray) -> list[int]:
        """Slice an HxWx3 image into patches and emit one token id per patch.

        Vectorized over all patches at once: the per-patch Python loop in the
        blueprint made throughput scale with patch count, which would stall the
        training cores on any realistically sized image.
        """
        if raw_image_pixel_matrix.ndim != 3 or raw_image_pixel_matrix.shape[2] != 3:
            raise ValueError("expected an (H, W, 3) RGB pixel matrix")

        height, width, _ = raw_image_pixel_matrix.shape
        p = self.patch_dim
        rows, cols = height // p, width // p
        if rows == 0 or cols == 0:
            return []

        cropped = raw_image_pixel_matrix[: rows * p, : cols * p, :]
        # (rows, p, cols, p, 3) -> (rows*cols, p*p*3)
        patches = (
            cropped.reshape(rows, p, cols, p, 3)
            .transpose(0, 2, 1, 3, 4)
            .reshape(rows * cols, p * p * 3)
            .astype(np.float32)
        )
        # Center pixel values so the sign pattern is informative rather than
        # dominated by overall brightness.
        patches -= patches.mean(axis=1, keepdims=True)

        projected = patches @ self.visual_projection
        bits = (projected > 0).astype(np.int64)
        codes = bits @ self._bit_weights
        return (codes + self.base_vocab_offset).tolist()
