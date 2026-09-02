from __future__ import annotations

import torch
import torch.nn as nn


class CPUAutoregressiveEngine(nn.Module):
    """A clean, high-performance neural language model core.

    Optimized specifically for high-throughput CPU operations: a learned
    embedding table, a GRU recurrent core (long-range tracking without
    quadratic transformer overhead), and a linear projection head.

    Weight initialization note: the parameters are initialized from PyTorch's
    standard random distributions (uniform/normal), NOT zeros. A zero-
    initialized network cannot learn -- every hidden unit would receive an
    identical gradient and stay symmetric forever, so the model could never
    differentiate features. "Blank slate" here means "no pretrained weights,
    trained entirely from scratch on this corpus", which is what random
    initialization provides.
    """

    def __init__(self, vocab_size: int = 16000, embedding_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Continuous learned weight tensors (the real synapses).
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Gated Recurrent Unit: long-range tracking without quadratic cost.
        self.rnn_core = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Output prediction projection head.
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_token_tensor: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the forward activation pass over continuous tensor tracks."""
        # 1. Discrete token ids -> continuous vector space.
        embedded = self.embedding(input_token_tensor)

        # 2. Sequence transformation through the recurrent core.
        rnn_out, next_hidden = self.rnn_core(embedded, hidden_state)

        # 3. Project hidden space back to vocabulary logits.
        logits = self.output_head(rnn_out)

        return logits, next_hidden

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
