import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# OUTREACHLM BACKPROPAGATION LAB
# ============================================================

# ============================================================
# 1. GRADIENT DESCENT
# ============================================================


def gradient_descent_demo():

    print()
    print("=" * 60)
    print("1. GRADIENT DESCENT")
    print("=" * 60)

    x = torch.tensor(
        2.0
    )

    target = torch.tensor(
        10.0
    )

    w = torch.tensor(
        3.0,
        requires_grad=True
    )

    learning_rate = 0.1

    for step in range(5):

        y = w * x

        loss = (
            y - target
        ) ** 2

        loss.backward()

        with torch.no_grad():

            w -= (
                learning_rate
                * w.grad
            )

        w.grad.zero_()

        print(
            f"Step {step + 1} | "
            f"w={w.item():.6f} | "
            f"loss={loss.item():.6f}"
        )

# ============================================================
# 2. MANUAL BACKPROPAGATION
# ============================================================


def manual_backprop_demo():

    print()
    print("=" * 60)
    print("2. MANUAL BACKPROPAGATION")
    print("=" * 60)

    # --------------------------------------------------
    # Forward:
    #
    # z = wx + b
    # y = z
    # L = (y - target)^2
    # --------------------------------------------------

    x = 2.0
    w = 3.0
    b = 1.0

    target = 10.0

    z = (
        w * x
        + b
    )

    y = z

    loss = (
        y - target
    ) ** 2

    # --------------------------------------------------
    # Backward
    #
    # dL/dy
    # --------------------------------------------------

    dL_dy = (
        2.0
        * (y - target)
    )

    # y = z
    #
    # dy/dz = 1

    dy_dz = 1.0

    # z = wx + b
    #
    # dz/dw = x
    # dz/db = 1

    dz_dw = x
    dz_db = 1.0

    # Chain rule

    dL_dw = (
        dL_dy
        * dy_dz
        * dz_dw
    )

    dL_db = (
        dL_dy
        * dy_dz
        * dz_db
    )

    print(
        f"z       = {z:.6f}"
    )

    print(
        f"loss    = {loss:.6f}"
    )

    print(
        f"dL/dw   = {dL_dw:.6f}"
    )

    print(
        f"dL/db   = {dL_db:.6f}"
    )

# ============================================================
# 3. LINEAR LAYER
# ============================================================


def linear_backprop_demo():

    print()
    print("=" * 60)
    print("3. BACKPROP THROUGH LINEAR")
    print("=" * 60)

    x = torch.tensor(
        [[2.0, -1.0]],
        requires_grad=True
    )

    W = torch.tensor(
        [
            [3.0, 4.0],
            [5.0, -2.0]
        ],
        requires_grad=True
    )

    b = torch.tensor(
        [1.0, -1.0],
        requires_grad=True
    )

    target = torch.tensor(
        [[2.0, 3.0]]
    )

    # Forward

    y = (
        x @ W.T
        + b
    )

    loss = (
        (y - target) ** 2
    ).mean()

    # Backward

    loss.backward()

    print()
    print("Output:")
    print(y.detach())

    print()
    print("dL/dW:")
    print(W.grad)

    print()
    print("dL/db:")
    print(b.grad)

    print()
    print("dL/dx:")
    print(x.grad)

# ============================================================
# 4. MLP
# ============================================================


def mlp_backprop_demo():

    print()
    print("=" * 60)
    print("4. BACKPROP THROUGH MLP")
    print("=" * 60)

    torch.manual_seed(0)

    model = nn.Sequential(

        nn.Linear(
            3,
            4
        ),

        nn.Tanh(),

        nn.Linear(
            4,
            2
        )

    )

    x = torch.tensor(
        [[0.5, -1.0, 2.0]]
    )

    target = torch.tensor(
        [[1.0, 0.0]]
    )

    y = model(
        x
    )

    loss = (
        (y - target) ** 2
    ).mean()

    loss.backward()

    print(
        f"Loss = {loss.item():.6f}"
    )

    print()

    for name, parameter in model.named_parameters():

        print(
            f"{name:15s} | "
            f"gradient norm = "
            f"{parameter.grad.norm().item():.6f}"
        )

# ============================================================
# 5. LAYER NORMALIZATION
# ============================================================


def layer_norm_backprop_demo():

    print()
    print("=" * 60)
    print("5. BACKPROP THROUGH LAYER NORM")
    print("=" * 60)

    torch.manual_seed(0)

    x = torch.randn(
        2,
        4,
        requires_grad=True
    )

    gamma = torch.ones(
        4,
        requires_grad=True
    )

    beta = torch.zeros(
        4,
        requires_grad=True
    )

    # --------------------------------------------------
    # LayerNorm forward
    # --------------------------------------------------

    mean = x.mean(
        dim=-1,
        keepdim=True
    )

    variance = (
        (x - mean) ** 2
    ).mean(
        dim=-1,
        keepdim=True
    )

    inverse_std = torch.rsqrt(
        variance + 1e-5
    )

    normalized = (
        (x - mean)
        * inverse_std
    )

    y = (
        normalized
        * gamma
        + beta
    )

    # Loss

    loss = (
        y ** 2
    ).mean()

    # Backward

    loss.backward()

    print(
        f"Loss = {loss.item():.6f}"
    )

    print(
        "Input gradient norm = "
        f"{x.grad.norm().item():.6f}"
    )

    print()

    print(
        "Gamma gradient:"
    )

    print(
        gamma.grad
    )

    print()

    print(
        "Beta gradient:"
    )

    print(
        beta.grad
    )

# ============================================================
# 6. SOFTMAX
# ============================================================


def softmax_backprop_demo():

    print()
    print("=" * 60)
    print("6. BACKPROP THROUGH SOFTMAX")
    print("=" * 60)

    logits = torch.tensor(
        [1.0, 2.0, 0.5],
        requires_grad=True
    )

    target = torch.tensor(
        [0.0, 1.0, 0.0]
    )

    probabilities = torch.softmax(
        logits,
        dim=0
    )

    # Cross entropy

    loss = -(
        target
        * torch.log(probabilities)
    ).sum()

    loss.backward()

    print()
    print(
        "Probabilities:"
    )

    print(
        probabilities.detach()
    )

    print()

    print(
        "dL/dlogits:"
    )

    print(
        logits.grad
    )

    print()

    print(
        "For softmax + cross entropy:"
    )

    print(
        "dL/dlogits = probabilities - target"
    )

    print()

    print(
        probabilities.detach()
        - target
    )

# ============================================================
# 7. Q/K/V
# ============================================================


def qkv_backprop_demo():

    print()
    print("=" * 60)
    print("7. BACKPROP THROUGH Q / K / V")
    print("=" * 60)

    torch.manual_seed(0)

    x = torch.randn(
        1,
        3,
        4,
        requires_grad=True
    )

    query = nn.Linear(
        4,
        4
    )

    key = nn.Linear(
        4,
        4
    )

    value = nn.Linear(
        4,
        4
    )

    Q = query(
        x
    )

    K = key(
        x
    )

    V = value(
        x
    )

    # Simple downstream operation

    output = (
        Q * K
    ).mean() + V.mean()

    output.backward()

    print(
        "Input gradient norm:",
        f"{x.grad.norm().item():.6f}"
    )

    print(
        "Q projection gradient:",
        f"{query.weight.grad.norm().item():.6f}"
    )

    print(
        "K projection gradient:",
        f"{key.weight.grad.norm().item():.6f}"
    )

    print(
        "V projection gradient:",
        f"{value.weight.grad.norm().item():.6f}"
    )

# ============================================================
# 8. FULL SELF-ATTENTION
# ============================================================


def attention_backprop_demo():

    print()
    print("=" * 60)
    print("8. BACKPROP THROUGH SELF-ATTENTION")
    print("=" * 60)

    torch.manual_seed(0)

    batch_size = 1
    sequence_length = 4
    embedding_dim = 8
    num_heads = 2

    head_dim = (
        embedding_dim
        // num_heads
    )

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
        requires_grad=True
    )

    query = nn.Linear(
        embedding_dim,
        embedding_dim
    )

    key = nn.Linear(
        embedding_dim,
        embedding_dim
    )

    value = nn.Linear(
        embedding_dim,
        embedding_dim
    )

    output_projection = nn.Linear(
        embedding_dim,
        embedding_dim
    )

    # --------------------------------------------------
    # Q K V
    # --------------------------------------------------

    Q = query(x)

    K = key(x)

    V = value(x)

    # --------------------------------------------------
    # Split heads
    # --------------------------------------------------

    Q = Q.view(
        batch_size,
        sequence_length,
        num_heads,
        head_dim
    ).transpose(
        1,
        2
    )

    K = K.view(
        batch_size,
        sequence_length,
        num_heads,
        head_dim
    ).transpose(
        1,
        2
    )

    V = V.view(
        batch_size,
        sequence_length,
        num_heads,
        head_dim
    ).transpose(
        1,
        2
    )

    # --------------------------------------------------
    # Attention scores
    # --------------------------------------------------

    scores = (
        Q
        @ K.transpose(
            -2,
            -1
        )
    )

    scores = (
        scores
        / math.sqrt(head_dim)
    )

    # --------------------------------------------------
    # Causal mask
    # --------------------------------------------------

    mask = torch.tril(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool
        )
    )

    scores = scores.masked_fill(
        ~mask,
        float("-inf")
    )

    # --------------------------------------------------
    # Softmax
    # --------------------------------------------------

    attention_weights = torch.softmax(
        scores,
        dim=-1
    )

    # --------------------------------------------------
    # Weighted values
    # --------------------------------------------------

    context = (
        attention_weights
        @ V
    )

    # --------------------------------------------------
    # Combine heads
    # --------------------------------------------------

    context = context.transpose(
        1,
        2
    )

    context = context.contiguous()

    context = context.view(
        batch_size,
        sequence_length,
        embedding_dim
    )

    # --------------------------------------------------
    # Output projection
    # --------------------------------------------------

    output = output_projection(
        context
    )

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    loss = (
        output ** 2
    ).mean()

    # --------------------------------------------------
    # BACKPROP
    # --------------------------------------------------

    loss.backward()

    print()
    print(
        "Attention shape:"
    )

    print(
        attention_weights.shape
    )

    print()

    print(
        "Input gradient norm:",
        f"{x.grad.norm().item():.6f}"
    )

    print(
        "Q gradient norm:",
        f"{query.weight.grad.norm().item():.6f}"
    )

    print(
        "K gradient norm:",
        f"{key.weight.grad.norm().item():.6f}"
    )

    print(
        "V gradient norm:",
        f"{value.weight.grad.norm().item():.6f}"
    )

# ============================================================
# 9. EMBEDDING BACKPROP
# ============================================================


def embedding_backprop_demo():

    print()
    print("=" * 60)
    print("9. BACKPROP THROUGH TOKEN EMBEDDINGS")
    print("=" * 60)

    embedding = nn.Embedding(
        10,
        4
    )

    token_ids = torch.tensor(
        [[1, 3, 3, 7]]
    )

    vectors = embedding(
        token_ids
    )

    loss = (
        vectors ** 2
    ).mean()

    loss.backward()

    print(
        "Embedding output shape:"
    )

    print(
        vectors.shape
    )

    print()

    print(
        "Embedding rows receiving gradients:"
    )

    for token_id in range(10):

        gradient = (
            embedding.weight.grad[
                token_id
            ]
        )

        if gradient.abs().sum() > 0:

            print(
                token_id
            )

    print()

    print(
        "Token 3 appears twice, so its "
        "gradient receives contributions "
        "from both occurrences."
    )

# ============================================================
# 10. FULL TRANSFORMER BACKPROP
# ============================================================


class TinyTransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim=16,
        num_heads=4
    ):

        super().__init__()

        self.embedding_dim = (
            embedding_dim
        )

        self.num_heads = (
            num_heads
        )

        self.head_dim = (
            embedding_dim
            // num_heads
        )

        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.output = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

        self.feed_forward = nn.Sequential(

            nn.Linear(
                embedding_dim,
                embedding_dim * 4
            ),

            nn.GELU(),

            nn.Linear(
                embedding_dim * 4,
                embedding_dim
            )

        )

    def forward(
        self,
        x
    ):

        # --------------------------------------------------
        # Attention branch
        # --------------------------------------------------

        residual = x

        normalized = self.norm1(
            x
        )

        batch_size, sequence_length, _ = (
            normalized.shape
        )

        Q = self.query(
            normalized
        )

        K = self.key(
            normalized
        )

        V = self.value(
            normalized
        )

        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        scores = (
            Q
            @ K.transpose(
                -2,
                -1
            )
        )

        scores = (
            scores
            / math.sqrt(
                self.head_dim
            )
        )

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        context = (
            attention_weights
            @ V
        )

        context = context.transpose(
            1,
            2
        )

        context = context.contiguous()

        context = context.view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        attention_output = self.output(
            context
        )

        x = (
            residual
            + attention_output
        )

        # --------------------------------------------------
        # Feed-forward branch
        # --------------------------------------------------

        residual = x

        normalized = self.norm2(
            x
        )

        feed_forward_output = (
            self.feed_forward(
                normalized
            )
        )

        x = (
            residual
            + feed_forward_output
        )

        return (
            x,
            attention_weights
        )


class TinyLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size=20,
        context_length=5,
        embedding_dim=16,
        num_heads=4
    ):

        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

        self.transformer = TinyTransformerBlock(
            embedding_dim,
            num_heads
        )

        self.output_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(
        self,
        input_ids
    ):

        batch_size, sequence_length = (
            input_ids.shape
        )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        token_vectors = (
            self.token_embedding(
                input_ids
            )
        )

        position_vectors = (
            self.position_embedding(
                positions
            )
        )

        position_vectors = (
            position_vectors
            .unsqueeze(0)
        )

        x = (
            token_vectors
            + position_vectors
        )

        x, attention_weights = (
            self.transformer(x)
        )

        logits = self.output_head(
            x
        )

        return (
            logits,
            attention_weights
        )


def full_transformer_backprop_demo():

    print()
    print("=" * 60)
    print("10. FULL TRANSFORMER BACKPROP")
    print("=" * 60)

    torch.manual_seed(0)

    model = TinyLanguageModel()

    input_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4]
        ]
    )

    targets = torch.tensor(
        [
            [1, 2, 3, 4, 5]
        ]
    )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    logits, attention_weights = (
        model(
            input_ids
        )
    )

    # --------------------------------------------------
    # Cross entropy
    # --------------------------------------------------

    loss = F.cross_entropy(
        logits.view(
            -1,
            logits.size(-1)
        ),
        targets.view(
            -1
        )
    )

    # --------------------------------------------------
    # BACKPROPAGATION
    # --------------------------------------------------

    loss.backward()

    print()
    print(
        "Logits shape:"
    )

    print(
        logits.shape
    )

    print()

    print(
        "Attention shape:"
    )

    print(
        attention_weights.shape
    )

    print()

    print(
        "Loss:",
        f"{loss.item():.6f}"
    )

    print()
    print(
        "Gradient flow:"
    )

    print(
        "Output head:",
        f"{model.output_head.weight.grad.norm().item():.6f}"
    )

    print(
        "Q:",
        f"{model.transformer.query.weight.grad.norm().item():.6f}"
    )

    print(
        "K:",
        f"{model.transformer.key.weight.grad.norm().item():.6f}"
    )

    print(
        "V:",
        f"{model.transformer.value.weight.grad.norm().item():.6f}"
    )

    print(
        "Attention output:",
        f"{model.transformer.output.weight.grad.norm().item():.6f}"
    )

    print(
        "Feed-forward:",
        f"{model.transformer.feed_forward[0].weight.grad.norm().item():.6f}"
    )

    print(
        "Token embeddings:",
        f"{model.token_embedding.weight.grad.norm().item():.6f}"
    )

    print(
        "Position embeddings:",
        f"{model.position_embedding.weight.grad.norm().item():.6f}"
    )

    print()
    print(
        "[OK] Gradient successfully flowed from the loss"
    )

    print(
        "  through the output head"
    )

    print(
        "  through the MLP"
    )

    print(
        "  through LayerNorm"
    )

    print(
        "  through attention"
    )

    print(
        "  through Q/K/V"
    )

    print(
        "  through embeddings."
    )

# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":

    print()
    print("=" * 60)
    print("OUTREACHLM BACKPROPAGATION")
    print("=" * 60)

    gradient_descent_demo()

    manual_backprop_demo()

    linear_backprop_demo()

    mlp_backprop_demo()

    layer_norm_backprop_demo()

    softmax_backprop_demo()

    qkv_backprop_demo()

    attention_backprop_demo()

    embedding_backprop_demo()

    full_transformer_backprop_demo()

    print()
    print("=" * 60)
    print("BACKPROPAGATION COMPLETE")
    print("=" * 60)
