import argparse
import json
from pathlib import Path

import torch

from outreachlm.tokenizer import CharacterTokenizer
from outreachlm.v4_model import OutreachV4Model


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def tokenizer_from_config(config):
    tokenizer = CharacterTokenizer.__new__(CharacterTokenizer)
    tokenizer.pad_token = config["pad_token"]
    tokenizer.unk_token = config["unk_token"]
    tokenizer.tokens = config["tokens"]
    tokenizer.token_to_id = {
        token: index
        for index, token in enumerate(tokenizer.tokens)
    }
    tokenizer.id_to_token = {
        index: token
        for token, index in tokenizer.token_to_id.items()
    }
    return tokenizer


def load_tokenizer(path, artifact):
    if path is not None and path.exists():
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if (
            "tokens" in data
            and "pad_token" in data
            and "unk_token" in data
        ):
            return tokenizer_from_config(data)

    artifact_tokenizer = artifact.get("tokenizer_config")
    if artifact_tokenizer is None:
        raise RuntimeError("Tokenizer not found in path or model artifact.")
    return tokenizer_from_config(artifact_tokenizer)


def load_model_and_tokenizer(model_path, tokenizer_path):
    artifact = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
    )
    model_config = artifact["model_config"]
    model = OutreachV4Model(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        embedding_dim=model_config["embedding_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        ffn_dim=model_config.get("ffn_dim", 684),
    ).to(DEVICE)
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()

    tokenizer = load_tokenizer(tokenizer_path, artifact)
    return model, tokenizer


def sample_next_token(logits, temperature, top_k):
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, k=min(top_k, logits.shape[-1]))
        min_keep = values[..., -1, None]
        logits = torch.where(
            logits < min_keep,
            torch.full_like(logits, -1e9),
            logits
        )
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_k):
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.token_to_id[tokenizer.unk_token]]

    generated = list(token_ids)
    for _ in range(max_new_tokens):
        context = generated[-model.context_length :]
        input_ids = torch.tensor(
            [context],
            dtype=torch.long,
            device=DEVICE,
        )
        with torch.no_grad():
            logits = model(input_ids)[0, -1, :]
        next_token_id = sample_next_token(
            logits=logits,
            temperature=temperature,
            top_k=top_k,
        )
        generated.append(next_token_id)
    return tokenizer.decode(generated)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text with OutreachLM V4."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("experiments") / "v4-training" / "v4-final.pt",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("experiments") / "v4-training" / "tokenizer.json",
    )
    parser.add_argument("--prompt", type=str, default="OutreachLM is")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
    )
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(generated)


if __name__ == "__main__":
    main()
