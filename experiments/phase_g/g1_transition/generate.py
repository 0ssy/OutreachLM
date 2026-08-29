from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tokenizer import StupidTokenizer
from transition_model import TransitionModel


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate with G1 transition model.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("steps must be > 0.")

    tokenizer = StupidTokenizer.load(args.artifact_dir / "tokenizer.json")
    model = TransitionModel.load(args.artifact_dir / "transition_model.npz")
    rng = np.random.default_rng(args.seed)

    token_ids = tokenizer.encode(args.prompt, add_bos=False, add_eos=False)
    if not token_ids:
        token_ids = [tokenizer.bos_id]

    for _ in range(args.steps):
        current_token = token_ids[-1]
        distribution = model.predict_next_distribution(current_token)
        if args.sample:
            next_token = int(rng.choice(np.arange(model.vocab_size), p=distribution))
        else:
            next_token = int(np.argmax(distribution))
        token_ids.append(next_token)
        if next_token == tokenizer.eos_id:
            break

    print(f"Prompt:   {args.prompt}")
    print(f"Token ids:{token_ids}")
    print(f"Decoded:  {tokenizer.decode(token_ids, skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
