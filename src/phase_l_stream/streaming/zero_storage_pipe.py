from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from src.phase_l_stream.optimization.cosine_scheduler import CosineWarmupScheduler


class ZeroStorageStreamingPipeline:
    """Single-pass streaming trainer: ingest, update, discard.

    Raw text never reaches disk and tensors are released as soon as the
    optimizer step completes, so RAM stays flat regardless of stream length.
    """

    @staticmethod
    def execute_single_pass_run(
        model: nn.Module,
        datastream_generator,
        config: dict,
        *,
        max_batches: int | None = None,
    ) -> dict:
        model.train()
        schedule = config["optimization_schedule"]
        optimizer = optim.AdamW(
            model.parameters(), lr=schedule["lr_max"], weight_decay=0.1
        )
        loss_criterion = nn.CrossEntropyLoss()
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_tokens=schedule["warmup_tokens"],
            total_tokens=schedule["total_target_tokens"],
            lr_max=schedule["lr_max"],
            lr_min=schedule["lr_min"],
        )

        losses: list[float] = []
        learning_rates: list[float] = []
        tokens_seen = 0

        for index, (input_ids, target_ids) in enumerate(datastream_generator):
            if max_batches is not None and index >= max_batches:
                break

            inputs = (
                input_ids
                if isinstance(input_ids, torch.Tensor)
                else torch.tensor(input_ids, dtype=torch.long)
            )
            targets = (
                target_ids
                if isinstance(target_ids, torch.Tensor)
                else torch.tensor(target_ids, dtype=torch.long)
            )
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
                targets = targets.unsqueeze(0)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            loss = loss_criterion(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=schedule["gradient_clip_norm"]
            )
            optimizer.step()

            batch_tokens = int(inputs.numel())
            tokens_seen += batch_tokens
            learning_rates.append(scheduler.step_tokens(batch_tokens))
            losses.append(float(loss.item()))

            # Zero-storage enforcement: drop references immediately.
            del inputs, targets, logits, loss

        return {
            "batches": len(losses),
            "tokens_seen": tokens_seen,
            "losses": losses,
            "learning_rates": learning_rates,
            "final_loss": losses[-1] if losses else None,
        }
