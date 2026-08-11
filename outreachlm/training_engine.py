import os
import torch
import torch.nn as nn

class TrainingEngine:
    def __init__(
        self,
        model,
        optimizer,
        device,
        check_point_path="outreachlm_checkpoint.pt"
    ):

        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.check_point_path = check_point_path

        self.training_step = 0
        self.history = {
            "training_loss": [],
            "validation_loss": [],
            "gradient_norm": []
        }

    # Training step
    def train_step(
        self,
        input_ids,
        targets
    ):
        self.model.train()

        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)

        # clear old gradients
        self.optimizer.zero_grad(set_to_none=True)
        # forward pass
        logits = self.model(input_ids)
        # calculate loss
        loss = self.calculate_loss(logits, targets)
        # backpropagation
        loss.backward()
        # measure gradients
        gradient_norm = self.calculate_gradient_norm()
        # update parameters
        self.optimizer.step()

        self.training_step += 1

        self.history[
            "training_loss"
        ].append(
            loss.item()
        )

        self.history[
            "gradient_norm"
        ].append(
            gradient_norm
        )

        return {
            "loss": loss.item(),
            "gradient_norm": gradient_norm
        }
# Loss
    def calculate_loss(self, logits, targets):
        batch_size = logits.shape[0]
        sequence_length = logits.shape[1]
        vocab_size = logits.shape[2]

        logits = logits.reshape(batch_size * sequence_length, vocab_size)
        targets = targets.reshape(batch_size * sequence_length)

        loss = nn. functional.cross_entropy(logits, targets)

        return loss

#gradient norm
    def calculate_gradient_norm(self):
        total_norm = 0.0
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            param_norm = (parameter.grad.detach().data.norm(2))
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

# validation
    @torch.no_grad()
    def validate(self, input_ids, targets):
        self.model.eval()
        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)
        logits = self.model(input_ids)
        loss = self.calculate_loss(logits, targets)
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == targets).float()
        accuracy = correct.mean().item()
        validation_loss = loss.item()
        self.history["validation_loss"].append(validation_loss)

        return {
            "loss": validation_loss,
            "accuracy": accuracy
        }

# checkpoint
    def save_checkpoint(self):
        checkpoint = {
            "training_step": self.training_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history
        }
        torch.save(
            checkpoint,
            self.checkpoint_path
        )

#load checkpoint
    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint["training_step"]
        self.history = checkpoint["history"]

        return checkpoint


