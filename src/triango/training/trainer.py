from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from triango.model.network import AlphaZeroNet
from triango.training.buffer import ReplayBuffer


def train(
    model: AlphaZeroNet,
    buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    hw_config: dict[str, Any],
) -> None:
    model.train()

    epochs = hw_config["train_epochs"]
    device = hw_config["device"]

    dataloader = DataLoader(buffer, batch_size=hw_config["train_batch_size"], shuffle=True)

    bce_loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        value_loss_sum = 0.0
        line_clear_loss_sum = 0.0
        policy_loss_sum = 0.0

        for states, targets_value, targets_line, targets_policy in dataloader:
            states = states.to(device)
            targets_value = targets_value.to(device) / 100.0
            targets_line = targets_line.to(device)
            # targets_policy: [Batch, 3, 50]
            targets_policy = targets_policy.to(device)

            optimizer.zero_grad()

            pred_values, pred_line_clears, pred_policy = model(states)

            value_loss = F.mse_loss(pred_values, targets_value)
            line_clear_loss = bce_loss_fn(pred_line_clears, targets_line)
            
            # Policy loss: Cross Entropy between Ground Truth MCTS distributions and Network Logits.
            # pred_policy outputs Softmax Probs already, so we use manual KL-Divergence / Cross Entropy
            # add epsilon for numerical stability
            log_preds = torch.log(pred_policy + 1e-8)
            policy_loss = -torch.sum(targets_policy * log_preds) / states.size(0)

            loss = value_loss + (line_clear_loss * 0.5) + (policy_loss * 2.0)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            value_loss_sum += value_loss.item()
            line_clear_loss_sum += line_clear_loss.item()
            policy_loss_sum += policy_loss.item()

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Total: {total_loss:.4f} (MSE: {value_loss_sum:.4f}, Line BCE: {line_clear_loss_sum:.4f}, Policy CE: {policy_loss_sum:.4f})"
        )
