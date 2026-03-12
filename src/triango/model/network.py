
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaZeroNet(nn.Module):
    """
    SOTA Deep Self-Attention Transformer Architecture for Triango.
    Instead of 1D convolutions that assume linear spatial geometry, this
    treats the 96 triangles as discrete tokens, natively learning their
    hexagonal adjacency through pure Attention.

    UPGRADE (March 2026): Next-State Evaluation
    - The model no longer outputs a 96-discrete probability map.
    - Instead it evaluates an "After-State" (board after piece is placed).
    - It explicitly "sees" the shapes via Spatial Overlays (Channels 1-3).
    - Outputs `(Value, Line_Clear_Probability)`.
    """

    def __init__(self, d_model: int = 256, nhead: int = 16, num_layers: int = 12):
        super().__init__()

        # Input features per triangle: 7 (1 board + 3x piece overlays + 3x valid masks)
        # Deep Feature Embedding (Juice)
        self.input_proj = nn.Sequential(
            nn.Linear(7, 64), nn.Mish(), nn.LayerNorm(64), nn.Linear(64, d_model)
        )

        # Positional Encoding to teach the Transformer the exact absolute ID of each triangle (0 to 95)
        self.pos_emb = nn.Embedding(96, d_model)

        # SOTA Self-Attention Core - PRE-LN Architecture for deeper gradient stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,  # Pre-LN for faster convergence
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Value Head: Predicts V(s) -> scalar mathematical expectation [0, 300+]
        self.value_fc1 = nn.Linear(d_model, 64)
        self.value_norm = nn.LayerNorm(64)
        self.value_fc2 = nn.Linear(64, 1)

        # Auxiliary Line Clear Head: Predicts probability of creating a line clear [0.0, 1.0]
        self.line_clear_fc = nn.Linear(64, 1)

        # Policy Head: Predicts action probabilities directly to speed up MCTS.
        # Outputs [Batch, 3, 50] representing probabilities across 3 tray slots & 50 discrete orientations
        self.policy_fc1 = nn.Linear(d_model, 128)
        self.policy_norm = nn.LayerNorm(128)
        self.policy_fc2 = nn.Linear(128, 3 * 50)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: [Batch, 7, 96]
        # Transformers require sequence-centric shapes: [Batch, SequenceLength=96, EmbedSize=7]
        x = x.transpose(1, 2).contiguous()

        batch_size = x.size(0)

        # 1. Project raw features to the deeper d_model space
        x = self.input_proj(x)

        # 2. Add spatial positional embeddings (0..95 offsets)
        positions = torch.arange(96, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_emb(positions)

        # 3. Deep Hexagonal Self-Attention
        attn_out = self.transformer(x)

        # 4. Next-State Evaluation Heads (Mean pool across all 96 spatial tokens logically)
        v_pooled = attn_out.mean(dim=1)

        # Mish activation and LayerNorm for value extraction stabilization
        v = F.mish(self.value_norm(self.value_fc1(v_pooled)))

        # Raw Value Expectation
        value = self.value_fc2(v)

        # Probability of clearing lines
        line_clear_prob = torch.sigmoid(self.line_clear_fc(v))

        # Policy computation (predicts prior directly to narrow MCTS without full rollouts!)
        p = F.mish(self.policy_norm(self.policy_fc1(v_pooled)))
        policy_logits = self.policy_fc2(p).view(batch_size, 3, 50)
        policy_probs = F.softmax(policy_logits.view(batch_size, -1), dim=-1).view(batch_size, 3, 50)

        return value, line_clear_prob, policy_probs
