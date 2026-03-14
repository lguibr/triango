
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*")


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
    - Outputs `(Value, Policy)`.
    """

    def __init__(self, d_model: int = 256, nhead: int = 16, num_layers: int = 12):
        super().__init__()

        # Input features per triangle: 7 (1 board + 3x piece overlays + 3x valid masks)
        # Deep Feature Embedding (Juice)
        self.input_proj = nn.Sequential(
            nn.Linear(7, 64), nn.Mish(), nn.LayerNorm(64), nn.Linear(64, d_model)
        )

        # Positional Encoding to teach the Transformer the exact absolute ID of each triangle (0 to 95)
        # We replace blind 1D embedding with mathematically rigid 3D hexagonal positions.
        self.register_buffer("pos_emb", self._build_geometric_embeddings(d_model))

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

        # Policy Head: Predicts action probabilities directly to speed up MCTS.
        # Outputs [Batch, 3, 50] representing probabilities across 3 tray slots & 50 discrete orientations
        self.policy_fc1 = nn.Linear(d_model, 128)
        self.policy_norm = nn.LayerNorm(128)
        self.policy_fc2 = nn.Linear(128, 3 * 50)

    def _build_geometric_embeddings(self, d_model: int) -> torch.Tensor:
        """
        Builds a frozen sinusoidal positional encoding tensor of shape [1, 96, d_model].
        Maps the exact 3D Hexagonal (x, y, z) coordinates into the latent space.
        """
        import math

        from triango.env.coords import INDEX_TO_COORD

        pe = torch.zeros(96, d_model)
        div_term = torch.exp(torch.arange(0, d_model, 6, dtype=torch.float) * (-math.log(10000.0) / d_model))

        for i in range(96):
            if i in INDEX_TO_COORD:
                x, y, z = INDEX_TO_COORD[i]
            else:
                x, y, z = 0, 0, 0
            
            # Pack x, y, z into the d_model dimension using interleaved Sine/Cosine
            # We stride by 6 because there are 3 variables (x,y,z) each producing a sin/cos pair = 6 dims
            for j in range(len(div_term)):
                if j * 6 < d_model:
                    pe[i, j * 6 + 0] = math.sin(x * div_term[j])
                    pe[i, j * 6 + 1] = math.cos(x * div_term[j])
                if j * 6 + 2 < d_model:
                    pe[i, j * 6 + 2] = math.sin(y * div_term[j])
                    pe[i, j * 6 + 3] = math.cos(y * div_term[j])
                if j * 6 + 4 < d_model:
                    pe[i, j * 6 + 4] = math.sin(z * div_term[j])
                    pe[i, j * 6 + 5] = math.cos(z * div_term[j])

        return pe.unsqueeze(0)  # [1, 96, d_model]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: [Batch, 7, 96]
        # Transformers require sequence-centric shapes: [Batch, SequenceLength=96, EmbedSize=7]
        x = x.transpose(1, 2).contiguous()

        batch_size = x.size(0)

        # 1. Project raw features to the deeper d_model space
        x = self.input_proj(x)

        # 2. Add spatial positional embeddings (0..95 offsets)
        # Using registered buffer pos_emb which matches device via standard broadcast
        x = x + self.pos_emb

        # 3. Deep Hexagonal Self-Attention
        attn_out = self.transformer(x)

        # 4. Next-State Evaluation Heads (Mean pool across all 96 spatial tokens logically)
        v_pooled = attn_out.mean(dim=1)

        # Mish activation and LayerNorm for value extraction stabilization
        v = F.mish(self.value_norm(self.value_fc1(v_pooled)))

        # Raw Value Expectation
        value = self.value_fc2(v)

        # Policy computation (predicts prior directly to narrow MCTS without full rollouts!)
        p = F.mish(self.policy_norm(self.policy_fc1(v_pooled)))
        policy_logits = self.policy_fc2(p).view(batch_size, 3, 50)
        policy_probs = F.softmax(policy_logits.view(batch_size, -1), dim=-1).view(batch_size, 3, 50)

        return value, policy_probs
