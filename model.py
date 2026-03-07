import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroNet(nn.Module):
    """
    SOTA Deep Self-Attention Transformer Architecture for Triango.
    Instead of 1D convolutions that assume linear spatial geometry, this 
    treats the 96 triangles as discrete tokens, natively learning their 
    hexagonal adjacency through pure Attention.
    """
    def __init__(self, d_model=256, nhead=16, num_layers=12):
        super(AlphaZeroNet, self).__init__()

        # Input features per triangle: 16 (4 frames * 4 feature channels)
        self.input_proj = nn.Linear(16, d_model)
        
        # Positional Encoding to teach the Transformer the exact absolute ID of each triangle (0 to 95)
        # We use a learned embedding since the board size is strictly fixed to exactly 96.
        self.pos_emb = nn.Embedding(96, d_model)
        
        # SOTA Self-Attention Core
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Policy Head: Predicts P(a|s) -> probability distribution over the 96 triangles
        # We project each of the 96 tokens down to exactly 1 logit score
        self.policy_head = nn.Linear(d_model, 1)
        
        # Value Head: Predicts V(s) -> scalar mathematical expectation [-1.0, 1.0]
        # We aggregate the transformer output over the 96 tokens via mean pooling natively
        self.value_fc1 = nn.Linear(d_model, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: [Batch, 16, 96]
        # Transformers require sequence-centric shapes: [Batch, SequenceLength=96, EmbedSize=16]
        x = x.transpose(1, 2).contiguous()
        
        batch_size = x.size(0)
        
        # 1. Project 16 raw features to the deeper d_model space
        x = self.input_proj(x)
        
        # 2. Add spatial positional embeddings (0..95 offsets)
        positions = torch.arange(96, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_emb(positions)
        
        # 3. Deep Hexagonal Self-Attention
        attn_out = self.transformer(x)
        
        # 4. Policy Head: Apply linear transformation to each token yielding 1 logit per cell
        # Output shape of policy_head: [Batch, 96, 1]. Squeeze collapses it to [Batch, 96]
        p = self.policy_head(attn_out).squeeze(-1) 
        policy_logits = F.log_softmax(p, dim=1)
        
        # 5. Value Head: Mean pool across all 96 spatial tokens logically
        # [Batch, 96, d_model] -> [Batch, d_model]
        v_pooled = attn_out.mean(dim=1)
        v = F.relu(self.value_fc1(v_pooled))
        
        # We REMOVE torch.tanh() here!
        # Triango scores are unbounded positive integers (0 to 300+).
        # Tanh bounds the output strictly to [-1.0, 1.0], meaning the MSELoss 
        # trying to pull the model to predict 150.0 was hopelessly saturating at 1.0.
        value = self.value_fc2(v)
        
        return policy_logits, value
