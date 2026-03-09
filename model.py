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
    def __init__(self, d_model=256, nhead=16, num_layers=12):
        super(AlphaZeroNet, self).__init__()

        # Input features per triangle: 7 (1 board + 3x piece overlays + 3x valid masks)
        self.input_proj = nn.Linear(7, d_model)
        
        # Positional Encoding to teach the Transformer the exact absolute ID of each triangle (0 to 95)
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
        
        # Value Head: Predicts V(s) -> scalar mathematical expectation [0, 300+]
        self.value_fc1 = nn.Linear(d_model, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
        # Auxiliary Line Clear Head: Predicts probability of creating a line clear [0.0, 1.0]
        self.line_clear_fc = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: [Batch, 4, 96]
        # Transformers require sequence-centric shapes: [Batch, SequenceLength=96, EmbedSize=4]
        x = x.transpose(1, 2).contiguous()
        
        batch_size = x.size(0)
        
        # 1. Project 4 raw features to the deeper d_model space
        x = self.input_proj(x)
        
        # 2. Add spatial positional embeddings (0..95 offsets)
        positions = torch.arange(96, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_emb(positions)
        
        # 3. Deep Hexagonal Self-Attention
        attn_out = self.transformer(x)
        
        # 4. Next-State Evaluation Heads (Mean pool across all 96 spatial tokens logically)
        # [Batch, 96, d_model] -> [Batch, d_model]
        v_pooled = attn_out.mean(dim=1)
        v = F.relu(self.value_fc1(v_pooled))
        
        # Raw Value Expectation
        value = self.value_fc2(v)
        
        # Probability of clearing lines
        # Using Sigmoid so BCE loss can cleanly evaluate [0.0, 1.0] target mappings
        line_clear_prob = torch.sigmoid(self.line_clear_fc(v))
        
        return value, line_clear_prob
