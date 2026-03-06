import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class AlphaZeroNet(nn.Module):
    """
    SOTA Dual-Headed ResNet for Triango MCTS.
    Input: [Batch, Channels, 96]
    Outputs:
      - Policy (P): [Batch, 96] (Probability distribution of best moves)
      - Value (V):  [Batch, 1]  (Expected final score)
    """
    def __init__(self, input_channels=4, num_res_blocks=5, num_filters=64):
        super().__init__()
        # 1. Initial Convolutional Block
        self.conv_in = nn.Conv1d(input_channels, num_filters, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm1d(num_filters)

        # 2. Residual Tower
        self.res_blocks = nn.ModuleList([
            ResBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # 3. Policy Head (Predicts where to place the pieces)
        self.policy_conv = nn.Conv1d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm1d(2)
        # 96 triangles * 2 filters = 192 features
        self.policy_fc = nn.Linear(96 * 2, 96) 

        # 4. Value Head (Predicts the final integer score)
        self.value_conv = nn.Conv1d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm1d(1)
        self.value_fc1 = nn.Linear(96, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x is expected to be [B, C, 96]
        x = F.relu(self.bn_in(self.conv_in(x)))
        
        for block in self.res_blocks:
            x = block(x)

        # --- Policy Head ---
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)  # Flatten
        p = self.policy_fc(p)
        # Using LogSoftmax for stable training (KL-Divergence / CrossEntropy)
        p = F.log_softmax(p, dim=1)

        # --- Value Head ---
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # Flatten
        v = F.relu(self.value_fc1(v))
        # The value is a raw score prediction, so no tanh/sigmoid clipping.
        # It literally predicts the expected Score scalar directly.
        v = self.value_fc2(v)

        return p, v

if __name__ == "__main__":
    # Test the architecture
    model = AlphaZeroNet()
    # Mock Batch: 2 games, 4 input channels (Board, Tray1, Tray2, Tray3), 96 geometric cells
    mock_input = torch.randn(2, 4, 96)
    policy, value = model(mock_input)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Input Shape:  {mock_input.shape}")
    print(f"Policy Shape: {policy.shape} (Expected: [2, 96])")
    print(f"Value Shape:  {value.shape}  (Expected: [2, 1])")
