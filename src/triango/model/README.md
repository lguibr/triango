# Transformer Neural Network (`src/triango/model`)

This directory contains the brain of the Triango AI. It translates raw positional data arrays into intelligent, tactical heuristic values to inform MCTS generation.

## AlphaZeroNet Architecture

The core model, `AlphaZeroNet` in `network.py`, is built entirely upon PyTorch. Rather than using legacy Convolutional layers (CNNs) historically used for grid maps, Triango leverages a powerful **Transformer Encoder** architecture capable of capturing deep, long-range dependencies across the fragmented 96-tile triangular grid.

### 1. The Input Stem
MCTS `extract_feature()` logic provides the network with a Boolean geometric map structured as `[Batch, 7, 96]`.
1. **Linear Projection**: The raw 7 channels (Occupation, Orientation, Shape 1-5 availability) are immediately projected into a wider `$d\_model$` embedding space.
2. **Positional Encoding**: Because Transformers have no inherent concept of spatial locality, we dynamically generate a `[96, $d\_model$]` learnable Positional Embedding matrix. This allows the AI to differentiate between corner-tiles, edges, and center grid properties.

### 2. The Transformer Backbone
The processed embeddings pass through $N$ custom `nn.TransformerEncoderLayer` modules.
- We utilize `norm_first=True` (Pre-Layer Normalization), which significantly stabilizes gradient propagation during the deeply volatile early stages of self-play reinforcement learning.
- Each layer applies **Multi-Headed Self-Attention**. This allows the network to logically deduce exactly how a piece dropped in the top-right corner might fundamentally block an imminent line-clear opportunity cascading from the bottom-left map geometry.

### 3. AlphaZero Output Heads
Most AlphaZero variants possess two predictive heads. The Triango Neural network utilizes **three** fully distinct readout layers derived from the `[CLS]` pooled summary vector.

1. **The Value Head (Q-value)**:
   - A standard Multi-Layer Perceptron (MLP) finishing with `Tanh()`.
   - Constrained strictly between `[-1.0, 1.0]`.
   - **Responsibility**: Predicts the absolute mathematical likelihood of winning (or earning extreme points) from the current board state. Used by MCTS to "exploit" golden paths.
2. **The Line Clear Head**:
   - Matches a `Sigmoid()` threshold boundary `[0.0, 1.0]`.
   - **Responsibility**: Predicts the boolean probability of the current player completing a contiguous horizontal/diagonal line on this exact turn. This heavily accelerates localized heuristic learning.
3. **The Learned Policy Head ($P(s,a)$)**:
   - Expands out to a massive `[3, 50]` tensor probability distribution flattened via `Softmax`.
   - **Responsibility**: Predicts exactly *which* geometric slot (1 of 3 available shapes) and *which* board orientation index (0-49 translations) the MCTS tree should search first. This acts as the algorithmic "prior", drastically reducing search space blindness.
