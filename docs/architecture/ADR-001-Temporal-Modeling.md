# ADR-001: AlphaZero Temporal & Sequence Modeling

## Status
Proposed

## Context
Currently, the Triango AlphaZero model evaluates board states in pure isolation (Markov Property). It only looks at the absolute geometrical placement of pieces at time $t$. 
However, game theory often benefits from understanding *how* a board state was reached (the timeline of movements), which reveals opponent strategy, traps, and tactical momentum.

We need to decide on an architectural upgrade to allow the Neural Network to process historical timesteps and sequence movements effectively, balancing computational speed with theoretical MCTS accuracy.

## Decision
*(Pending User Feedback)*

### Option A: Frame Stacking (AlphaStar / Standard DeepMind)
Instead of feeding 1 board state (4 channels), we feed the last $N$ board states concatenated together (e.g., $N=4$ means $4 \times 4 = 16$ input channels).
- **Pros:** Extremely fast, requires zero changes to the ResNet architecture (just change the input `Conv1d` from 4 to 16 channels), proven to work elegantly in chess and Go.
- **Cons:** Hardcoded timeline horizon. Can only "see" $N$ turns into the past.

### Option B: Recurrent ResNet (LSTM / GRU)
Inject an LSTM cell between the ResNet feature extractor and the Policy/Value heads. The network keeps a rolling "hidden state" memory of the entire game from turn 1.
- **Pros:** Theoretically infinite memory horizon. Can learn long-term setups.
- **Cons:** Severely complicates MCTS. During the MCTS tree search, every node simulation must uniquely clone and step the LSTM hidden state forward, which geometrically destroys parallelization speed.

### Option C: Transformer Attention (Decision Transformer)
Replace the ResNet entirely with a Sequence Transformer. Every piece placement is a "token" in a sequence.
- **Pros:** Absolute State-of-the-Art for sequence modeling. Understands global context beautifully.
- **Cons:** Massive engineering rewrite of both the Go generator and the PyTorch server. Much slower inference speed than ResNet.

## Consequences
- **Option A** retains our >1,000 games/minute speed but provides limited history.
- **Option B** tanks MCTS simulation speed due to sequential hidden states.
- **Option C** sets a SOTA foundation but forces a multi-day rewrite of our pipeline infrastructure.
