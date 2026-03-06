# Triango Core Pipeline (`/core`)

The `core` module forms the bedrock of the Tricrack mathematical engine. It manages all zero-allocation state management and foundational geometries.

## Architecture Map

- **`bitboard.go`:** Implements the core 128-bit `[2]uint64` structure, enabling blazingly fast `SetBit`, `HasBit`, `Equals`, `And`, `AndNot`, and popcount integrations over the 96-cell geometry.
- **`grid.go`:** Calculates `FlatIndex` 1D mapping. Most notably, it resolves the exact mathematical geometry mapping of the **24 absolute line-clearing masks** (Horizontal, Left-Up Diagonal, Right-Up Diagonal) caching them at system boot for instant $O(1)$ intersections.
- **`pieces.go`:** Takes relative axial 3D-space definitions `Coord{X,Y,Z}` and compiles them securely into complete 96-array absolute Bitboards. It correctly evaluates triangle-polarity constraints and prevents out-of-bounds spatial overflow.
- **`print.go`:** Maps compressed bitboards back into an authentic visual hexagonal triangular array for CLI debugging and inspection via interlocking Up (`▲`, `△`) and Down (`▼`, `▽`) triangles. It also encapsulates the visualizer for rendering piece trays.
- **`state.go`:** The heart of Triango—managing the mutable-free `GameState`. Contains all transitions, processing absolute masks, executing cascading line clears, and automatically computing overlapping Score chain-combos, validating the game logic precisely without garbage collection overhead.

## Performance Philosophy
While a visual Tricrack game features variable geometry, the mathematically defined internal playable triangles span precisely 96 vertices. Caching this inside standard array sizes allows Go to perfectly layout structures ensuring cache-line alignment and extreme memory locality.

## Scoring Rules

Triango relies on standard tile-placement score maximizing formulas identical to 1010! mechanics:
- **Placement Points:** Every time a piece is legally placed, you are instantly awarded `1 Point` per triangle occupying the piece configuration (e.g., placing the 6-triangle Hexagon shape instantly awards 6 points).
- **Line Completion Bonus:** If your placement causes any of the 24 internal absolute axes to form a fully unbroken line (Horizontal, Red Diagonal Up-Right, or Black Diagonal Up-Left), the line completely collapses.
  - You are awarded **`2 Points`** for every single triangle erased inside the collapsing geometric line masks.

## AlphaZero Dataset Architecture

Triango is architected to be the backbone of a deep Reforcement Learning network. 
The core state emits dense, non-lossy `[96]int8` binary arrays representing exact positional data along with perfectly tracked legal geometric sub-state. This exact schema guarantees PyTorch Tensors natively digest 2D features mapping $V$ (Score) and $P$ (MCTS Visit Volume) sequentially.
