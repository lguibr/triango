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
