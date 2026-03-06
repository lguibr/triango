package core

// GameState represents the immutable SOTA state for MCTS parallelization.
// It is specifically designed to be < 48 bytes to be passed efficiently by value.
type GameState struct {
	Board           Bitboard
	Score           int
	AvailablePieces [3]int8 // Stores Piece IDs. -1 implies slot is empty.
	PiecesLeft      int8    // 0 implies tray needs a refresh.
	IsTerminal      bool
}

// NewGameState initializes a fresh Tricrack board.
func NewGameState(initialPieces [3]int8) GameState {
	var left int8
	for _, p := range initialPieces {
		if p != -1 {
			left++
		}
	}
	return GameState{
		Board:           EmptyBitboard,
		Score:           0,
		AvailablePieces: initialPieces,
		PiecesLeft:      left,
		IsTerminal:      false,
	}
}

// ApplyMove creates a new GameState entirely immutably by placing a piece.
// Arguments:
// - pieceSlot: 0, 1, or 2 (which tray slot to use)
// - boardIndex: 0-95 (where to place the origin of the piece)
// Returns the next state and a boolean indicating if the move was valid.
func (s GameState) ApplyMove(pieceSlot int, boardIndex int) (GameState, bool) {
	if pieceSlot < 0 || pieceSlot > 2 {
		return s, false
	}

	pieceID := s.AvailablePieces[pieceSlot]
	if pieceID == -1 {
		return s, false // Slot already used
	}

	// In a real implementation this lookup is safe and instant.
	// We assume StandardPieces holds the piece library.
	piece := StandardPieces[pieceID]
	mask := piece.Masks[boardIndex]

	// 1. Check validity (bounds and overlaps)
	if mask.Equals(EmptyBitboard) {
		return s, false // Out of bounds or invalid polarity placement
	}
	if s.Board.Intersects(mask) {
		return s, false // Overlaps with existing pieces
	}

	// 2. Clone state purely by value (Zero allocation SOTA)
	next := s
	next.AvailablePieces[pieceSlot] = -1
	next.PiecesLeft--

	// 3. Place piece and score points for placement (1 point per triangle)
	next.Board = next.Board.Or(mask)
	next.Score += mask.PopCount()

	// 4. Check for line clears across all 24 axes
	var clearedMask Bitboard
	linesCleared := 0

	for _, lineMask := range AllMasks {
		if next.Board.Contains(lineMask) {
			clearedMask = clearedMask.Or(lineMask)
			linesCleared++
		}
	}

	if linesCleared > 0 {
		// Clear the triangles forming the full lines
		next.Board = next.Board.AndNot(clearedMask)

		// Bonus points for line clears. 2 Points per triangle that was collapsed!
		next.Score += clearedMask.PopCount() * 2
	}

	// 5. Terminal Check (can any of the remaining pieces be legally placed?)
	// In MCTS, if PiecesLeft == 0, the tray refreshes natively as an "environment" step
	// before the agent's next turn. So if PiecesLeft > 0, we check terminal status.
	if next.PiecesLeft > 0 {
		hasValidMove := false
		for slot := 0; slot < 3; slot++ {
			pid := next.AvailablePieces[slot]
			if pid != -1 {
				p := StandardPieces[pid]
				for idx := 0; idx < TotalTriangles; idx++ {
					m := p.Masks[idx]
					if !m.Equals(EmptyBitboard) && !next.Board.Intersects(m) {
						hasValidMove = true
						break
					}
				}
				if hasValidMove {
					break
				}
			}
		}
		if !hasValidMove {
			next.IsTerminal = true
		}
	}

	return next, true
}

// RefillTray creates a new state from the current one, replacing the available pieces.
// This is used by the environment or MCTS rollouts when PiecesLeft drops to zero.
func (s GameState) RefillTray(newPieces [3]int8) GameState {
	var left int8
	for _, p := range newPieces {
		if p != -1 {
			left++
		}
	}
	next := s
	next.AvailablePieces = newPieces
	next.PiecesLeft = left

	// We must also evaluate if the new pieces immediately result in a terminal state
	if next.PiecesLeft > 0 {
		hasValidMove := false
		for slot := 0; slot < 3; slot++ {
			pid := next.AvailablePieces[slot]
			if pid != -1 {
				p := StandardPieces[pid]
				for idx := 0; idx < TotalTriangles; idx++ {
					m := p.Masks[idx]
					if !m.Equals(EmptyBitboard) && !next.Board.Intersects(m) {
						hasValidMove = true
						break
					}
				}
				if hasValidMove {
					break
				}
			}
		}
		next.IsTerminal = !hasValidMove
	}
	return next
}
