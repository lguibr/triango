package core

import "testing"

func TestStateTransition(t *testing.T) {
	// Let's create a state with a single triangle and a double triangle.
	// StandardPieces[0] is single, StandardPieces[1] is double.
	s := NewGameState([3]int8{0, 1, -1})

	// Valid Move 1: Place a single triangle at index 0 (Top-left UP triangle)
	s2, valid := s.ApplyMove(0, 0)
	if !valid {
		t.Fatalf("Failed to place single triangle at origin")
	}

	if s2.Score != 1 {
		t.Errorf("Expected score 1, got %d", s2.Score)
	}

	if s2.Board.PopCount() != 1 {
		t.Errorf("Expected board to have 1 bit set, has: %d", s2.Board.PopCount())
	}

	if s2.AvailablePieces[0] != -1 {
		t.Errorf("Expected piece 0 to be consumed")
	}
	if s2.PiecesLeft != 1 {
		t.Errorf("Expected 1 piece left (the double)")
	}

	// Valid Move 2: Place Double Triangle.
	// As defined in pieces.go, StandardPiece[1] is a Down triangle.
	// Index 1 is DOWN. Let's place it at index 1 (Row 0, Col 1).
	if IsUp(0, 1) {
		t.Fatalf("Index 1 is UP, but piece 1 requires DOWN")
	}
	s3, valid2 := s2.ApplyMove(1, 1)
	if !valid2 {
		t.Fatalf("Failed to place double triangle at index 1")
	}

	if s3.Score != 2 { // 1 existing + 1 for the Single Down
		t.Errorf("Expected score 2, got %d", s3.Score)
	}

	if s3.Board.PopCount() != 2 {
		t.Errorf("Expected board to have 2 bits set, has: %d", s3.Board.PopCount())
	}

	// Since PiecesLeft == 0 now, what is terminal state?
	// It's not terminal. The controller must refill the tray.
	if s3.IsTerminal {
		t.Errorf("Expected game not to be terminal on empty tray")
	}

	// Move 3: Try to place over existing pieces!
	// E.g. placing single at 0 again on s3 to test failure.
	s3bad := s3
	s3bad.AvailablePieces[0] = 0 // Simulate having a single again
	s3bad.PiecesLeft = 1
	_, validOverlap := s3bad.ApplyMove(0, 0)
	if validOverlap {
		t.Errorf("Move should be rejected due to overlap")
	}
}

func BenchmarkStateCopy(b *testing.B) {
	s := NewGameState([3]int8{0, 0, 0})

	// A benchmark to measure immutable applying
	// Using the single triangle piece.
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Just simulate cloning and modifying the Bitboard
		next := s
		next.Board.SetBit(10)
		next.Score++
	}
}
