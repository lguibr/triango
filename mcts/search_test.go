package mcts

import (
	"fmt"
	"testing"
	"triango/core"
)

func TestParallelSearch(t *testing.T) {
	// A basic test to see if MCTS returns a valid move.
	// We give it 3 singles, so valid moves exist.
	s := core.NewGameState([3]int8{0, 0, 0})

	cfg := SearchConfig{
		Simulations: 1000,
		Threads:     4,
	}

	bestMove := ParallelSearch(s, cfg)

	if bestMove.PieceSlot < 0 || bestMove.PieceSlot > 2 {
		t.Errorf("Invalid piece slot returned: %d", bestMove.PieceSlot)
	}
	if bestMove.BoardIndex < 0 || bestMove.BoardIndex >= core.TotalTriangles {
		t.Errorf("Invalid board index returned: %d", bestMove.BoardIndex)
	}

	fmt.Printf("MCTS Best Move: Slot %d, Index %d\n", bestMove.PieceSlot, bestMove.BoardIndex)
}

func BenchmarkMCTS_SingleThread(b *testing.B) {
	s := core.NewGameState([3]int8{0, 1, 0})
	cfg := SearchConfig{
		Simulations: 1000,
		Threads:     1,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ParallelSearch(s, cfg)
	}
}

func BenchmarkMCTS_QuadThread(b *testing.B) {
	s := core.NewGameState([3]int8{0, 1, 0})
	cfg := SearchConfig{
		Simulations: 1000,
		Threads:     4,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ParallelSearch(s, cfg)
	}
}
