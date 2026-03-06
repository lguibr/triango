package mcts

import (
	"fmt"
	"math/rand"
	"testing"
	"triango/core"
)

func TestParallelSearch(t *testing.T) {
	// A basic test to see if MCTS returns a valid move.
	// We give it 3 singles, so valid moves exist.
	rootState := core.NewGameState([3]int8{0, 0, 0})

	cfg := SearchConfig{
		Simulations: 100,
		Threads:     2,
	}

	bestMove, _ := ParallelSearch(rootState, cfg)

	if bestMove.PieceSlot < 0 || bestMove.PieceSlot > 2 {
		t.Errorf("Invalid piece slot returned: %d", bestMove.PieceSlot)
	}
	if bestMove.BoardIndex < 0 || bestMove.BoardIndex >= core.TotalTriangles {
		t.Errorf("Invalid board index returned: %d", bestMove.BoardIndex)
	}

	fmt.Printf("MCTS Best Move: Slot %d, Index %d\n", bestMove.PieceSlot, bestMove.BoardIndex)
}

func BenchmarkMCTS_SingleThread(b *testing.B) {
	rootState := core.NewGameState([3]int8{0, 1, 0})
	cfg := SearchConfig{
		Simulations: 1000,
		Threads:     1,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ParallelSearch(rootState, cfg)
	}
}

func BenchmarkMCTS_QuadThread(b *testing.B) {
	rnd := rand.New(rand.NewSource(42))
	rootState := core.NewGameState([3]int8{
		RandomPiece(rnd), RandomPiece(rnd), RandomPiece(rnd),
	})

	cfg := SearchConfig{
		Simulations: 1000,
		Threads:     4,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ParallelSearch(rootState, cfg)
	}
}
