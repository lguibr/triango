package main

import (
	"fmt"
	"math/rand"
	"time"
	"triango/core"
	"triango/mcts"
)

func main() {
	recorder, err := mcts.NewRecorder("data/games.jsonl")
	if err != nil {
		fmt.Printf("Failed to init recorder: %v\n", err)
		return
	}
	defer recorder.Close()

	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))
	s := core.NewGameState([3]int8{
		mcts.RandomPiece(rnd),
		mcts.RandomPiece(rnd),
		mcts.RandomPiece(rnd),
	})

	cfg := mcts.SearchConfig{
		Simulations: 20000,
		Threads:     4,
	}

	var boardColors [core.TotalTriangles]int8
	for i := 0; i < core.TotalTriangles; i++ {
		boardColors[i] = -1
	}

	turn := 1
	for !s.IsTerminal {
		if s.PiecesLeft == 0 {
			s = s.RefillTray([3]int8{
				mcts.RandomPiece(rnd),
				mcts.RandomPiece(rnd),
				mcts.RandomPiece(rnd),
			})
			if s.IsTerminal {
				break
			}
		}

		fmt.Printf("\n==================================\n")
		fmt.Printf("=== Turn %d | Score: %d ===\n", turn, s.Score)
		fmt.Printf("Tray:\n")
		fmt.Println(core.PrintTray(s.AvailablePieces))

		fmt.Println("\nCurrent Board:")
		fmt.Println(core.PrintBoardColored(s.Board, &boardColors))

		// MCTS Think
		fmt.Printf("Thinking (Simulations: %d on %d cores)...\n", cfg.Simulations, cfg.Threads)
		start := time.Now()
		bestMove, visits := mcts.ParallelSearch(s, cfg)
		elapsed := time.Since(start)

		recorder.RecordStep(s, visits)

		p := core.StandardPieces[s.AvailablePieces[bestMove.PieceSlot]]
		fmt.Printf("\n-> Agent placed [%s%s%s] piece at index %d (took %v)\n", p.ANSIColor, p.ColorName, core.ANSIReset, bestMove.BoardIndex, elapsed)

		next, valid := s.ApplyMove(bestMove.PieceSlot, bestMove.BoardIndex)
		if !valid {
			fmt.Println("CRITICAL ERROR: Agent generated an invalid move!")
			break
		}

		mask := p.Masks[bestMove.BoardIndex]
		cleared := s.Board.Or(mask).AndNot(next.Board)

		for i := 0; i < core.TotalTriangles; i++ {
			if cleared.HasBit(i) {
				boardColors[i] = -1
			} else if mask.HasBit(i) {
				boardColors[i] = int8(p.ID)
			}
		}

		s = next
		turn++
	}

	fmt.Printf("\n==================================\n")
	fmt.Printf("=== GAME OVER ===\nFinal Score: %d\n", s.Score)
	fmt.Println("Final Board:")
	fmt.Println(core.PrintBoardColored(s.Board, &boardColors))

	if err := recorder.FlushGame(s.Score); err != nil {
		fmt.Printf("Failed to flush dataset: %v\n", err)
	} else {
		fmt.Printf("Successfully saved self-play record to data/games.jsonl!\n")
	}
}
