package mcts

import (
	"math/rand"
	"sync"
	"time"
	"triango/core"
)

// RandomPiece returns a random standard piece ID
func RandomPiece(rnd *rand.Rand) int8 {
	return int8(rnd.Intn(len(core.StandardPieces)))
}

// FindValidMoves returns all uniformly valid moves for the given state.
func FindValidMoves(s *core.GameState) []Move {
	var moves []Move
	for slot := 0; slot < 3; slot++ {
		pid := s.AvailablePieces[slot]
		if pid == -1 {
			continue
		}
		p := core.StandardPieces[pid]
		for idx := 0; idx < core.TotalTriangles; idx++ {
			m := p.Masks[idx]
			if !m.Equals(core.EmptyBitboard) && !s.Board.Intersects(m) {
				moves = append(moves, Move{slot, idx})
			}
		}
	}
	return moves
}

// Rollout Simulates a random game from the given state until terminal and returns Score delta
func Rollout(s core.GameState, rnd *rand.Rand) float64 {
	curr := s
	startScore := s.Score

	// Terminal states naturally bound the depth
	for !curr.IsTerminal {
		if curr.PiecesLeft == 0 {
			curr = curr.RefillTray([3]int8{RandomPiece(rnd), RandomPiece(rnd), RandomPiece(rnd)})
			if curr.IsTerminal {
				break
			}
		}

		moves := FindValidMoves(&curr)
		if len(moves) == 0 {
			break
		}

		rm := moves[rnd.Intn(len(moves))]
		next, valid := curr.ApplyMove(rm.PieceSlot, rm.BoardIndex)
		if !valid {
			break
		}
		curr = next
	}

	return float64(curr.Score - startScore)
}

// SearchConfig configures the MCTS execution.
type SearchConfig struct {
	Simulations int
	Threads     int
	MaxTime     time.Duration
}

type ThreadResult struct {
	Visits   map[Move]int
	ValueSum map[Move]float64
}

// ParallelSearch uses root parallelization to build multiple independent MCTS trees
// totally lock-free and merges the visit counts at the root level asynchronously.
func ParallelSearch(rootState core.GameState, config SearchConfig) Move {
	if config.Threads <= 0 {
		config.Threads = 1
	}

	simsPerThread := config.Simulations / config.Threads
	results := make(chan ThreadResult, config.Threads)
	var wg sync.WaitGroup

	for i := 0; i < config.Threads; i++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()

			// Independent local random generator for pure lock-free throughput
			rnd := rand.New(rand.NewSource(seed))

			root := NewNode(rootState, nil, Move{})

			for s := 0; s < simsPerThread; s++ {
				node := root

				// 1. Select
				for len(node.Untried) == 0 && len(node.Children) > 0 {
					node = node.SelectChild()
				}

				// 2. Expand
				if !node.State.IsTerminal && len(node.Untried) > 0 {
					node = node.Expand()
				}

				// 3. Simulate
				reward := Rollout(node.State, rnd)

				// 4. Backpropagate
				node.Backpropagate(reward)
			}

			res := ThreadResult{
				Visits:   make(map[Move]int),
				ValueSum: make(map[Move]float64),
			}
			for _, c := range root.Children {
				res.Visits[c.MoveInfo] = c.Visits
				res.ValueSum[c.MoveInfo] = c.ValueSum
			}
			results <- res
		}(time.Now().UnixNano() + int64(i))
	}

	wg.Wait()
	close(results)

	// Merge root results purely sequentially (no locks)
	mergedVisits := make(map[Move]int)
	for r := range results {
		for m, v := range r.Visits {
			mergedVisits[m] += v
		}
	}

	// Select best robust move (highest absolute traversal volume)
	var bestMove Move
	bestVisits := -1
	for m, v := range mergedVisits {
		if v > bestVisits {
			bestVisits = v
			bestMove = m
		}
	}

	return bestMove
}
