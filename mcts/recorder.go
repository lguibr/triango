package mcts

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"triango/core"
)

// DataRecord represents a single SOTA observation for PyTorch training.
// It maps the pure Triango bitboards into dense feature arrays so
// Python can simply ingest it natively directly to CUDA Tensors.
type DataRecord struct {
	// Features [TotalTriangles]int
	// 1 if empty, 0 if occupied (we can make it more complex later like showing piece type)
	Board [core.TotalTriangles]int8 `json:"board"`

	// Pieces available to play [0,0,0,1,0...] Length = len(StandardPieces)
	Tray [3]int8 `json:"tray"`

	// Policy array where Policy[idx] is the MCTS visit probability of playing at that index.
	// For simplicity in this flat array notation, we map board index directly.
	// If a piece from a specific slot needs to be tracked, we can expand this.
	// For AlphaZero standard, this is purely the target Prior probability.
	Policy [core.TotalTriangles]float32 `json:"policy"`

	// Value is the final score of the game from this perspective.
	// It is populated after the game ends.
	Value float32 `json:"value"`
}

// Recorder handles lock-free parallel streaming of JSONL training data
type Recorder struct {
	file *os.File
	mu   sync.Mutex
	// We buffer records in memory because we don't know the final Value (score)
	// until the game actually legally terminates.
	gameBuffer []DataRecord
}

// NewRecorder opens a JSONL file for appending data
func NewRecorder(path string) (*Recorder, error) {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open dataset file: %w", err)
	}
	return &Recorder{
		file:       f,
		gameBuffer: make([]DataRecord, 0, 200),
	}, nil
}

// RecordStep converts a GameState and the aggregated MCTS visits into a DataRecord
func (r *Recorder) RecordStep(state core.GameState, visits map[Move]int) {
	record := DataRecord{}

	// 1. Encode Board
	for i := 0; i < core.TotalTriangles; i++ {
		if state.Board.HasBit(i) {
			record.Board[i] = 1
		} else {
			record.Board[i] = 0
		}
	}

	// 2. Encode Tray
	for i := 0; i < 3; i++ {
		record.Tray[i] = state.AvailablePieces[i]
	}

	// 3. Compute Normalized Policy (Target P)
	totalVisits := 0
	for _, v := range visits {
		totalVisits += v
	}

	if totalVisits > 0 {
		// Aggregate visits by board index (simplified target)
		indexVisits := make(map[int]int)
		for m, v := range visits {
			indexVisits[m.BoardIndex] += v
		}

		for idx, count := range indexVisits {
			record.Policy[idx] = float32(count) / float32(totalVisits)
		}
	}

	// 4. Save to temporary game buffer (Value is unknown right now)
	r.mu.Lock()
	r.gameBuffer = append(r.gameBuffer, record)
	r.mu.Unlock()
}

// FlushGame backpropagates the final actual game score to all steps and writes JSONL
func (r *Recorder) FlushGame(finalScore int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	encoder := json.NewEncoder(r.file)
	for i := range r.gameBuffer {
		// Backpropagate the ground truth Reward (Value)
		r.gameBuffer[i].Value = float32(finalScore)
		if err := encoder.Encode(r.gameBuffer[i]); err != nil {
			return fmt.Errorf("failed to encode record: %w", err)
		}
	}

	// Reset buffer for the next game
	r.gameBuffer = r.gameBuffer[:0]
	return nil
}

// Close gracefully flushes the file descriptor
func (r *Recorder) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.file.Close()
}
