package core

import "fmt"

var (
	// AllMasks contains all 24 line-clearing masks (8 horizontal, 8 red diagonal, 8 black diagonal).
	AllMasks []Bitboard

	RowLengths = []int{9, 11, 13, 15, 15, 13, 11, 9}
)

const TotalTriangles = 96

func init() {
	generateMasks()
}

// FlatIndex returns the 0-95 index for a given (row, col).
func FlatIndex(r, c int) int {
	idx := 0
	for i := 0; i < r; i++ {
		idx += RowLengths[i]
	}
	idx += c
	return idx
}

// GetRowCol returns the (row, col) for a given 0-95 index.
func GetRowCol(idx int) (int, int) {
	rem := idx
	for r := 0; r < 8; r++ {
		if rem < RowLengths[r] {
			return r, rem
		}
		rem -= RowLengths[r]
	}
	return -1, -1
}

// IsUp returns true if the triangle at (row, col) points UP.
func IsUp(r, c int) bool {
	if r < 4 {
		return c%2 == 0
	}
	return c%2 == 1
}

// IsUpFlat evaluates polarity directly from a flat index.
func IsUpFlat(idx int) bool {
	r, c := GetRowCol(idx)
	return IsUp(r, c)
}

// VerticalNeighbor returns the (row, col) of the triangle sharing the horizontal edge.
// For UP triangles, it's the triangle below. For DOWN triangles, it's the triangle above.
// Returns -1, -1 if no neighbor exists (board boundary).
func VerticalNeighbor(r, c int) (int, int) {
	if IsUp(r, c) {
		// UP triangle: neighbor is DOWN, in row r+1
		if r == 7 {
			return -1, -1 // Bottom edge
		}
		if r < 3 {
			return r + 1, c + 1
		} else if r == 3 {
			return r + 1, c
		} else {
			return r + 1, c - 1
		}
	} else {
		// DOWN triangle: neighbor is UP, in row r-1
		if r == 0 {
			return -1, -1 // Top edge
		}
		if r < 4 {
			return r - 1, c - 1
		} else if r == 4 {
			return r - 1, c
		} else {
			return r - 1, c + 1
		}
	}
}

func generateMasks() {
	AllMasks = make([]Bitboard, 0, 24)

	// 1. Horizontal Masks (8 rows)
	for r := 0; r < 8; r++ {
		var mask Bitboard
		for c := 0; c < RowLengths[r]; c++ {
			mask.SetBit(FlatIndex(r, c))
		}
		AllMasks = append(AllMasks, mask)
	}

	// 2. Red Diagonal Lines (/)
	// A / strip goes UP-RIGHT. Alternate right -> up -> right -> up.
	redLines := extractLines(func(r, c int) (int, int) { // Next in / direction
		if IsUp(r, c) {
			if c+1 < RowLengths[r] {
				return r, c + 1 // cross right edge
			}
		} else {
			nr, nc := VerticalNeighbor(r, c) // cross top edge
			if nr != -1 {
				return nr, nc
			}
		}
		return -1, -1
	})

	// 3. Black Diagonal Lines (\)
	// A \ strip goes UP-LEFT. Alternate left -> up -> left -> up.
	blackLines := extractLines(func(r, c int) (int, int) { // Next in \ direction
		if IsUp(r, c) {
			if c-1 >= 0 {
				return r, c - 1 // cross left edge
			}
		} else {
			nr, nc := VerticalNeighbor(r, c) // cross top edge
			if nr != -1 {
				return nr, nc
			}
		}
		return -1, -1
	})

	AllMasks = append(AllMasks, redLines...)
	AllMasks = append(AllMasks, blackLines...)

	if len(AllMasks) != 24 {
		panic(fmt.Sprintf("Expected 24 line masks, generated %d", len(AllMasks)))
	}
}

func extractLines(next func(r, c int) (int, int)) []Bitboard {
	visited := make([]bool, TotalTriangles)
	var lines []Bitboard

	for r := 0; r < 8; r++ {
		for c := 0; c < RowLengths[r]; c++ {
			idx := FlatIndex(r, c)
			if visited[idx] {
				continue
			}

			// Traverse backwards to find the absolute start of this line strip
			startR, startC := r, c
			for {
				prevR, prevC := -1, -1
				// Check which valid neighbor points to startR, startC
				for pr := 0; pr < 8; pr++ {
					for pc := 0; pc < RowLengths[pr]; pc++ {
						nr, nc := next(pr, pc)
						if nr == startR && nc == startC {
							prevR, prevC = pr, pc
						}
					}
				}
				if prevR == -1 {
					break // Found the absolute start
				}
				startR, startC = prevR, prevC
			}

			// Now traverse forwards and build the mask
			var mask Bitboard
			currR, currC := startR, startC
			length := 0
			for currR != -1 && currC != -1 {
				idx := FlatIndex(currR, currC)
				mask.SetBit(idx)
				visited[idx] = true
				currR, currC = next(currR, currC)
				length++
			}

			// In a normal Tricrack board, line masks must completely span the board.
			// A valid line mask is one that touches boundaries fully.
			// All maximal strips generated this way are valid line masks.
			lines = append(lines, mask)
		}
	}
	return lines
}
