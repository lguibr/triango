package core

import (
	"fmt"
	"strings"
	"testing"
)

func printMonochrome(b Bitboard) string {
	var sb strings.Builder
	for r := 0; r < 8; r++ {
		// Padding for visual alignment
		padding := ""
		if r == 0 || r == 7 {
			padding = "   "
		} else if r == 1 || r == 6 {
			padding = "  "
		} else if r == 2 || r == 5 {
			padding = " "
		}
		sb.WriteString(padding)
		for c := 0; c < RowLengths[r]; c++ {
			if b.HasBit(FlatIndex(r, c)) {
				sb.WriteString("X ")
			} else {
				sb.WriteString(". ")
			}
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

func TestGridMasks(t *testing.T) {
	if len(AllMasks) != 24 {
		t.Fatalf("Expected 24 masks, got %d", len(AllMasks))
	}

	// Dump the masks to console for visual inspection
	fmt.Println("=== Horizontal Masks ===")
	for i := 0; i < 8; i++ {
	}

	fmt.Println("=== Red Diagonal Masks (/) ===")
	for i := 8; i < 16; i++ {
	}

	fmt.Println("=== Black Diagonal Masks (\\) ===")
	for i := 16; i < 24; i++ {
	}

	// Let's ensure every cell is covered by exactly 3 masks
	coverage := make([]int, TotalTriangles)
	for _, mask := range AllMasks {
		for i := 0; i < TotalTriangles; i++ {
			if mask.HasBit(i) {
				coverage[i]++
			}
		}
	}

	for i, c := range coverage {
		if c != 3 {
			t.Errorf("Cell %d is covered by %d masks, expected 3", i, c)
		}
	}
}
