package core

import (
	"fmt"
	"strings"
)

// PrintBoard returns a visual string representation of a Bitboard in monochrome.
func PrintBoard(b Bitboard) string {
	return PrintBoardColored(b, nil)
}

// PrintBoardColored returns a visual string representation of a Bitboard,
// painting occupied triangles with the matching ANSI color from the tracks.
func PrintBoardColored(b Bitboard, colors *[TotalTriangles]int8) string {
	var sb strings.Builder
	for r := 0; r < 8; r++ {
		// Padding for visual alignment
		pad := 3 - r
		if r >= 4 {
			pad = r - 4
		}

		sb.WriteString(strings.Repeat(" ", pad))
		for c := 0; c < RowLengths[r]; c++ {
			idx := FlatIndex(r, c)
			up := IsUp(r, c)

			if b.HasBit(idx) {
				colorPrefix := ""
				colorSuffix := ""
				if colors != nil && colors[idx] != -1 {
					colorPrefix = StandardPieces[colors[idx]].ANSIColor
					colorSuffix = ANSIReset
				}

				if up {
					sb.WriteString(colorPrefix + "▲" + colorSuffix)
				} else {
					sb.WriteString(colorPrefix + "▼" + colorSuffix)
				}
			} else {
				if up {
					sb.WriteString("△")
				} else {
					sb.WriteString("▽")
				}
			}
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// PrintTray returns a side-by-side or stacked visual representation
// of the available pieces in the tray.
func PrintTray(pieces [3]int8) string {
	var sb strings.Builder
	for i, pid := range pieces {
		if pid == -1 {
			sb.WriteString(fmt.Sprintf("Slot %d: Empty\n\n", i))
			continue
		}

		p := StandardPieces[pid]
		sb.WriteString(fmt.Sprintf("Slot %d: [%s%s%s]\n", i, p.ANSIColor, p.ColorName, ANSIReset))

		var bestMask Bitboard
		for idx := 40; idx < TotalTriangles; idx++ {
			if !p.Masks[idx].Equals(EmptyBitboard) {
				bestMask = p.Masks[idx]
				break
			}
		}
		// Fallback if 40 onwards doesn't work (rare)
		if bestMask.Equals(EmptyBitboard) {
			for idx := 0; idx < TotalTriangles; idx++ {
				if !p.Masks[idx].Equals(EmptyBitboard) {
					bestMask = p.Masks[idx]
					break
				}
			}
		}

		if bestMask.Equals(EmptyBitboard) {
			continue
		}

		minR, maxR := 8, -1
		minC, maxC := 20, -1
		for r := 0; r < 8; r++ {
			for c := 0; c < RowLengths[r]; c++ {
				if bestMask.HasBit(FlatIndex(r, c)) {
					if r < minR {
						minR = r
					}
					if r > maxR {
						maxR = r
					}
					if c < minC {
						minC = c
					}
					if c > maxC {
						maxC = c
					}
				}
			}
		}

		for r := minR; r <= maxR; r++ {
			pad := 3 - r
			if r >= 4 {
				pad = r - 4
			}
			// Adjust pad safely if minR > 3
			if pad < 0 {
				pad = 0
			}

			sb.WriteString("   " + strings.Repeat(" ", pad))
			for c := minC; c <= maxC; c++ {
				if c >= RowLengths[r] {
					break // out of row bounds
				}
				if bestMask.HasBit(FlatIndex(r, c)) {
					if IsUp(r, c) {
						sb.WriteString(p.ANSIColor + "▲" + ANSIReset)
					} else {
						sb.WriteString(p.ANSIColor + "▼" + ANSIReset)
					}
				} else {
					// We use blank space so ONLY the piece is drawn
					sb.WriteString(" ")
				}
			}
			sb.WriteString("\n")
		}
		sb.WriteString("\n")
	}
	return sb.String()
}
