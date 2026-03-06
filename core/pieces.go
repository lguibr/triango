package core

import "fmt"

type Coord struct {
	X int
	Y int
	Z int
}

// Global maps for coordinate translations
var (
	IndexToCoord [TotalTriangles]Coord
	CoordToIndex map[Coord]int
)

func init() {
	CoordToIndex = make(map[Coord]int)
	buildCoords()
}

func buildCoords() {
	// Start at Top-Left UP triangle (index 0)
	// We assign it (0, 0, 1) and flood fill the rest of the board.
	visited := make([]bool, TotalTriangles)

	// Queue for BFS flood-fill
	queue := []int{0}
	IndexToCoord[0] = Coord{0, 0, 1}
	visited[0] = true

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		p := IndexToCoord[curr]

		r, c := -1, -1
		for ir := 0; ir < 8; ir++ {
			if curr >= FlatIndex(ir, 0) && curr < FlatIndex(ir, RowLengths[ir]) {
				r = ir
				c = curr - FlatIndex(ir, 0)
				break
			}
		}

		if IsUp(r, c) {
			// UP -> Right edge -> DOWN at (r, c+1)
			if c+1 < RowLengths[r] {
				n := FlatIndex(r, c+1)
				assignCoord(n, Coord{p.X, p.Y, p.Z - 1}, &queue, visited)
			}
			// UP -> Left edge -> DOWN at (r, c-1)
			if c-1 >= 0 {
				n := FlatIndex(r, c-1)
				assignCoord(n, Coord{p.X - 1, p.Y, p.Z}, &queue, visited)
			}
			// UP -> Bottom edge -> DOWN VerticalNeighbor
			nr, nc := VerticalNeighbor(r, c)
			if nr != -1 {
				n := FlatIndex(nr, nc)
				assignCoord(n, Coord{p.X, p.Y - 1, p.Z}, &queue, visited)
			}
		} else {
			// DOWN -> Right edge -> UP at (r, c+1)
			if c+1 < RowLengths[r] {
				n := FlatIndex(r, c+1)
				assignCoord(n, Coord{p.X + 1, p.Y, p.Z}, &queue, visited)
			}
			// DOWN -> Left edge -> UP at (r, c-1)
			if c-1 >= 0 {
				n := FlatIndex(r, c-1)
				assignCoord(n, Coord{p.X, p.Y, p.Z + 1}, &queue, visited)
			}
			// DOWN -> Top edge -> UP VerticalNeighbor
			nr, nc := VerticalNeighbor(r, c)
			if nr != -1 {
				n := FlatIndex(nr, nc)
				assignCoord(n, Coord{p.X, p.Y + 1, p.Z}, &queue, visited)
			}
		}
	}

	for i := 0; i < TotalTriangles; i++ {
		if !visited[i] {
			panic(fmt.Sprintf("Triangle %d was not reached during coordinate flood fill!", i))
		}
		CoordToIndex[IndexToCoord[i]] = i
	}
}

func assignCoord(idx int, c Coord, queue *[]int, visited []bool) {
	if !visited[idx] {
		IndexToCoord[idx] = c
		visited[idx] = true
		*queue = append(*queue, idx)
	} else {
		// Verify consistency
		if IndexToCoord[idx] != c {
			panic(fmt.Sprintf("Coordinate inconsistency at index %d: expected %v, got %v", idx, IndexToCoord[idx], c))
		}
	}
}

// --- Piece Definition ---

// ANSI Colors
const (
	ANSIReset  = "\033[0m"
	ANSIRed    = "\033[31m"
	ANSIYellow = "\033[33m"
	ANSIBlue   = "\033[34m"
	ANSIPurple = "\033[35m"
	ANSICyan   = "\033[36m"
	ANSIOrange = "\033[38;5;208m"
	ANSIGreen  = "\033[32m"
)

// PieceDefinition represents the relative coordinate offsets defining a shape.
// The first offset is always (0,0,0) representing the origin triangle.
type PieceDefinition struct {
	RequireUp   bool
	RequireDown bool
	Offsets     []Coord
	ColorName   string
	ANSIColor   string
}

// Piece represents a compiled shape with precomputed masks for all 96 valid origin positions.
type Piece struct {
	ID        int
	ColorName string
	ANSIColor string
	// Masks[i] is the Bitboard when the piece's origin is placed at board index `i`.
	// If placement is out of bounds or invalid polarity, Masks[i] == EmptyBitboard.
	Masks [TotalTriangles]Bitboard
}

// CompilePiece takes a PieceDefinition and generates the absolute bitboards.
func CompilePiece(id int, def PieceDefinition) Piece {
	p := Piece{
		ID:        id,
		ColorName: def.ColorName,
		ANSIColor: def.ANSIColor,
	}

	for i := 0; i < TotalTriangles; i++ {
		isUp := IsUpFlat(i)

		// Strict structural polarity binding
		if def.RequireUp && !isUp {
			p.Masks[i] = EmptyBitboard
			continue
		}
		if def.RequireDown && isUp {
			p.Masks[i] = EmptyBitboard
			continue
		}

		originCoord := IndexToCoord[i]
		var mask Bitboard
		valid := true

		for _, off := range def.Offsets {
			targetCoord := Coord{
				X: originCoord.X + off.X,
				Y: originCoord.Y + off.Y,
				Z: originCoord.Z + off.Z,
			}
			idx, exists := CoordToIndex[targetCoord]
			if !exists {
				valid = false
				break
			}
			mask.SetBit(idx)
		}

		if valid {
			p.Masks[i] = mask
		} else {
			p.Masks[i] = EmptyBitboard
		}
	}

	return p
}

// Generate the standard piece library
var StandardPieces []Piece

// Initialize shapes.
func init() {
	// 1. Single Triangle UP
	StandardPieces = append(StandardPieces, CompilePiece(0, PieceDefinition{
		RequireUp: true,
		Offsets:   []Coord{{0, 0, 0}},
		ColorName: "Yellow",
		ANSIColor: ANSIYellow,
	}))

	// 2. Single Triangle DOWN
	StandardPieces = append(StandardPieces, CompilePiece(1, PieceDefinition{
		RequireDown: true,
		Offsets:     []Coord{{0, 0, 0}},
		ColorName:   "Yellow",
		ANSIColor:   ANSIYellow,
	}))

	// 3. Hexagon (6 triangles around a vertex)
	StandardPieces = append(StandardPieces, CompilePiece(2, PieceDefinition{
		RequireUp: true,
		Offsets: []Coord{
			{0, 0, 0}, {0, 0, -1}, {1, 0, -1}, // Top half
			{0, -1, 0}, {1, -1, 0}, {1, -1, -1}, // Bottom half
		},
		ColorName: "Red",
		ANSIColor: ANSIRed,
	}))

	// 4. Pentagon (Hexagon minus Top-Right)
	StandardPieces = append(StandardPieces, CompilePiece(3, PieceDefinition{
		RequireUp: true,
		Offsets: []Coord{
			{0, 0, 0}, {0, 0, -1},
			{0, -1, 0}, {1, -1, 0}, {1, -1, -1},
		},
		ColorName: "Purple",
		ANSIColor: ANSIPurple,
	}))

	// 5. Trapezoid (Hexagon minus top 2 rights) -> Line of 3 + 1 below
	StandardPieces = append(StandardPieces, CompilePiece(4, PieceDefinition{
		RequireUp: true,
		Offsets: []Coord{
			{0, 0, 0}, {0, 0, -1}, {1, 0, -1},
			{0, -1, 0},
		},
		ColorName: "Blue",
		ANSIColor: ANSIBlue,
	}))

	// 6. Large Triangle UP (1 top, 3 bottom)
	StandardPieces = append(StandardPieces, CompilePiece(5, PieceDefinition{
		RequireUp: true,
		Offsets: []Coord{
			{0, 0, 0},
			{0, -1, 1}, {0, -1, 0}, {1, -1, 0},
		},
		ColorName: "Orange",
		ANSIColor: ANSIOrange,
	}))

	// 7. Large Triangle DOWN (3 top, 1 bottom)
	StandardPieces = append(StandardPieces, CompilePiece(6, PieceDefinition{
		RequireDown: true,
		Offsets: []Coord{
			{0, 0, 0}, {1, 0, 0}, {1, 0, -1}, // top 3
			{1, -1, 0}, // bottom 1
		},
		ColorName: "Orange",
		ANSIColor: ANSIOrange,
	}))

	// 8. Line of 3 UP-Anchor
	StandardPieces = append(StandardPieces, CompilePiece(7, PieceDefinition{
		RequireUp: true,
		Offsets: []Coord{
			{0, 0, 0}, {0, 0, -1}, {1, 0, -1},
		},
		ColorName: "Cyan",
		ANSIColor: ANSICyan,
	}))

	// 9. Line of 3 DOWN-Anchor
	StandardPieces = append(StandardPieces, CompilePiece(8, PieceDefinition{
		RequireDown: true,
		Offsets: []Coord{
			{0, 0, 0}, {1, 0, 0}, {1, 0, -1},
		},
		ColorName: "Cyan",
		ANSIColor: ANSICyan,
	}))

	// 10. Pair Horizontal
	StandardPieces = append(StandardPieces, CompilePiece(9, PieceDefinition{
		RequireUp: true,
		Offsets: []Coord{
			{0, 0, 0}, {0, 0, -1},
		},
		ColorName: "Green",
		ANSIColor: ANSIGreen,
	}))

	// 11. Pair Vertical (Rhombus)
	StandardPieces = append(StandardPieces, CompilePiece(10, PieceDefinition{
		RequireUp: true,
		Offsets: []Coord{
			{0, 0, 0},
			{0, -1, 0},
		},
		ColorName: "Green",
		ANSIColor: ANSIGreen,
	}))

	// 12. Chevron UP (3 triangles forming an arrow ^)
	StandardPieces = append(StandardPieces, CompilePiece(11, PieceDefinition{
		RequireUp: true,
		Offsets: []Coord{
			{0, 0, 0},
			{0, -1, 1}, {0, -1, 0},
		},
		ColorName: "Purple",
		ANSIColor: ANSIPurple,
	}))
}
