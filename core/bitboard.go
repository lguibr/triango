package core

import (
	"math/bits"
)

// Bitboard represents a 128-bit state mask, capable of holding up to 128 triangles.
// Since the Tricrack board has 96 triangles, this fits perfectly.
// b[0] holds bits 0-63.
// b[1] holds bits 64-127.
type Bitboard [2]uint64

// EmptyBitboard is a predefined empty board.
var EmptyBitboard = Bitboard{0, 0}

// SetBit sets a specific bit on the board.
func (b *Bitboard) SetBit(index int) {
	if index < 64 {
		b[0] |= (1 << index)
	} else {
		b[1] |= (1 << (index - 64))
	}
}

// ClearBit clears a specific bit.
func (b *Bitboard) ClearBit(index int) {
	if index < 64 {
		b[0] &^= (1 << index)
	} else {
		b[1] &^= (1 << (index - 64))
	}
}

// HasBit checks if a specific bit is set.
func (b *Bitboard) HasBit(index int) bool {
	if index < 64 {
		return (b[0] & (1 << index)) != 0
	}
	return (b[1] & (1 << (index - 64))) != 0
}

// Or performs a bitwise OR operation.
func (b Bitboard) Or(other Bitboard) Bitboard {
	return Bitboard{b[0] | other[0], b[1] | other[1]}
}

// And performs a bitwise AND operation.
func (b Bitboard) And(other Bitboard) Bitboard {
	return Bitboard{b[0] & other[0], b[1] & other[1]}
}

// AndNot performs a bitwise AND NOT operation (b &^ other).
func (b Bitboard) AndNot(other Bitboard) Bitboard {
	return Bitboard{b[0] &^ other[0], b[1] &^ other[1]}
}

// Intersects checks if two bitboards share any bits.
func (b Bitboard) Intersects(other Bitboard) bool {
	return (b[0]&other[0]) != 0 || (b[1]&other[1]) != 0
}

// Contains checks if `b` contains all bits of `other`.
func (b Bitboard) Contains(other Bitboard) bool {
	return (b[0]&other[0]) == other[0] && (b[1]&other[1]) == other[1]
}

// PopCount returns the number of set bits (Hamming weight).
func (b Bitboard) PopCount() int {
	return bits.OnesCount64(b[0]) + bits.OnesCount64(b[1])
}

// Equals checks if two bitboards are identical.
func (b Bitboard) Equals(other Bitboard) bool {
	return b[0] == other[0] && b[1] == other[1]
}
