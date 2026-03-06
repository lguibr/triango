package mcts

import (
	"math"
	"triango/core"
)

// Move represents an action in the game.
type Move struct {
	PieceSlot  int // 0, 1, or 2
	BoardIndex int // 0 to 95
}

// Node represents a state node in the Monte Carlo Tree.
type Node struct {
	State    core.GameState
	MoveInfo Move // The move taken to reach this node (from parent)
	Visits   int
	ValueSum float64
	Parent   *Node
	Children []*Node
	Untried  []Move
	Expanded bool
}

// NewNode creates a new MCTS node and populates its continuous valid moves.
func NewNode(s core.GameState, parent *Node, move Move) *Node {
	n := &Node{
		State:    s,
		MoveInfo: move,
		Parent:   parent,
		Children: nil,
		Untried:  nil,
		Expanded: false,
	}

	if !s.IsTerminal {
		// Enumerate all valid moves using Bitboard intersections
		for slot := 0; slot < 3; slot++ {
			pid := s.AvailablePieces[slot]
			if pid == -1 {
				continue
			}
			p := core.StandardPieces[pid]
			for idx := 0; idx < core.TotalTriangles; idx++ {
				m := p.Masks[idx]
				if !m.Equals(core.EmptyBitboard) && !s.Board.Intersects(m) {
					n.Untried = append(n.Untried, Move{slot, idx})
				}
			}
		}
	}

	if len(n.Untried) == 0 {
		n.Expanded = true // If no moves exist, it is functionally expanded as a terminal
	}

	return n
}

// UCT computes the Upper Confidence Bound for Trees to handle exploration/exploitation trade-off.
func (n *Node) UCT(explorationWeight float64) float64 {
	if n.Visits == 0 {
		return math.Inf(1)
	}
	// Exploitation term: average reward
	exploit := n.ValueSum / float64(n.Visits)

	if n.Parent == nil {
		return exploit
	}

	// Exploration term: C * sqrt(ln(N) / n)
	explore := explorationWeight * math.Sqrt(math.Log(float64(n.Parent.Visits))/float64(n.Visits))
	return exploit + explore
}

// SelectChild selects the best child according to the UCT formula.
func (n *Node) SelectChild() *Node {
	var bestChild *Node
	bestScore := math.Inf(-1)

	// Tuning constant specifically for standard 1010 puzzle scores (which can grow to thousands)
	// We might need a scalable constant, but 1.41 * Max Expected Turn Score usually works.
	explorationC := 10.0

	for _, child := range n.Children {
		score := child.UCT(explorationC)
		if score > bestScore {
			bestScore = score
			bestChild = child
		}
	}
	return bestChild
}

// Expand randomly selects an untried move, applies it entirely immutably, and returns the newly instantiated child.
func (n *Node) Expand() *Node {
	if len(n.Untried) == 0 {
		return n
	}

	// For standard SOTA speed, we just pop from the end of untried moves rather than generating random indexes.
	// Randomness happens during rollout. Expanding sequentially is fine.
	move := n.Untried[len(n.Untried)-1]
	n.Untried = n.Untried[:len(n.Untried)-1]

	nextState, valid := n.State.ApplyMove(move.PieceSlot, move.BoardIndex)
	if !valid {
		// Should completely never happen due to enumeration checks, but defensively panic here
		panic("SOTA enumeration generated invalid move logic!")
	}

	if len(n.Untried) == 0 {
		n.Expanded = true
	}

	child := NewNode(nextState, n, move)
	n.Children = append(n.Children, child)
	return child
}

// Backpropagate updates node statistics up to the Root lock-free.
func (n *Node) Backpropagate(reward float64) {
	curr := n
	for curr != nil {
		curr.Visits++
		curr.ValueSum += reward
		curr = curr.Parent
	}
}
