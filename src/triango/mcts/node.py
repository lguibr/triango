import math
from typing import Optional

from triango.env.constants import TOTAL_TRIANGLES
from triango.env.pieces import STANDARD_PIECES
from triango.env.state import GameState


class Node:
    __slots__ = [
        "state",
        "move",
        "visits",
        "value_sum",
        "prior",
        "parent",
        "children",
        "untried",
        "expanded",
        "virtual_loss",
    ]

    def __init__(
        self,
        state: GameState,
        parent: Optional["Node"] = None,
        move: tuple[int, int] | None = None,
    ):
        self.state = state
        self.move = move
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0
        self.parent = parent
        self.children: list[Node] = []
        self.untried: list[tuple[int, int]] = []
        self.expanded = False
        self.virtual_loss = 0

        if not self.state.terminal:
            for slot in range(3):
                p_id = self.state.available[slot]
                if p_id == -1:
                    continue
                # We reverse iter so pop() is O(1) from the end
                for idx in range(TOTAL_TRIANGLES - 1, -1, -1):
                    m = STANDARD_PIECES[p_id][idx]
                    if m != 0 and (self.state.board & m) == 0:
                        self.untried.append((slot, idx))

        if len(self.untried) == 0:
            self.expanded = True

    def puct(self, c_puct: float = 1.5) -> float:
        # First Play Urgency (FPU): If node is unvisited, assume it's roughly as good as its parent
        if self.visits + self.virtual_loss == 0:
            exploit = self.parent.value_sum / max(1, self.parent.visits) if self.parent else 0.0
        else:
            exploit = (self.value_sum - self.virtual_loss) / (self.visits + self.virtual_loss)

        parent_visits = self.parent.visits + self.parent.virtual_loss if self.parent else 1
        explore = (
            c_puct * self.prior * math.sqrt(parent_visits) / (1.0 + self.visits + self.virtual_loss)
        )
        return exploit + explore

    def select_child(self) -> Optional["Node"]:
        best_child = None
        best_score = -float("inf")
        for child in self.children:
            score = child.puct()
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self) -> "Node":
        if len(self.untried) == 0:
            return self

        slot, idx = self.untried.pop()
        next_state = self.state.apply_move(slot, idx)

        if next_state is None:
            # Should not happen physically, but helps type hinting
            return self

        if len(self.untried) == 0:
            self.expanded = True

        child = Node(next_state, self, (slot, idx))
        self.children.append(child)
        return child

    def backpropagate(self, reward: float) -> None:
        curr: Node | None = self
        while curr is not None:
            curr.visits += 1
            curr.value_sum += reward
            curr = curr.parent
