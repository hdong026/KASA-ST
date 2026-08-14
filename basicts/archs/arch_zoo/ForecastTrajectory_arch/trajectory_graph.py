"""ForecastTrajectory state graph: increasing-resolution legal transitions."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Optional, Sequence


START_STATE = 0


class ForecastTrajectoryGraph:
    """Directed increasing-resolution forecast state graph.

    Nodes are ``{START=0} ∪ states``.  A directed edge ``s -> s'`` exists iff
    ``s' > s`` and ``s'`` is a configured forecasting resolution (or START is
    the implicit source).  Every terminal trajectory ends at ``H``.
    """

    def __init__(
        self,
        H: int = 12,
        states: Optional[Sequence[int]] = None,
    ):
        self.H = int(H)
        raw_states = list(states) if states is not None else [2, 3, 4, 6, 12]
        if not raw_states:
            raise ValueError("states must be non-empty")
        if any(int(s) <= START_STATE for s in raw_states):
            raise ValueError("forecasting states must be strictly greater than START=0")
        ordered = sorted({int(s) for s in raw_states})
        if ordered[-1] != self.H:
            raise ValueError(
                f"states must terminate at H={self.H}, got last state {ordered[-1]}"
            )
        if self.H not in ordered:
            raise ValueError(f"H={self.H} must be included in states")
        self.states = ordered
        self.START = START_STATE
        self.nodes = [self.START] + [s for s in self.states if s != self.START]
        # H is already in states; ensure it is the last node.
        if self.nodes[-1] != self.H:
            self.nodes = [n for n in self.nodes if n != self.H] + [self.H]
        self._edges = self._enumerate_edges()
        self._edge_set = set(self._edges)
        self._trajectories = self._enumerate_terminal_trajectories()
        self._successors = {s: [] for s in self.nodes}
        for src, dst in self._edges:
            self._successors[src].append(dst)
        for s in self._successors:
            self._successors[s].sort()

    def _enumerate_edges(self) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []
        for i, src in enumerate(self.nodes):
            for dst in self.nodes[i + 1 :]:
                if dst > src:
                    edges.append((int(src), int(dst)))
        return edges

    def _enumerate_terminal_trajectories(self) -> list[tuple[int, ...]]:
        """All START→…→H paths. Intermediate states are optional subsets."""
        intermediates = [s for s in self.states if s != self.H]
        trajectories: list[tuple[int, ...]] = []
        n = len(intermediates)
        for r in range(n + 1):
            for combo in combinations(intermediates, r):
                # combinations already yields increasing order for a sorted input.
                trajectories.append(tuple(list(combo) + [self.H]))
        trajectories.sort(key=lambda t: (len(t), t))
        return trajectories

    def legal_edges(self) -> list[tuple[int, int]]:
        return list(self._edges)

    def terminal_trajectories(self) -> list[tuple[int, ...]]:
        return list(self._trajectories)

    def successors(self, state: int) -> list[int]:
        return list(self._successors.get(int(state), []))

    def is_legal_edge(self, s_prev: int, s_next: int) -> bool:
        return (int(s_prev), int(s_next)) in self._edge_set

    def edges_of_trajectory(self, trajectory: Sequence[int]) -> list[tuple[int, int]]:
        tau = [int(s) for s in trajectory]
        if not tau:
            raise ValueError("trajectory must be non-empty")
        if tau[-1] != self.H:
            raise ValueError(f"trajectory must terminate at H={self.H}, got {tau}")
        edges = [(self.START, tau[0])]
        for a, b in zip(tau[:-1], tau[1:]):
            edges.append((a, b))
        for e in edges:
            if e not in self._edge_set:
                raise ValueError(f"illegal edge {e} in trajectory {tau}")
        return edges

    def prefixes(self, trajectory: Sequence[int], include_empty: bool = True) -> list[tuple[int, ...]]:
        tau = tuple(int(s) for s in trajectory)
        out: list[tuple[int, ...]] = [()] if include_empty else []
        for i in range(len(tau)):
            out.append(tau[: i + 1])
        return out

    def all_nonterminal_prefixes(self) -> list[tuple[int, ...]]:
        """Unique prefixes that do not yet include H (includes empty START)."""
        seen = set()
        prefixes: list[tuple[int, ...]] = []
        for tau in self._trajectories:
            for pref in self.prefixes(tau, include_empty=True):
                if pref and pref[-1] == self.H:
                    continue
                if pref not in seen:
                    seen.add(pref)
                    prefixes.append(pref)
        prefixes.sort(key=lambda p: (len(p), p))
        return prefixes

    def current_state(self, prefix: Sequence[int]) -> int:
        if not prefix:
            return self.START
        return int(prefix[-1])

    def assert_h12_defaults(self) -> None:
        if self.H != 12 or self.states != [2, 3, 4, 6, 12]:
            raise AssertionError(
                f"default graph expected H=12 states=[2,3,4,6,12], "
                f"got H={self.H} states={self.states}"
            )
        nodes = list(self.nodes)
        if nodes != [0, 2, 3, 4, 6, 12]:
            raise AssertionError(f"expected nodes [0,2,3,4,6,12], got {nodes}")
        n_edges = len(self._edges)
        n_tau = len(self._trajectories)
        if n_edges != 15:
            raise AssertionError(f"expected 15 legal edges, got {n_edges}: {self._edges}")
        if n_tau != 16:
            raise AssertionError(
                f"expected 16 terminal trajectories, got {n_tau}: {self._trajectories}"
            )

    def trajectory_key(self, trajectory: Sequence[int]) -> str:
        return "-".join(str(int(s)) for s in trajectory)

    def parse_trajectory_key(self, key: str) -> tuple[int, ...]:
        return tuple(int(x) for x in str(key).split("-") if x)

    def dense_trajectory(self) -> tuple[int, ...]:
        return tuple(self.states)

    def direct_trajectory(self) -> tuple[int, ...]:
        return (self.H,)

    def min_finish_edge_cost(
        self,
        edge_cost: dict[tuple[int, int], float],
        extra_per_edge: float = 0.0,
    ) -> dict[int, float]:
        """Dynamic program: cheapest remaining edge-path cost from each state to H.

        ``extra_per_edge`` may include a measured policy-step latency.
        History-encoder cost is NOT included (paid once at START).
        """
        finish = {int(s): float("inf") for s in self.nodes}
        finish[self.H] = 0.0
        for src in reversed(self.nodes):
            if src == self.H:
                continue
            best = float("inf")
            for dst in self.successors(src):
                c = float(edge_cost[(src, dst)]) + float(extra_per_edge) + finish[dst]
                if c < best:
                    best = c
            finish[src] = best
        return finish

    def edge_feasible(
        self,
        s_prev: int,
        s_next: int,
        remaining_ms: float,
        edge_cost: dict[tuple[int, int], float],
        min_finish: dict[int, float],
        extra_per_edge: float = 0.0,
    ) -> bool:
        if not self.is_legal_edge(s_prev, s_next):
            return False
        need = (
            float(edge_cost[(int(s_prev), int(s_next))])
            + float(extra_per_edge)
            + float(min_finish[int(s_next)])
        )
        return need <= float(remaining_ms) + 1e-9

    def summary(self) -> dict:
        return {
            "H": self.H,
            "START": self.START,
            "states": list(self.states),
            "nodes": list(self.nodes),
            "n_edges": len(self._edges),
            "n_trajectories": len(self._trajectories),
            "edges": [list(e) for e in self._edges],
            "trajectories": [list(t) for t in self._trajectories],
            "nonterminal_prefixes": [list(p) for p in self.all_nonterminal_prefixes()],
        }


def default_graph() -> ForecastTrajectoryGraph:
    g = ForecastTrajectoryGraph(H=12, states=[2, 3, 4, 6, 12])
    g.assert_h12_defaults()
    return g


def trajectories_from_states(H: int, states: Iterable[int]) -> list[tuple[int, ...]]:
    return ForecastTrajectoryGraph(H=H, states=list(states)).terminal_trajectories()
