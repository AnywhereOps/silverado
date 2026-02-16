"""
Match-Strike Engine — The quantum carburetor cycle.

Classical controller fires the quantum VM repeatedly:
  Prepare → Superpose → Phase-kick (problem) → Mix → Measure → Learn → Reset

Supports both state vector (QuantumVM) and tensor network (TensorVM) backends.
Target problem: MaxCut on random graphs.
"""

import numpy as np
from dataclasses import dataclass, field
from quantum_vm import QuantumVM


@dataclass
class MaxCutProblem:
    """A MaxCut instance on a random graph."""
    n_nodes: int
    edges: list[tuple[int, int]]
    weights: list[float]

    @staticmethod
    def random(n_nodes: int, edge_prob: float = 0.5, seed: int | None = None) -> "MaxCutProblem":
        rng = np.random.default_rng(seed)
        edges = []
        weights = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < edge_prob:
                    edges.append((i, j))
                    weights.append(rng.uniform(0.5, 2.0))
        return MaxCutProblem(n_nodes=n_nodes, edges=edges, weights=weights)

    def cut_value(self, bitstring: int) -> float:
        """Compute the cut value for a given partition (as integer)."""
        total = 0.0
        for (i, j), w in zip(self.edges, self.weights):
            bi = (bitstring >> i) & 1
            bj = (bitstring >> j) & 1
            if bi != bj:
                total += w
        return total

    def all_cut_values(self) -> np.ndarray:
        """Compute cut values for all 2^n partitions."""
        dim = 2 ** self.n_nodes
        costs = np.zeros(dim)
        for idx in range(dim):
            costs[idx] = self.cut_value(idx)
        return costs

    def brute_force_optimal(self) -> tuple[int, float]:
        """Find the optimal cut by exhaustive search. Only for small problems."""
        if self.n_nodes > 25:
            raise ValueError("Brute force not feasible for > 25 nodes")
        costs = self.all_cut_values()
        best_idx = int(np.argmax(costs))
        return best_idx, costs[best_idx]

    def to_dict(self) -> dict:
        """Serialize for network transmission."""
        return {
            "n_nodes": self.n_nodes,
            "edges": self.edges,
            "weights": self.weights,
        }

    @staticmethod
    def from_dict(d: dict) -> "MaxCutProblem":
        return MaxCutProblem(
            n_nodes=d["n_nodes"],
            edges=[tuple(e) for e in d["edges"]],
            weights=d["weights"],
        )


@dataclass
class StrikeResult:
    """One ignition event."""
    bitstring: int
    cut_value: float
    sigma_before: float
    sigma_after: float
    gamma: float = 0.0
    beta: float = 0.0


@dataclass
class CycleLog:
    """Full log of a match-strike run."""
    strikes: list[StrikeResult] = field(default_factory=list)
    best_bitstring: int = 0
    best_cut: float = 0.0
    convergence_history: list[float] = field(default_factory=list)
    sigma_history: list[float] = field(default_factory=list)

    def record(self, result: StrikeResult):
        self.strikes.append(result)
        if result.cut_value > self.best_cut:
            self.best_cut = result.cut_value
            self.best_bitstring = result.bitstring
        self.convergence_history.append(self.best_cut)
        self.sigma_history.append(result.sigma_before)


class MatchStrikeEngine:
    """The quantum carburetor. Classical brain, quantum ignition."""

    def __init__(self, problem: MaxCutProblem, n_layers: int = 3,
                 backend: str = "statevector", max_bond: int = 64,
                 seed: int | None = None):
        self.problem = problem
        self.n_layers = n_layers
        self.backend = backend
        self.rng = np.random.default_rng(seed)

        if backend == "statevector":
            self.vm = QuantumVM(n_qubits=problem.n_nodes)
            # Precompute full cost landscape for phase oracle
            self.costs = problem.all_cut_values()
            max_cost = self.costs.max() if self.costs.max() > 0 else 1.0
            self.phase_costs = (self.costs / max_cost) * np.pi
        elif backend == "tensor":
            from tensor_vm import TensorVM
            self.vm = TensorVM(n_qubits=problem.n_nodes, max_bond=max_bond)
            self.costs = None
            self.phase_costs = None
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def single_strike(self, gamma: float, beta: float) -> StrikeResult:
        """One ignition cycle."""
        self.vm.reset()
        self.vm.superpose_all()

        for _ in range(self.n_layers):
            if self.backend == "statevector":
                self.vm.apply_phase_oracle(gamma * self.phase_costs)
            else:
                self.vm.apply_phase_oracle(self.problem, gamma=gamma)
            self.vm.apply_mixer(beta)

        sigma_before = self.vm.sigma()
        bitstring = self.vm.measure()

        return StrikeResult(
            bitstring=bitstring,
            cut_value=self.problem.cut_value(bitstring),
            sigma_before=sigma_before,
            sigma_after=1.0,
            gamma=gamma,
            beta=beta,
        )

    def run(self, n_strikes: int = 500, strategy: str = "random") -> CycleLog:
        """Fire the carburetor n_strikes times."""
        log = CycleLog()

        if strategy == "random":
            for _ in range(n_strikes):
                gamma = self.rng.uniform(0, 2 * np.pi)
                beta = self.rng.uniform(0, np.pi)
                result = self.single_strike(gamma, beta)
                log.record(result)

        elif strategy == "sweep":
            n_gamma = int(np.sqrt(n_strikes))
            n_beta = n_strikes // n_gamma
            for gamma in np.linspace(0.1, 2 * np.pi, n_gamma):
                for beta in np.linspace(0.1, np.pi, n_beta):
                    result = self.single_strike(gamma, beta)
                    log.record(result)

        elif strategy == "greedy":
            best_gamma, best_beta = np.pi, np.pi / 2
            spread = np.pi
            for i in range(n_strikes):
                t = i / max(n_strikes - 1, 1)
                current_spread = spread * (1 - 0.8 * t)
                gamma = best_gamma + self.rng.uniform(-current_spread, current_spread)
                beta = best_beta + self.rng.uniform(-current_spread / 2, current_spread / 2)
                result = self.single_strike(gamma, beta)
                log.record(result)
                if result.cut_value >= log.best_cut:
                    best_gamma = gamma
                    best_beta = beta

        return log

    def run_circuit(self, gamma: float, beta: float, n_shots: int = 1) -> list[StrikeResult]:
        """Run a specific circuit configuration. Used by distributed workers."""
        results = []
        for _ in range(n_shots):
            results.append(self.single_strike(gamma, beta))
        return results


def classical_random_search(problem: MaxCutProblem, n_samples: int = 500,
                            seed: int | None = None) -> CycleLog:
    """Pure classical random sampling. No quantum anything."""
    rng = np.random.default_rng(seed)
    log = CycleLog()
    for _ in range(n_samples):
        bitstring = rng.integers(0, 2 ** problem.n_nodes)
        cut_val = problem.cut_value(bitstring)
        result = StrikeResult(
            bitstring=bitstring,
            cut_value=cut_val,
            sigma_before=1.0,
            sigma_after=1.0,
        )
        log.record(result)
    return log


if __name__ == "__main__":
    problem = MaxCutProblem.random(n_nodes=10, edge_prob=0.5, seed=42)
    optimal_bs, optimal_cut = problem.brute_force_optimal()
    print(f"Problem: {len(problem.edges)} edges, optimal cut = {optimal_cut:.2f}")

    for backend in ["statevector", "tensor"]:
        engine = MatchStrikeEngine(problem, n_layers=3, backend=backend, seed=42)
        log = engine.run(n_strikes=300, strategy="greedy")
        print(f"  {backend:>12s} greedy: best = {log.best_cut:.2f} "
              f"({log.best_cut / optimal_cut * 100:.1f}%)")

    classical_log = classical_random_search(problem, n_samples=300, seed=42)
    print(f"  {'classical':>12s} random: best = {classical_log.best_cut:.2f} "
          f"({classical_log.best_cut / optimal_cut * 100:.1f}%)")
