"""
Coordinator — Task distribution and sample aggregation for distributed quantum carburetor.

The brain of the distributed system. Assigns match-strike tasks to solver nodes,
aggregates measurement samples, runs the adaptive controller to decide the next query.

Handles two parallelism strategies:
  A. Sample parallelism: same circuit → many nodes → aggregate samples
  B. (Future) Tensor sharding: decompose large circuits across nodes

No blockchain yet. Just coordination.
"""

import time
import hashlib
import numpy as np
from dataclasses import dataclass, field
from core.match_strike import MaxCutProblem, StrikeResult, CycleLog
from network.solver_node import SolverNode, TaskSpec, TaskResult, HardwareCapabilities
from core.adaptive_controller import AdaptiveController, MultiScaleController, MatchStrikeEngine


@dataclass
class AggregatedResult:
    """Results from multiple nodes for the same circuit parameters."""
    gamma: float
    beta: float
    all_bitstrings: list[int] = field(default_factory=list)
    all_cut_values: list[float] = field(default_factory=list)
    node_results: list[TaskResult] = field(default_factory=list)

    @property
    def best_cut(self) -> float:
        return max(self.all_cut_values) if self.all_cut_values else 0.0

    @property
    def best_bitstring(self) -> int:
        if not self.all_cut_values:
            return 0
        idx = int(np.argmax(self.all_cut_values))
        return self.all_bitstrings[idx]

    @property
    def mean_cut(self) -> float:
        return float(np.mean(self.all_cut_values)) if self.all_cut_values else 0.0

    @property
    def total_samples(self) -> int:
        return len(self.all_bitstrings)

    @property
    def total_time_ms(self) -> float:
        return max((r.wall_time_ms for r in self.node_results), default=0)


class Coordinator:
    """Distributes quantum simulation tasks across solver nodes.

    Orchestrates the match-strike cycle at scale:
    1. Decide what to ask (adaptive controller)
    2. Fan out the computation to nodes
    3. Collect and aggregate results
    4. Feed back into the controller
    5. Repeat
    """

    def __init__(self, problem: MaxCutProblem, n_layers: int = 3):
        self.problem = problem
        self.n_layers = n_layers
        self.nodes: list[SolverNode] = []
        self.results: list[AggregatedResult] = []
        self.log = CycleLog()
        self.round_count = 0
        self._task_counter = 0

    def register_node(self, node: SolverNode):
        """Add a solver node to the pool."""
        self.nodes.append(node)

    def register_nodes(self, n: int) -> list[SolverNode]:
        """Create and register n solver nodes."""
        nodes = []
        for _ in range(n):
            node = SolverNode()
            self.register_node(node)
            nodes.append(node)
        return nodes

    def _next_task_id(self) -> str:
        self._task_counter += 1
        return f"task-{self._task_counter:06d}"

    def _select_backend(self, node: SolverNode) -> str:
        """Choose the best backend for this node and problem."""
        if self.problem.n_nodes <= node.capabilities.max_qubits_statevector:
            return "statevector"
        elif self.problem.n_nodes <= node.capabilities.max_qubits_tensor:
            return "tensor"
        else:
            raise ValueError(
                f"Problem ({self.problem.n_nodes} qubits) too large for node {node.node_id}"
            )

    def distribute_round(self, gamma: float, beta: float,
                         shots_per_node: int = 100) -> AggregatedResult:
        """Fan out one circuit configuration to all available nodes.

        This is Strategy A: Sample Parallelism.
        Every node runs the same circuit independently.
        """
        agg = AggregatedResult(gamma=gamma, beta=beta)
        available_nodes = [n for n in self.nodes
                          if n.can_handle(TaskSpec(
                              task_id="", problem=self.problem,
                              gamma=gamma, beta=beta,
                              n_layers=self.n_layers, n_shots=1))]

        if not available_nodes:
            raise RuntimeError("No nodes available for this problem size")

        for node in available_nodes:
            backend = self._select_backend(node)
            task = TaskSpec(
                task_id=self._next_task_id(),
                problem=self.problem,
                gamma=gamma,
                beta=beta,
                n_layers=self.n_layers,
                n_shots=shots_per_node,
                backend=backend,
            )
            result = node.execute(task)
            agg.all_bitstrings.extend(result.bitstrings)
            agg.all_cut_values.extend(result.cut_values)
            agg.node_results.append(result)

        self.results.append(agg)
        self.round_count += 1

        # Record all results in the cycle log
        for bs, cv in zip(agg.all_bitstrings, agg.all_cut_values):
            self.log.record(StrikeResult(
                bitstring=bs, cut_value=cv,
                sigma_before=0.5, sigma_after=1.0,
                gamma=gamma, beta=beta,
            ))

        return agg

    def run_adaptive(self, n_rounds: int = 50, shots_per_node: int = 100,
                     grid_size: int = 15) -> CycleLog:
        """Run the full adaptive distributed cycle.

        Each round:
        1. Controller picks (γ, β)
        2. All nodes run that circuit
        3. Aggregate results update the controller
        """
        # Create a dummy engine for the controller (it won't actually execute)
        engine = MatchStrikeEngine(self.problem, n_layers=self.n_layers)
        controller = AdaptiveController(engine, grid_size=grid_size)

        # Warmup: random exploration
        n_warmup = max(n_rounds // 5, grid_size // 2)

        gammas = np.linspace(0.1, 2 * np.pi, grid_size)
        betas = np.linspace(0.1, np.pi, grid_size)
        rng = np.random.default_rng()

        for r in range(n_rounds):
            if r < n_warmup:
                # Random
                gi = int(rng.integers(grid_size))
                bi = int(rng.integers(grid_size))
                gamma, beta = gammas[gi], betas[bi]
            else:
                # UCB selection
                gi, bi = controller._select_next_point()
                point = controller.grid[(gi, bi)]
                gamma, beta = point.gamma, point.beta

            # Distribute to all nodes
            agg = self.distribute_round(gamma, beta, shots_per_node=shots_per_node)

            # Update controller with results
            # Find the grid point closest to (gamma, beta)
            gi = int(np.argmin(np.abs(gammas - gamma)))
            bi = int(np.argmin(np.abs(betas - beta)))
            if (gi, bi) in controller.grid:
                controller.grid[(gi, bi)].cut_values.extend(agg.all_cut_values)

        return self.log

    def run_random(self, n_rounds: int = 50, shots_per_node: int = 100) -> CycleLog:
        """Run with random parameter selection (baseline)."""
        rng = np.random.default_rng()
        for _ in range(n_rounds):
            gamma = rng.uniform(0.1, 2 * np.pi)
            beta = rng.uniform(0.1, np.pi)
            self.distribute_round(gamma, beta, shots_per_node=shots_per_node)
        return self.log

    def stats(self) -> dict:
        """Summary statistics of the distributed run."""
        total_samples = sum(len(r.all_bitstrings) for r in self.results)
        total_time = sum(r.total_time_ms for r in self.results)
        return {
            "n_nodes": len(self.nodes),
            "n_rounds": self.round_count,
            "total_samples": total_samples,
            "total_time_ms": total_time,
            "best_cut": self.log.best_cut,
            "best_bitstring": self.log.best_bitstring,
            "samples_per_second": total_samples / (total_time / 1000) if total_time > 0 else 0,
            "node_stats": {
                n.node_id: {
                    "tasks": n.tasks_completed,
                    "strikes": n.total_strikes,
                    "avg_ms": n.avg_strike_time_ms,
                }
                for n in self.nodes
            },
        }


def benchmark_distributed(n_nodes_list: list[int] | None = None,
                          n_qubits: int = 12, n_rounds: int = 30,
                          shots_per_node: int = 50):
    """Benchmark distributed performance scaling."""
    if n_nodes_list is None:
        n_nodes_list = [1, 2, 4, 8, 16]

    problem = MaxCutProblem.random(n_nodes=n_qubits, edge_prob=0.5, seed=42)
    optimal_bs, optimal_cut = problem.brute_force_optimal()
    print(f"Problem: {n_qubits} nodes, {len(problem.edges)} edges, optimal cut = {optimal_cut:.2f}")
    print()

    # Single-node baseline
    print(f"  {'Nodes':>5s} | {'Samples':>7s} | {'Best Cut':>8s} | {'Ratio':>6s} | "
          f"{'Time (ms)':>9s} | {'Samples/s':>9s} | {'Speedup':>7s}")
    print(f"  {'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}")

    baseline_time = None

    for n_nodes in n_nodes_list:
        coord = Coordinator(problem, n_layers=3)
        coord.register_nodes(n_nodes)
        log = coord.run_adaptive(n_rounds=n_rounds, shots_per_node=shots_per_node)
        stats = coord.stats()

        total_time = stats["total_time_ms"]
        if baseline_time is None:
            baseline_time = total_time
        speedup = baseline_time / total_time if total_time > 0 else 0

        print(f"  {n_nodes:>5d} | {stats['total_samples']:>7d} | "
              f"{stats['best_cut']:>8.2f} | {stats['best_cut']/optimal_cut:>5.1%} | "
              f"{total_time:>9.1f} | {stats['samples_per_second']:>9.0f} | "
              f"{speedup:>6.1f}x")


if __name__ == "__main__":
    benchmark_distributed(
        n_nodes_list=[1, 2, 4, 8],
        n_qubits=12,
        n_rounds=30,
        shots_per_node=50,
    )
