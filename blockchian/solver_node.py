"""
Solver Node — Compute worker for the distributed quantum carburetor.

Receives circuit specifications from the coordinator, runs quantum simulation
(state vector or tensor network), returns measurement samples.
Reports hardware capabilities and tracks its own performance.

In production this would be a network daemon. For now it's an in-process
worker that the coordinator calls directly.
"""

import time
import hashlib
import numpy as np
from dataclasses import dataclass, field
from match_strike import MatchStrikeEngine, MaxCutProblem, StrikeResult


@dataclass
class HardwareCapabilities:
    """What this node can do."""
    node_id: str
    max_qubits_statevector: int = 20  # Limited by RAM
    max_qubits_tensor: int = 50       # MPS with truncation
    max_bond_dimension: int = 64
    has_gpu: bool = False
    ram_gb: float = 8.0
    cores: int = 4
    estimated_strikes_per_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "max_qubits_statevector": self.max_qubits_statevector,
            "max_qubits_tensor": self.max_qubits_tensor,
            "max_bond_dimension": self.max_bond_dimension,
            "has_gpu": self.has_gpu,
            "ram_gb": self.ram_gb,
            "cores": self.cores,
            "estimated_strikes_per_sec": self.estimated_strikes_per_sec,
        }


@dataclass
class TaskSpec:
    """A unit of work from the coordinator."""
    task_id: str
    problem: MaxCutProblem
    gamma: float
    beta: float
    n_layers: int
    n_shots: int
    backend: str = "statevector"
    max_bond: int = 64


@dataclass
class TaskResult:
    """What the solver sends back."""
    task_id: str
    node_id: str
    bitstrings: list[int]
    cut_values: list[float]
    sigma_values: list[float]
    wall_time_ms: float
    # Proof of work: hash of sorted results for verification
    result_hash: str = ""

    def compute_hash(self) -> str:
        """Deterministic hash of results for verification."""
        data = str(sorted(self.bitstrings)) + str(sorted(self.cut_values))
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class SolverNode:
    """A compute worker in the distributed network."""

    def __init__(self, node_id: str | None = None, capabilities: HardwareCapabilities | None = None):
        self.node_id = node_id or hashlib.sha256(
            str(time.time_ns()).encode()
        ).hexdigest()[:12]
        self.capabilities = capabilities or HardwareCapabilities(node_id=self.node_id)
        self.capabilities.node_id = self.node_id
        self.tasks_completed = 0
        self.total_strikes = 0
        self.total_time_ms = 0.0
        self._calibrated = False

    def calibrate(self, n_qubits: int = 10, n_shots: int = 100) -> float:
        """Benchmark this node's speed. Returns strikes/sec."""
        problem = MaxCutProblem.random(n_nodes=n_qubits, seed=0)
        engine = MatchStrikeEngine(problem, n_layers=3, backend="statevector")

        t0 = time.perf_counter()
        for _ in range(n_shots):
            engine.single_strike(np.pi, np.pi / 2)
        t1 = time.perf_counter()

        strikes_per_sec = n_shots / (t1 - t0)
        self.capabilities.estimated_strikes_per_sec = strikes_per_sec
        self._calibrated = True
        return strikes_per_sec

    def execute(self, task: TaskSpec) -> TaskResult:
        """Execute a task and return results."""
        engine = MatchStrikeEngine(
            task.problem,
            n_layers=task.n_layers,
            backend=task.backend,
            max_bond=task.max_bond,
        )

        bitstrings = []
        cut_values = []
        sigma_values = []

        t0 = time.perf_counter()
        for _ in range(task.n_shots):
            result = engine.single_strike(task.gamma, task.beta)
            bitstrings.append(result.bitstring)
            cut_values.append(result.cut_value)
            sigma_values.append(result.sigma_before)
        t1 = time.perf_counter()

        wall_time_ms = (t1 - t0) * 1000

        task_result = TaskResult(
            task_id=task.task_id,
            node_id=self.node_id,
            bitstrings=bitstrings,
            cut_values=cut_values,
            sigma_values=sigma_values,
            wall_time_ms=wall_time_ms,
        )
        task_result.result_hash = task_result.compute_hash()

        self.tasks_completed += 1
        self.total_strikes += task.n_shots
        self.total_time_ms += wall_time_ms

        return task_result

    def can_handle(self, task: TaskSpec) -> bool:
        """Check if this node can handle the given task."""
        n = task.problem.n_nodes
        if task.backend == "statevector":
            return n <= self.capabilities.max_qubits_statevector
        elif task.backend == "tensor":
            return (n <= self.capabilities.max_qubits_tensor and
                    task.max_bond <= self.capabilities.max_bond_dimension)
        return False

    @property
    def avg_strike_time_ms(self) -> float:
        if self.total_strikes == 0:
            return 0.0
        return self.total_time_ms / self.total_strikes

    def __repr__(self):
        return (f"SolverNode(id={self.node_id}, "
                f"tasks={self.tasks_completed}, "
                f"strikes={self.total_strikes}, "
                f"avg={self.avg_strike_time_ms:.2f}ms/strike)")


if __name__ == "__main__":
    # Demo: create a node, calibrate, execute a task
    node = SolverNode()
    speed = node.calibrate()
    print(f"Node {node.node_id}: {speed:.0f} strikes/sec")

    problem = MaxCutProblem.random(n_nodes=12, seed=42)
    task = TaskSpec(
        task_id="test-001",
        problem=problem,
        gamma=np.pi * 0.7,
        beta=np.pi * 0.3,
        n_layers=3,
        n_shots=100,
    )
    result = node.execute(task)
    print(f"Task {result.task_id}: {len(result.bitstrings)} shots in {result.wall_time_ms:.1f}ms")
    print(f"  Best cut: {max(result.cut_values):.2f}")
    print(f"  Hash: {result.result_hash}")
    print(f"  Node stats: {node}")
