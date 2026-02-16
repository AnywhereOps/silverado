"""
DAG Consensus — Directed Acyclic Graph for quantum work ordering.

Instead of blocks in a chain, each quantum work unit is its own micro-
transaction that references 2-3 previous transactions. No mining, no
block intervals. Work units flow continuously.

Confirmation emerges from accumulation: a work unit is confirmed when
enough subsequent units reference it (directly or transitively).

Similar to IOTA's Tangle but purpose-built for quantum work verification.
"""

import time
import hashlib
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class WorkUnitStatus(Enum):
    PENDING = "pending"
    SOFT_CONFIRMED = "soft_confirmed"      # Referenced by enough units
    HARD_CONFIRMED = "hard_confirmed"      # Past challenge window
    REJECTED = "rejected"                   # Failed verification


@dataclass
class WorkUnit:
    """A single quantum work submission in the DAG."""
    unit_id: str
    solver_id: str
    problem_id: str
    merkle_root: bytes
    best_energy: float
    total_samples: int
    parent_ids: list[str]           # References to 2-3 previous units
    timestamp: float
    status: WorkUnitStatus = WorkUnitStatus.PENDING
    verification_votes: dict = field(default_factory=dict)  # verifier_id → bool
    confirmation_depth: int = 0
    cumulative_samples: int = 0     # Samples in this unit + all ancestors

    @property
    def unit_hash(self) -> str:
        data = (f"{self.solver_id}{self.problem_id}"
                f"{self.merkle_root.hex()}{self.best_energy}"
                f"{''.join(self.parent_ids)}{self.timestamp}")
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class DAGStats:
    """Statistics about the current DAG state."""
    total_units: int = 0
    pending: int = 0
    soft_confirmed: int = 0
    hard_confirmed: int = 0
    rejected: int = 0
    tips: int = 0                   # Units with no children
    avg_confirmation_depth: float = 0.0
    throughput_per_sec: float = 0.0
    total_samples: int = 0


class QuantumDAG:
    """Directed Acyclic Graph for ordering quantum work units.

    Rules:
    1. Each new unit references 2-3 existing tip units (no children)
    2. Confirmation depth = number of units that transitively reference this one
    3. Soft confirmation at depth >= soft_threshold
    4. Hard confirmation at depth >= hard_threshold AND age >= challenge_window
    5. Tips are selected by weighted random (prefer higher-energy, more-verified)
    """

    def __init__(self, soft_threshold: int = 5, hard_threshold: int = 20,
                 challenge_window_sec: float = 300.0,
                 n_parents: int = 2):
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.challenge_window = challenge_window_sec
        self.n_parents = n_parents

        # DAG storage
        self.units: dict[str, WorkUnit] = {}
        self.children: dict[str, set[str]] = {}   # unit_id → set of child unit_ids
        self.tips: set[str] = set()                # Units with no children
        self._rng = np.random.default_rng()

        # Genesis unit
        genesis = WorkUnit(
            unit_id="genesis",
            solver_id="system",
            problem_id="",
            merkle_root=b'\x00' * 32,
            best_energy=0.0,
            total_samples=0,
            parent_ids=[],
            timestamp=time.time(),
            status=WorkUnitStatus.HARD_CONFIRMED,
            confirmation_depth=999,
        )
        self.units["genesis"] = genesis
        self.children["genesis"] = set()
        self.tips.add("genesis")

    def select_tips(self, n: int | None = None) -> list[str]:
        """Select tip units to reference as parents.

        Weighted by:
        - Higher verification votes
        - Higher energy solutions
        - More recent timestamps
        """
        n = n or self.n_parents
        available = list(self.tips)
        if len(available) <= n:
            return available

        # Weight by recency and energy
        weights = []
        now = time.time()
        for uid in available:
            unit = self.units[uid]
            recency = max(0.1, 1.0 / (1 + (now - unit.timestamp)))
            energy_weight = max(0.1, unit.best_energy)
            weights.append(recency * energy_weight)

        weights = np.array(weights)
        weights /= weights.sum()

        selected = self._rng.choice(len(available), size=min(n, len(available)),
                                    replace=False, p=weights)
        return [available[i] for i in selected]

    def add_unit(self, solver_id: str, problem_id: str,
                 merkle_root: bytes, best_energy: float,
                 total_samples: int, parent_ids: list[str] | None = None) -> WorkUnit:
        """Add a new work unit to the DAG."""
        # Select parents if not specified
        if parent_ids is None:
            parent_ids = self.select_tips()

        # Validate parents exist
        parent_ids = [pid for pid in parent_ids if pid in self.units]
        if not parent_ids:
            parent_ids = ["genesis"]

        # Compute cumulative samples
        max_parent_cumulative = max(
            self.units[pid].cumulative_samples for pid in parent_ids
        )

        unit_id = hashlib.sha256(
            f"{solver_id}{time.time_ns()}{self._rng.integers(0, 2**32)}".encode()
        ).hexdigest()[:16]

        unit = WorkUnit(
            unit_id=unit_id,
            solver_id=solver_id,
            problem_id=problem_id,
            merkle_root=merkle_root,
            best_energy=best_energy,
            total_samples=total_samples,
            parent_ids=parent_ids,
            timestamp=time.time(),
            cumulative_samples=max_parent_cumulative + total_samples,
        )

        self.units[unit_id] = unit
        self.children[unit_id] = set()

        # Update parent → child links
        for pid in parent_ids:
            self.children[pid].add(unit_id)
            # Parent is no longer a tip
            self.tips.discard(pid)

        # New unit is a tip
        self.tips.add(unit_id)

        # Update confirmation depths for ancestors
        self._update_confirmations(unit_id)

        return unit

    def _update_confirmations(self, new_unit_id: str):
        """Update confirmation depths for all ancestors of a new unit."""
        # BFS backwards through parents
        visited = set()
        queue = [new_unit_id]

        while queue:
            uid = queue.pop(0)
            if uid in visited:
                continue
            visited.add(uid)

            unit = self.units[uid]
            # Count all descendants (children, grandchildren, etc.)
            depth = self._count_descendants(uid)
            unit.confirmation_depth = depth

            # Check for status transitions
            now = time.time()
            age = now - unit.timestamp

            if unit.status == WorkUnitStatus.PENDING:
                if depth >= self.soft_threshold:
                    unit.status = WorkUnitStatus.SOFT_CONFIRMED
                if depth >= self.hard_threshold and age >= self.challenge_window:
                    unit.status = WorkUnitStatus.HARD_CONFIRMED

            elif unit.status == WorkUnitStatus.SOFT_CONFIRMED:
                if depth >= self.hard_threshold and age >= self.challenge_window:
                    unit.status = WorkUnitStatus.HARD_CONFIRMED

            # Continue to parents
            for pid in unit.parent_ids:
                if pid not in visited:
                    queue.append(pid)

    def _count_descendants(self, unit_id: str) -> int:
        """Count all descendants of a unit (BFS)."""
        visited = set()
        queue = list(self.children.get(unit_id, set()))

        while queue:
            uid = queue.pop(0)
            if uid in visited:
                continue
            visited.add(uid)
            queue.extend(self.children.get(uid, set()))

        return len(visited)

    def add_verification(self, unit_id: str, verifier_id: str, approved: bool):
        """Add a verification vote to a work unit."""
        if unit_id in self.units:
            self.units[unit_id].verification_votes[verifier_id] = approved

    def reject_unit(self, unit_id: str):
        """Mark a unit as rejected (fraud detected)."""
        if unit_id in self.units:
            self.units[unit_id].status = WorkUnitStatus.REJECTED

    def get_best_solution(self, problem_id: str) -> tuple[str, float] | None:
        """Get the best confirmed solution for a problem."""
        best_energy = -float('inf')
        best_unit_id = None

        for uid, unit in self.units.items():
            if (unit.problem_id == problem_id and
                unit.status in (WorkUnitStatus.SOFT_CONFIRMED, WorkUnitStatus.HARD_CONFIRMED) and
                unit.best_energy > best_energy):
                best_energy = unit.best_energy
                best_unit_id = uid

        if best_unit_id:
            return best_unit_id, best_energy
        return None

    def stats(self) -> DAGStats:
        """Compute DAG statistics."""
        status_counts = {s: 0 for s in WorkUnitStatus}
        total_samples = 0
        depths = []

        for unit in self.units.values():
            if unit.unit_id == "genesis":
                continue
            status_counts[unit.status] += 1
            total_samples += unit.total_samples
            depths.append(unit.confirmation_depth)

        first_time = min(u.timestamp for u in self.units.values())
        last_time = max(u.timestamp for u in self.units.values())
        elapsed = max(last_time - first_time, 0.001)

        return DAGStats(
            total_units=len(self.units) - 1,  # Exclude genesis
            pending=status_counts[WorkUnitStatus.PENDING],
            soft_confirmed=status_counts[WorkUnitStatus.SOFT_CONFIRMED],
            hard_confirmed=status_counts[WorkUnitStatus.HARD_CONFIRMED],
            rejected=status_counts[WorkUnitStatus.REJECTED],
            tips=len(self.tips),
            avg_confirmation_depth=float(np.mean(depths)) if depths else 0,
            throughput_per_sec=(len(self.units) - 1) / elapsed,
            total_samples=total_samples,
        )


def simulate_dag(n_solvers: int = 10, n_rounds: int = 100):
    """Simulate DAG building with multiple concurrent solvers."""
    print("DAG Consensus Simulation")
    print("=" * 60)

    # Use very short challenge window for simulation
    dag = QuantumDAG(soft_threshold=3, hard_threshold=10, challenge_window_sec=0.0)
    rng = np.random.default_rng(42)

    print(f"\nSimulating {n_solvers} solvers, {n_rounds} rounds...")

    for round_num in range(n_rounds):
        # Each solver submits a work unit
        for s in range(n_solvers):
            solver_id = f"solver-{s}"
            energy = float(rng.uniform(10, 35))
            samples = int(rng.integers(100, 1000))
            merkle_root = hashlib.sha256(
                f"{solver_id}{round_num}{rng.integers(0, 2**32)}".encode()
            ).digest()

            dag.add_unit(
                solver_id=solver_id,
                problem_id="maxcut-42",
                merkle_root=merkle_root,
                best_energy=energy,
                total_samples=samples,
            )

    s = dag.stats()
    print(f"\n  Total units: {s.total_units}")
    print(f"  Pending: {s.pending}")
    print(f"  Soft confirmed: {s.soft_confirmed}")
    print(f"  Hard confirmed: {s.hard_confirmed}")
    print(f"  Tips: {s.tips}")
    print(f"  Avg confirmation depth: {s.avg_confirmation_depth:.1f}")
    print(f"  Total samples: {s.total_samples:,}")
    print(f"  Throughput: {s.throughput_per_sec:.0f} units/sec")

    best = dag.get_best_solution("maxcut-42")
    if best:
        uid, energy = best
        print(f"\n  Best solution: energy={energy:.2f} (unit {uid})")


if __name__ == "__main__":
    simulate_dag(n_solvers=10, n_rounds=100)
