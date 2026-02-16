"""
Commit — Solver-side batch commitment scheme.

The solver runs match-strike cycles, accumulates samples in a buffer,
and periodically commits a Merkle root covering the batch. This is the
bridge between Speed 1 (microsecond compute) and Speed 2 (millisecond
peer verification).
"""

import time
import hashlib
import numpy as np
from dataclasses import dataclass, field
from merkle import MerkleTree, QuantumSample, MerkleProof
from match_strike import MatchStrikeEngine, MaxCutProblem


@dataclass
class BatchCommitment:
    """A committed batch of quantum samples."""
    batch_id: str
    solver_id: str
    problem_hash: str          # Hash of the problem being solved
    merkle_root: bytes         # 32-byte root hash
    n_samples: int
    best_bitstring: int
    best_energy: float
    mean_energy: float
    timestamp: float
    circuit_params: list[tuple[float, float]]  # (gamma, beta) pairs used

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "solver_id": self.solver_id,
            "problem_hash": self.problem_hash,
            "merkle_root": self.merkle_root.hex(),
            "n_samples": self.n_samples,
            "best_bitstring": self.best_bitstring,
            "best_energy": self.best_energy,
            "mean_energy": self.mean_energy,
            "timestamp": self.timestamp,
            "circuit_params": self.circuit_params,
        }


@dataclass
class CommitmentResponse:
    """Response from a verifier to a batch commitment."""
    batch_id: str
    verifier_id: str
    accepted: bool
    confidence: float
    checked_leaves: int
    reason: str = ""


class SolverCommitter:
    """Manages the solver's commitment pipeline.

    Flow:
    1. Run match-strike cycles, buffer samples
    2. When buffer reaches threshold, build Merkle tree
    3. Commit root hash
    4. Respond to verifier spot-check requests with Merkle proofs
    """

    def __init__(self, solver_id: str, problem: MaxCutProblem,
                 n_layers: int = 3, batch_size: int = 1000,
                 backend: str = "statevector"):
        self.solver_id = solver_id
        self.problem = problem
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.engine = MatchStrikeEngine(problem, n_layers=n_layers, backend=backend)
        self.rng = np.random.default_rng()

        # State
        self.sample_buffer: list[QuantumSample] = []
        self.committed_batches: list[BatchCommitment] = []
        self.trees: dict[str, MerkleTree] = {}  # batch_id → tree
        self.samples_by_batch: dict[str, list[QuantumSample]] = {}
        self._batch_counter = 0
        self._strike_counter = 0

        # Problem hash for commitment
        self.problem_hash = hashlib.sha256(
            str(problem.to_dict()).encode()
        ).hexdigest()[:16]

    def run_strikes(self, n_strikes: int,
                    gamma: float | None = None,
                    beta: float | None = None) -> list[QuantumSample]:
        """Run match-strike cycles and buffer the samples."""
        samples = []
        for _ in range(n_strikes):
            g = gamma if gamma is not None else float(self.rng.uniform(0, 2 * np.pi))
            b = beta if beta is not None else float(self.rng.uniform(0, np.pi))
            seed = int(self.rng.integers(0, 2**32))

            # Set VM seed for reproducibility
            self.engine.vm._rng = np.random.default_rng(seed)
            result = self.engine.single_strike(g, b)

            sample = QuantumSample(
                strike_index=self._strike_counter,
                gamma=g,
                beta=b,
                n_layers=self.n_layers,
                bitstring=result.bitstring,
                cut_value=result.cut_value,
                random_seed=seed,
                timestamp=time.time(),
            )
            samples.append(sample)
            self.sample_buffer.append(sample)
            self._strike_counter += 1

        return samples

    def commit_batch(self) -> BatchCommitment | None:
        """Commit the current buffer as a Merkle tree. Returns None if buffer empty."""
        if not self.sample_buffer:
            return None

        # Build Merkle tree
        tree = MerkleTree(self.sample_buffer)

        # Create commitment
        self._batch_counter += 1
        batch_id = f"{self.solver_id}-batch-{self._batch_counter:06d}"

        cut_values = [s.cut_value for s in self.sample_buffer]
        best_idx = int(np.argmax(cut_values))
        params_used = list(set((s.gamma, s.beta) for s in self.sample_buffer))

        commitment = BatchCommitment(
            batch_id=batch_id,
            solver_id=self.solver_id,
            problem_hash=self.problem_hash,
            merkle_root=tree.root,
            n_samples=len(self.sample_buffer),
            best_bitstring=self.sample_buffer[best_idx].bitstring,
            best_energy=max(cut_values),
            mean_energy=float(np.mean(cut_values)),
            timestamp=time.time(),
            circuit_params=params_used[:20],  # Cap for serialization
        )

        # Store
        self.committed_batches.append(commitment)
        self.trees[batch_id] = tree
        self.samples_by_batch[batch_id] = self.sample_buffer[:]

        # Clear buffer
        self.sample_buffer = []

        return commitment

    def respond_to_challenge(self, batch_id: str,
                             leaf_indices: list[int]) -> list[tuple[QuantumSample, MerkleProof]]:
        """Respond to a verifier's spot-check request.

        Returns the requested samples along with their Merkle proofs.
        """
        if batch_id not in self.trees:
            raise KeyError(f"Unknown batch: {batch_id}")

        tree = self.trees[batch_id]
        samples = self.samples_by_batch[batch_id]
        responses = []

        for idx in leaf_indices:
            if idx < len(samples):
                proof = tree.get_proof(idx)
                responses.append((samples[idx], proof))

        return responses

    def run_and_commit(self, n_strikes: int,
                       gamma: float | None = None,
                       beta: float | None = None) -> BatchCommitment:
        """Convenience: run strikes and immediately commit."""
        self.run_strikes(n_strikes, gamma, beta)
        return self.commit_batch()

    @property
    def total_committed_samples(self) -> int:
        return sum(c.n_samples for c in self.committed_batches)

    @property
    def best_energy_found(self) -> float:
        if not self.committed_batches:
            return 0.0
        return max(c.best_energy for c in self.committed_batches)


def demo_commitment_flow():
    """Demonstrate the full commitment pipeline."""
    print("Solver Commitment Pipeline Demo")
    print("=" * 60)

    problem = MaxCutProblem.random(n_nodes=12, edge_prob=0.5, seed=42)
    _, optimal_cut = problem.brute_force_optimal()
    print(f"Problem: {problem.n_nodes} nodes, {len(problem.edges)} edges, "
          f"optimal = {optimal_cut:.2f}")

    solver = SolverCommitter(
        solver_id="solver-alpha",
        problem=problem,
        batch_size=500,
    )

    # Run 5 batches
    for batch_num in range(5):
        gamma = float(np.random.uniform(0.5, 5.0))
        beta = float(np.random.uniform(0.2, 2.5))

        t0 = time.perf_counter()
        solver.run_strikes(500, gamma=gamma, beta=beta)
        strike_time = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        commitment = solver.commit_batch()
        commit_time = (time.perf_counter() - t0) * 1000

        # Simulate verifier challenge
        t0 = time.perf_counter()
        challenge_indices = list(np.random.choice(500, size=50, replace=False))
        responses = solver.respond_to_challenge(commitment.batch_id, challenge_indices)
        challenge_time = (time.perf_counter() - t0) * 1000

        # Verify proofs
        tree = solver.trees[commitment.batch_id]
        all_valid = all(tree.verify_proof(proof) for _, proof in responses)

        print(f"\n  Batch {batch_num + 1}:")
        print(f"    500 strikes: {strike_time:.1f}ms")
        print(f"    Merkle commit: {commit_time:.1f}ms")
        print(f"    50-leaf challenge: {challenge_time:.1f}ms")
        print(f"    Proofs valid: {all_valid}")
        print(f"    Best energy: {commitment.best_energy:.2f} "
              f"({commitment.best_energy / optimal_cut:.1%} of optimal)")
        print(f"    Root: {commitment.merkle_root.hex()[:24]}...")

    print(f"\n  Total committed: {solver.total_committed_samples} samples")
    print(f"  Best found: {solver.best_energy_found:.2f} "
          f"({solver.best_energy_found / optimal_cut:.1%} of optimal)")


if __name__ == "__main__":
    demo_commitment_flow()
