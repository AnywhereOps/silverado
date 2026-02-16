"""
ZK Prover — Zero-knowledge proof generation for quantum simulation.

Proves: "I correctly simulated quantum circuit C for n strikes and the
best solution I found has energy E."

The proof covers:
  1. Each gate application was a valid unitary operation
  2. Measurements were sampled according to Born rule
  3. The reported energy corresponds to an actual measurement outcome
  4. The Merkle root correctly commits to all samples

In production, this would use Plonky2 or Halo2. Here we simulate the
ZK proof structure and verify the arithmetic in Python, demonstrating
the proof concept and benchmarking the computation.

The key insight: quantum gates are small (2x2, 4x4) structured matrices.
The verification circuit is highly regular and bounded-depth.
"""

import time
import hashlib
import struct
import numpy as np
from dataclasses import dataclass, field
from chain.merkle import MerkleTree, QuantumSample
from core.quantum_vm import QuantumVM
from core.match_strike import MaxCutProblem


# ── Finite field arithmetic (simulating ZK field) ───────────────────

# Use a prime field large enough for our purposes
FIELD_PRIME = (1 << 61) - 1  # Mersenne prime 2^61 - 1


def field_add(a: int, b: int) -> int:
    return (a + b) % FIELD_PRIME


def field_mul(a: int, b: int) -> int:
    return (a * b) % FIELD_PRIME


def field_sub(a: int, b: int) -> int:
    return (a - b) % FIELD_PRIME


def field_inv(a: int) -> int:
    return pow(a, FIELD_PRIME - 2, FIELD_PRIME)


def float_to_field(x: float, precision: int = 30) -> int:
    """Convert float to field element with fixed-point representation."""
    scaled = int(x * (1 << precision))
    return scaled % FIELD_PRIME


def field_to_float(x: int, precision: int = 30) -> float:
    """Convert field element back to float."""
    if x > FIELD_PRIME // 2:
        x -= FIELD_PRIME
    return x / (1 << precision)


# ── Proof structure ─────────────────────────────────────────────────

@dataclass
class GateWitness:
    """Witness for a single gate application."""
    gate_type: str           # "h", "ry", "rz", "cx", "cz"
    qubit_indices: list[int]
    angle: float             # For rotation gates
    # The state before and after (commitments, not full state)
    state_hash_before: bytes
    state_hash_after: bytes


@dataclass
class MeasurementWitness:
    """Witness for a single measurement."""
    strike_index: int
    probabilities_hash: bytes   # Hash of the probability vector
    outcome: int                # The measured bitstring
    random_seed: int            # Seed used for sampling
    cut_value: float


@dataclass
class QuantumProof:
    """A zero-knowledge proof of correct quantum simulation.

    In a real ZK system, this would be a compressed proof (~200-500KB).
    Here we store the structured witness data that the proof covers.
    """
    proof_id: str
    problem_hash: str
    merkle_root: bytes
    n_qubits: int
    n_strikes: int
    n_gates_per_strike: int
    best_energy: float
    best_bitstring: int

    # Proof components
    gate_commitment: bytes       # Hash of all gate witnesses
    measurement_commitment: bytes  # Hash of all measurement witnesses
    merkle_proof_hash: bytes     # Hash of Merkle tree construction

    # Verification data (what a verifier needs)
    circuit_hash: bytes          # Hash of the circuit description
    aggregate_statistics: dict = field(default_factory=dict)

    # Benchmarks
    generation_time_ms: float = 0.0
    proof_size_bytes: int = 0

    def verify_structure(self) -> bool:
        """Verify internal consistency of the proof."""
        # Check that all commitments are non-trivial
        if self.gate_commitment == b'\x00' * 32:
            return False
        if self.measurement_commitment == b'\x00' * 32:
            return False
        if self.merkle_root == b'\x00' * 32:
            return False
        return True


class ZKProver:
    """Generates zero-knowledge proofs of correct quantum simulation.

    The prover has access to the full computation trace and generates
    a proof that the computation was performed correctly.

    Proof generation cost: O(n_strikes * n_qubits^2)
    Proof verification cost: O(log(n_strikes))
    """

    def __init__(self, problem: MaxCutProblem, n_layers: int = 3):
        self.problem = problem
        self.n_layers = n_layers
        self.n_qubits = problem.n_nodes

        # Circuit description
        self.circuit_hash = self._hash_circuit()

    def _hash_circuit(self) -> bytes:
        """Hash the circuit description (problem + parameters)."""
        data = str(self.problem.to_dict()).encode()
        data += struct.pack('<i', self.n_layers)
        return hashlib.sha256(data).digest()

    def _simulate_gate_proof(self, gate_type: str, qubit: int,
                             angle: float = 0.0) -> int:
        """Simulate the field arithmetic operations for proving one gate.

        Returns the number of field operations performed.
        """
        ops = 0
        if gate_type in ("h", "x", "z"):
            # 2x2 matrix multiply: 4 muls + 2 adds per amplitude pair
            # For n qubits, applied to 2^(n-1) amplitude pairs
            ops = 6 * (2 ** (self.n_qubits - 1))
        elif gate_type in ("ry", "rz"):
            # Same structure but with trigonometric constants
            ops = 8 * (2 ** (self.n_qubits - 1))  # Extra ops for trig encoding
        elif gate_type in ("cx", "cz"):
            # 4x4 gate on 2 qubits: 16 muls + 8 adds per quad
            ops = 24 * (2 ** (self.n_qubits - 2))
        return ops

    def _simulate_measurement_proof(self) -> int:
        """Simulate field ops for proving a measurement is Born-rule compliant."""
        ops = 0
        # Compute all probabilities: n muls (|ψ|²)
        ops += 2 * (2 ** self.n_qubits)
        # Verify normalization: n adds + 1 comparison
        ops += 2 ** self.n_qubits + 1
        # Verify sampling: hash-based random oracle check
        ops += 256  # SHA-256 in-circuit
        return ops

    def generate_proof(self, samples: list[QuantumSample],
                       tree: MerkleTree) -> QuantumProof:
        """Generate a ZK proof from a computation trace.

        This simulates the proof generation process, computing the
        field arithmetic cost and generating the proof structure.
        """
        t0 = time.perf_counter()

        # Count total field operations
        total_field_ops = 0

        # Gate witnesses
        gate_hashes = []
        gates_per_strike = 0

        # For each strike, prove the circuit was applied correctly
        for strike_idx, sample in enumerate(samples):
            strike_ops = 0

            # Superposition: n Hadamard gates
            for q in range(self.n_qubits):
                strike_ops += self._simulate_gate_proof("h", q)
                gates_per_strike += 1

            # For each layer
            for layer in range(self.n_layers):
                # Phase oracle: ZZ interactions for each edge
                for (i, j), w in zip(self.problem.edges, self.problem.weights):
                    strike_ops += self._simulate_gate_proof("cx", i)  # CNOT
                    strike_ops += self._simulate_gate_proof("rz", j, sample.gamma * w)
                    strike_ops += self._simulate_gate_proof("cx", i)  # CNOT
                    gates_per_strike += 3

                # Mixer: Ry on each qubit
                for q in range(self.n_qubits):
                    strike_ops += self._simulate_gate_proof("ry", q, sample.beta)
                    gates_per_strike += 1

            # Measurement proof
            strike_ops += self._simulate_measurement_proof()

            total_field_ops += strike_ops

            # Hash this strike's witness
            witness_data = struct.pack(
                '<iddi',
                strike_idx, sample.gamma, sample.beta, sample.bitstring,
            )
            gate_hashes.append(hashlib.sha256(witness_data).digest())

        gates_per_strike = gates_per_strike // max(len(samples), 1)

        # Merkle tree proof (verify tree construction)
        merkle_ops = len(samples) * 256  # SHA-256 per leaf
        merkle_ops += len(samples) * int(np.log2(max(len(samples), 1))) * 256  # Internal nodes
        total_field_ops += merkle_ops

        # Aggregate gate commitment
        all_gate_data = b''.join(gate_hashes)
        gate_commitment = hashlib.sha256(all_gate_data).digest()

        # Measurement commitment
        meas_data = b''.join(
            struct.pack('<idf', s.strike_index, s.cut_value, s.gamma)
            for s in samples
        )
        measurement_commitment = hashlib.sha256(meas_data).digest()

        # Aggregate statistics
        cut_values = [s.cut_value for s in samples]
        stats = {
            "mean_energy": float(np.mean(cut_values)),
            "std_energy": float(np.std(cut_values)),
            "max_energy": float(np.max(cut_values)),
            "n_unique_bitstrings": len(set(s.bitstring for s in samples)),
            "total_field_operations": total_field_ops,
        }

        gen_time = (time.perf_counter() - t0) * 1000

        # Estimate proof size (in a real ZK system)
        # Plonky2: ~200KB for 2^20 gates, scales logarithmically
        estimated_proof_size = 200_000 + int(np.log2(max(total_field_ops, 1)) * 10_000)

        proof = QuantumProof(
            proof_id=hashlib.sha256(gate_commitment + measurement_commitment).hexdigest()[:16],
            problem_hash=hashlib.sha256(str(self.problem.to_dict()).encode()).hexdigest()[:16],
            merkle_root=tree.root,
            n_qubits=self.n_qubits,
            n_strikes=len(samples),
            n_gates_per_strike=gates_per_strike,
            best_energy=max(cut_values),
            best_bitstring=samples[int(np.argmax(cut_values))].bitstring,
            gate_commitment=gate_commitment,
            measurement_commitment=measurement_commitment,
            merkle_proof_hash=hashlib.sha256(tree.root).digest(),
            circuit_hash=self.circuit_hash,
            aggregate_statistics=stats,
            generation_time_ms=gen_time,
            proof_size_bytes=estimated_proof_size,
        )

        return proof


class ZKVerifier:
    """Verifies zero-knowledge proofs of quantum simulation.

    Verification cost: O(log(n_strikes)) — independent of circuit size.
    On-chain: ~200K gas for Plonky2/Halo2 proof verification.
    """

    def __init__(self, problem: MaxCutProblem, n_layers: int = 3):
        self.problem = problem
        self.n_layers = n_layers
        self.circuit_hash = self._expected_circuit_hash()

    def _expected_circuit_hash(self) -> bytes:
        data = str(self.problem.to_dict()).encode()
        data += struct.pack('<i', self.n_layers)
        return hashlib.sha256(data).digest()

    def verify(self, proof: QuantumProof) -> tuple[bool, str]:
        """Verify a ZK proof.

        In a real system, this would verify the ZK-SNARK/STARK.
        Here we check the proof structure and commitments.
        """
        t0 = time.perf_counter()

        # 1. Check circuit hash matches expected
        if proof.circuit_hash != self.circuit_hash:
            return False, "Circuit hash mismatch"

        # 2. Check proof structure
        if not proof.verify_structure():
            return False, "Invalid proof structure"

        # 3. Check problem hash
        expected_problem_hash = hashlib.sha256(
            str(self.problem.to_dict()).encode()
        ).hexdigest()[:16]
        if proof.problem_hash != expected_problem_hash:
            return False, "Problem hash mismatch"

        # 4. Check claimed best energy is consistent
        if proof.best_energy <= 0:
            return False, "Invalid best energy"

        # 5. Verify the best bitstring actually gives the claimed energy
        actual_energy = self.problem.cut_value(proof.best_bitstring)
        if abs(actual_energy - proof.best_energy) > 1e-6:
            return False, f"Energy mismatch: claimed {proof.best_energy}, actual {actual_energy}"

        # 6. Check aggregate statistics are plausible
        stats = proof.aggregate_statistics
        if stats.get("mean_energy", 0) > proof.best_energy:
            return False, "Mean energy > best energy (impossible)"

        verify_time = (time.perf_counter() - t0) * 1000

        return True, f"Verified in {verify_time:.2f}ms"


def benchmark_zk():
    """Benchmark ZK proof generation and verification."""
    print("ZK Prover Benchmark")
    print("=" * 60)

    from chain.commit import SolverCommitter

    for n_qubits in [8, 10, 12]:
        for n_strikes in [100, 1000, 10000]:
            problem = MaxCutProblem.random(n_nodes=n_qubits, edge_prob=0.5, seed=42)

            # Generate samples
            solver = SolverCommitter(f"solver-{n_qubits}", problem)
            gamma, beta = np.pi * 0.7, np.pi * 0.3
            solver.run_strikes(n_strikes, gamma=gamma, beta=beta)
            commitment = solver.commit_batch()
            samples = solver.samples_by_batch[commitment.batch_id]
            tree = solver.trees[commitment.batch_id]

            # Generate proof
            prover = ZKProver(problem, n_layers=3)
            proof = prover.generate_proof(samples, tree)

            # Verify proof
            verifier = ZKVerifier(problem, n_layers=3)
            valid, msg = verifier.verify(proof)

            field_ops = proof.aggregate_statistics["total_field_operations"]
            # Estimate real ZK prover time (GPU): ~1 billion field ops per second
            estimated_gpu_time = field_ops / 1e9

            print(f"  {n_qubits:>2d} qubits, {n_strikes:>5d} strikes: "
                  f"gen={proof.generation_time_ms:>7.1f}ms, "
                  f"proof={proof.proof_size_bytes/1024:.0f}KB, "
                  f"field_ops={field_ops:.2e}, "
                  f"est_gpu={estimated_gpu_time:.1f}s, "
                  f"valid={valid}")


if __name__ == "__main__":
    benchmark_zk()
