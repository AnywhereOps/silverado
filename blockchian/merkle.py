"""
Merkle Tree — Commitment scheme for quantum simulation samples.

Each leaf contains a quantum sample: circuit parameters + measurement outcome + seed.
The solver publishes only the root hash. Verifiers can request any leaf and check
it against the root with a logarithmic proof.

Checking 50 random leaves catches a 10% cheater with 99.5% probability.
"""

import hashlib
import struct
import time
import numpy as np
from dataclasses import dataclass, field


@dataclass
class QuantumSample:
    """A single match-strike measurement — one leaf of the Merkle tree."""
    strike_index: int
    gamma: float
    beta: float
    n_layers: int
    bitstring: int
    cut_value: float
    random_seed: int
    timestamp: float = 0.0

    def serialize(self) -> bytes:
        """Deterministic serialization for hashing."""
        return struct.pack(
            '<IddidfQ',
            self.strike_index,
            self.gamma,
            self.beta,
            self.n_layers,
            self.cut_value,
            self.bitstring,
            self.random_seed,
        )

    def hash(self) -> bytes:
        """SHA-256 hash of this sample."""
        return hashlib.sha256(self.serialize()).digest()

    def to_dict(self) -> dict:
        return {
            "strike_index": self.strike_index,
            "gamma": self.gamma,
            "beta": self.beta,
            "n_layers": self.n_layers,
            "bitstring": self.bitstring,
            "cut_value": self.cut_value,
            "random_seed": self.random_seed,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: dict) -> "QuantumSample":
        return QuantumSample(**d)


def _hash_pair(left: bytes, right: bytes) -> bytes:
    """Hash two 32-byte nodes together."""
    return hashlib.sha256(left + right).digest()


@dataclass
class MerkleProof:
    """Proof that a specific leaf is in the tree."""
    leaf_index: int
    leaf_hash: bytes
    siblings: list[tuple[bytes, str]]  # (hash, "left" or "right")

    def verify(self, root: bytes) -> bool:
        """Verify this proof against a known root hash."""
        current = self.leaf_hash
        for sibling_hash, direction in self.siblings:
            if direction == "left":
                current = _hash_pair(sibling_hash, current)
            else:
                current = _hash_pair(current, sibling_hash)
        return current == root


class MerkleTree:
    """Binary Merkle tree over quantum samples.

    Optimized for the quantum carburetor use case:
    - Builds from a list of QuantumSample objects
    - Supports efficient proof generation for any leaf
    - Root hash is the 32-byte commitment that goes on-chain
    """

    def __init__(self, samples: list[QuantumSample] | None = None):
        self.leaves: list[bytes] = []
        self.layers: list[list[bytes]] = []
        self.samples: list[QuantumSample] = []
        self.root: bytes = b'\x00' * 32

        if samples:
            self.build(samples)

    def build(self, samples: list[QuantumSample]):
        """Build the tree from a list of quantum samples."""
        self.samples = samples
        self.leaves = [s.hash() for s in samples]

        # Pad to power of 2
        n = len(self.leaves)
        next_pow2 = 1
        while next_pow2 < n:
            next_pow2 *= 2
        while len(self.leaves) < next_pow2:
            self.leaves.append(b'\x00' * 32)

        # Build layers bottom-up
        self.layers = [self.leaves[:]]
        current = self.leaves[:]

        while len(current) > 1:
            next_layer = []
            for i in range(0, len(current), 2):
                next_layer.append(_hash_pair(current[i], current[i + 1]))
            self.layers.append(next_layer)
            current = next_layer

        self.root = current[0] if current else b'\x00' * 32

    def get_proof(self, leaf_index: int) -> MerkleProof:
        """Generate a Merkle proof for the leaf at the given index."""
        if leaf_index >= len(self.samples):
            raise IndexError(f"Leaf {leaf_index} out of range (have {len(self.samples)})")

        siblings = []
        idx = leaf_index

        for layer in self.layers[:-1]:  # All layers except root
            if idx % 2 == 0:
                # This node is on the left, sibling is on the right
                sibling_idx = idx + 1
                direction = "right"
            else:
                # This node is on the right, sibling is on the left
                sibling_idx = idx - 1
                direction = "left"

            if sibling_idx < len(layer):
                siblings.append((layer[sibling_idx], direction))
            else:
                siblings.append((b'\x00' * 32, direction))

            idx //= 2

        return MerkleProof(
            leaf_index=leaf_index,
            leaf_hash=self.leaves[leaf_index],
            siblings=siblings,
        )

    def verify_proof(self, proof: MerkleProof) -> bool:
        """Verify a proof against this tree's root."""
        return proof.verify(self.root)

    @property
    def root_hex(self) -> str:
        return self.root.hex()

    @property
    def depth(self) -> int:
        return len(self.layers) - 1

    @property
    def n_leaves(self) -> int:
        return len(self.samples)

    def __repr__(self):
        return (f"MerkleTree(leaves={self.n_leaves}, depth={self.depth}, "
                f"root={self.root_hex[:16]}...)")


def benchmark_merkle():
    """Benchmark Merkle tree operations at various scales."""
    print("Merkle Tree Benchmark")
    print("=" * 70)

    rng = np.random.default_rng(42)

    for n_samples in [1_000, 10_000, 100_000]:
        # Generate samples
        samples = []
        for i in range(n_samples):
            samples.append(QuantumSample(
                strike_index=i,
                gamma=float(rng.uniform(0, 2 * np.pi)),
                beta=float(rng.uniform(0, np.pi)),
                n_layers=3,
                bitstring=int(rng.integers(0, 2**15)),
                cut_value=float(rng.uniform(0, 30)),
                random_seed=int(rng.integers(0, 2**32)),
                timestamp=time.time(),
            ))

        # Build tree
        t0 = time.perf_counter()
        tree = MerkleTree(samples)
        build_time = (time.perf_counter() - t0) * 1000

        # Generate proofs for random leaves
        for n_checks in [50, 100, 500]:
            if n_checks > n_samples:
                continue
            indices = rng.choice(n_samples, size=min(n_checks, n_samples), replace=False)

            t0 = time.perf_counter()
            proofs = [tree.get_proof(int(idx)) for idx in indices]
            proof_time = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            all_valid = all(tree.verify_proof(p) for p in proofs)
            verify_time = (time.perf_counter() - t0) * 1000

            print(f"  {n_samples:>7,d} samples | build: {build_time:>7.1f}ms | "
                  f"{n_checks:>3d} proofs: gen={proof_time:>5.1f}ms, "
                  f"verify={verify_time:>5.1f}ms | valid={all_valid}")

        # Detection probability
        for cheat_pct in [5, 10, 20]:
            for n_checks in [50, 100]:
                detect_prob = 1 - (1 - cheat_pct / 100) ** n_checks
                print(f"    → {cheat_pct}% cheater, {n_checks} checks: "
                      f"{detect_prob:.1%} detection probability")

        print()


if __name__ == "__main__":
    benchmark_merkle()
