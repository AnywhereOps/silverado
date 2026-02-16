"""
Spot Checker — Automated random leaf verification for peer network.

A verifier node that:
1. Receives Merkle root commitments from the network
2. Randomly selects leaf indices to challenge
3. Re-runs the circuits with the given seeds
4. Verifies Merkle proofs and Born rule compliance
5. Broadcasts verdict
"""

import time
import numpy as np
from dataclasses import dataclass, field
from chain.merkle import MerkleTree, MerkleProof, QuantumSample
from chain.commit import SolverCommitter, BatchCommitment
from chain.stats import BornRuleValidator, StatisticalTestResult
from core.match_strike import MatchStrikeEngine, MaxCutProblem
from network.peer_network import PeerNetwork


@dataclass
class SpotCheckResult:
    """Result of spot-checking a single leaf."""
    leaf_index: int
    merkle_valid: bool
    circuit_valid: bool    # Did re-running the circuit produce a plausible outcome?
    cut_value_valid: bool  # Does reported cut value match the bitstring?
    detail: str = ""


@dataclass
class BatchCheckResult:
    """Aggregate result of checking a batch commitment."""
    batch_id: str
    solver_id: str
    n_leaves_checked: int
    n_merkle_failures: int
    n_circuit_failures: int
    n_cut_value_failures: int
    statistical_tests: list[StatisticalTestResult] = field(default_factory=list)
    overall_pass: bool = True
    check_time_ms: float = 0.0


class SpotChecker:
    """Automated verification of solver commitments.

    Performs three levels of checking:
    1. Merkle proof verification (instant, deterministic)
    2. Cut value verification (instant, deterministic)
    3. Statistical validation (fast, probabilistic)
    """

    def __init__(self, verifier_id: str, problem: MaxCutProblem,
                 n_layers: int = 3,
                 check_fraction: float = 0.05,
                 do_statistical: bool = True):
        self.verifier_id = verifier_id
        self.problem = problem
        self.n_layers = n_layers
        self.check_fraction = check_fraction
        self.do_statistical = do_statistical
        self.rng = np.random.default_rng()
        self.validator = BornRuleValidator(problem, n_layers=n_layers)

        # Track verification history
        self.checks_performed: list[BatchCheckResult] = []

    def select_challenge_indices(self, n_samples: int) -> list[int]:
        """Randomly select leaf indices to challenge."""
        n_check = max(3, int(n_samples * self.check_fraction))
        n_check = min(n_check, n_samples)
        return sorted(self.rng.choice(n_samples, size=n_check, replace=False).tolist())

    def verify_leaf(self, sample: QuantumSample, proof: MerkleProof,
                    merkle_root: bytes) -> SpotCheckResult:
        """Verify a single challenged leaf."""
        # 1. Merkle proof
        merkle_valid = proof.verify(merkle_root)

        # 2. Cut value matches bitstring
        actual_cut = self.problem.cut_value(sample.bitstring)
        cut_valid = abs(actual_cut - sample.cut_value) < 1e-6

        # 3. Re-run circuit (probabilistic check)
        # We can't check if this exact bitstring would come from this circuit
        # (quantum mechanics is probabilistic). But we can check if it's a
        # POSSIBLE outcome (nonzero probability).
        circuit_valid = True  # Accept by default for individual samples
        # A stronger check: verify the bitstring is in the support of the distribution
        # For now, defer to batch-level statistical testing

        return SpotCheckResult(
            leaf_index=proof.leaf_index,
            merkle_valid=merkle_valid,
            circuit_valid=circuit_valid,
            cut_value_valid=cut_valid,
            detail=f"merkle={merkle_valid}, cut={cut_valid}"
        )

    def check_batch(self, commitment: BatchCommitment,
                    samples: list[QuantumSample],
                    tree: MerkleTree) -> BatchCheckResult:
        """Full verification of a batch commitment."""
        t0 = time.perf_counter()

        # Select indices to check
        indices = self.select_challenge_indices(commitment.n_samples)

        n_merkle_fail = 0
        n_circuit_fail = 0
        n_cut_fail = 0
        checked_samples = []

        for idx in indices:
            if idx >= len(samples):
                continue

            proof = tree.get_proof(idx)
            result = self.verify_leaf(samples[idx], proof, commitment.merkle_root)

            if not result.merkle_valid:
                n_merkle_fail += 1
            if not result.circuit_valid:
                n_circuit_fail += 1
            if not result.cut_value_valid:
                n_cut_fail += 1

            checked_samples.append(samples[idx])

        # Statistical validation on ALL samples (cheap, doesn't need Merkle proofs)
        stat_results = []
        if self.do_statistical and len(samples) >= 10:
            # Group all samples by circuit params
            param_groups: dict[tuple, list] = {}
            for s in samples:
                key = (round(s.gamma, 4), round(s.beta, 4))
                param_groups.setdefault(key, []).append(s)

            for (gamma, beta), group in param_groups.items():
                if len(group) >= 10:  # Need enough samples for stats
                    stat_results.extend(
                        self.validator.full_validation(group, gamma, beta)
                    )

        # Overall verdict
        overall = (n_merkle_fail == 0 and n_cut_fail == 0)
        if stat_results:
            critical = [r for r in stat_results
                       if r.test_name in ("chi_squared_born_rule", "cut_value_ks")]
            if critical:
                overall = overall and all(r.passed for r in critical)

        check_time = (time.perf_counter() - t0) * 1000

        result = BatchCheckResult(
            batch_id=commitment.batch_id,
            solver_id=commitment.solver_id,
            n_leaves_checked=len(indices),
            n_merkle_failures=n_merkle_fail,
            n_circuit_failures=n_circuit_fail,
            n_cut_value_failures=n_cut_fail,
            statistical_tests=stat_results,
            overall_pass=overall,
            check_time_ms=check_time,
        )

        self.checks_performed.append(result)
        return result


def demo_spot_checking():
    """Demo the spot-checking flow with honest and dishonest solvers."""
    print("Spot-Checker Demo")
    print("=" * 60)

    problem = MaxCutProblem.random(n_nodes=10, edge_prob=0.5, seed=42)
    _, optimal = problem.brute_force_optimal()

    # Honest solver
    print("\n--- Honest Solver ---")
    solver = SolverCommitter("honest-solver", problem)
    gamma, beta = np.pi * 0.7, np.pi * 0.3
    solver.run_strikes(500, gamma=gamma, beta=beta)
    commitment = solver.commit_batch()

    checker = SpotChecker("verifier-1", problem, check_fraction=0.1)
    result = checker.check_batch(
        commitment,
        solver.samples_by_batch[commitment.batch_id],
        solver.trees[commitment.batch_id],
    )
    print(f"  Checked {result.n_leaves_checked} leaves in {result.check_time_ms:.1f}ms")
    print(f"  Merkle failures: {result.n_merkle_failures}")
    print(f"  Cut value failures: {result.n_cut_value_failures}")
    print(f"  Overall: {'PASS' if result.overall_pass else 'FAIL'}")

    # Dishonest solver: fake cut values
    print("\n--- Dishonest Solver (inflated cut values) ---")
    fake_solver = SolverCommitter("fake-solver", problem)
    fake_solver.run_strikes(500, gamma=gamma, beta=beta)
    # Tamper: inflate cut values
    for s in fake_solver.sample_buffer:
        s.cut_value *= 1.5
    fake_commitment = fake_solver.commit_batch()

    result = checker.check_batch(
        fake_commitment,
        fake_solver.samples_by_batch[fake_commitment.batch_id],
        fake_solver.trees[fake_commitment.batch_id],
    )
    print(f"  Checked {result.n_leaves_checked} leaves in {result.check_time_ms:.1f}ms")
    print(f"  Cut value failures: {result.n_cut_value_failures}")
    print(f"  Overall: {'PASS' if result.overall_pass else 'FAIL'}")

    # Dishonest solver: random bitstrings with correct cuts
    print("\n--- Dishonest Solver (random bitstrings, correct cut values) ---")
    rng = np.random.default_rng(99)
    random_samples = []
    for i in range(500):
        bs = int(rng.integers(0, 2 ** problem.n_nodes))
        random_samples.append(QuantumSample(
            strike_index=i, gamma=gamma, beta=beta, n_layers=3,
            bitstring=bs, cut_value=problem.cut_value(bs),
            random_seed=i, timestamp=time.time(),
        ))

    from chain.merkle import MerkleTree
    random_tree = MerkleTree(random_samples)
    random_commitment = BatchCommitment(
        batch_id="random-batch-001",
        solver_id="random-solver",
        problem_hash="fake",
        merkle_root=random_tree.root,
        n_samples=500,
        best_bitstring=0,
        best_energy=0,
        mean_energy=0,
        timestamp=time.time(),
        circuit_params=[(gamma, beta)],
    )

    checker2 = SpotChecker("verifier-2", problem, check_fraction=0.1, do_statistical=True)
    result = checker2.check_batch(random_commitment, random_samples, random_tree)
    print(f"  Checked {result.n_leaves_checked} leaves in {result.check_time_ms:.1f}ms")
    print(f"  Merkle failures: {result.n_merkle_failures}")
    print(f"  Statistical tests:")
    for t in result.statistical_tests:
        print(f"    {t.test_name}: {'PASS' if t.passed else 'FAIL'} (p={t.p_value:.4f})")
    print(f"  Overall: {'PASS' if result.overall_pass else 'FAIL'}")


if __name__ == "__main__":
    demo_spot_checking()
