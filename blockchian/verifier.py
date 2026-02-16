"""
Verifier — Spot-check solver honesty.

The key asymmetry: running 10,000 match-strike cycles is expensive.
Verifying 50 of them is cheap. A verifier re-runs a random subset of
claimed circuits and checks that the reported measurement statistics
are plausible.

This is what makes the decentralized network work without trust.
"""

import hashlib
import numpy as np
from dataclasses import dataclass
from scipy import stats as scipy_stats
from match_strike import MatchStrikeEngine, MaxCutProblem, StrikeResult
from solver_node import TaskSpec, TaskResult


@dataclass
class VerificationResult:
    """Outcome of verifying a solver's claimed results."""
    task_id: str
    node_id: str
    verified: bool
    confidence: float     # How confident we are in the verdict
    n_checked: int        # How many samples we spot-checked
    n_suspicious: int     # How many samples looked wrong
    reason: str = ""      # Why it failed (if it did)


class Verifier:
    """Spot-checks solver results for honesty.

    Three levels of verification:
    1. Hash check: Did the solver report consistent data?
    2. Distribution check: Do the samples look like they came from the right circuit?
    3. Quality check: Are the cut values consistent with the reported bitstrings?
    """

    def __init__(self, check_fraction: float = 0.05, confidence_threshold: float = 0.95):
        self.check_fraction = check_fraction  # Fraction of samples to verify
        self.confidence_threshold = confidence_threshold
        self.rng = np.random.default_rng()

    def verify_hash(self, result: TaskResult) -> bool:
        """Level 1: Check internal consistency of results."""
        expected_hash = result.compute_hash()
        return result.result_hash == expected_hash

    def verify_cut_values(self, result: TaskResult, problem: MaxCutProblem) -> tuple[bool, int]:
        """Level 2: Spot-check that reported cut values match bitstrings."""
        n_check = max(1, int(len(result.bitstrings) * self.check_fraction))
        indices = self.rng.choice(len(result.bitstrings), size=n_check, replace=False)

        n_bad = 0
        for idx in indices:
            bs = result.bitstrings[idx]
            claimed_cut = result.cut_values[idx]
            actual_cut = problem.cut_value(bs)
            if abs(claimed_cut - actual_cut) > 1e-6:
                n_bad += 1

        return n_bad == 0, n_bad

    def verify_distribution(self, result: TaskResult, task: TaskSpec,
                            n_reference_shots: int = 1000) -> tuple[bool, float]:
        """Level 3: Statistical test — do samples match the expected distribution?

        Re-runs the same circuit and compares the distribution of cut values.
        Uses a two-sample Kolmogorov-Smirnov test.
        """
        # Re-run the circuit independently
        engine = MatchStrikeEngine(
            task.problem,
            n_layers=task.n_layers,
            backend="statevector" if task.problem.n_nodes <= 20 else "tensor",
        )
        reference_cuts = []
        for _ in range(n_reference_shots):
            strike = engine.single_strike(task.gamma, task.beta)
            reference_cuts.append(strike.cut_value)

        # Compare distributions
        claimed_cuts = result.cut_values
        ks_stat, p_value = scipy_stats.ks_2samp(claimed_cuts, reference_cuts)

        # p_value < 0.01 means the distributions are significantly different
        honest = p_value > 0.01
        return honest, p_value

    def verify(self, result: TaskResult, task: TaskSpec,
               deep: bool = False) -> VerificationResult:
        """Full verification pipeline."""
        # Level 1: Hash
        if not self.verify_hash(result):
            return VerificationResult(
                task_id=result.task_id,
                node_id=result.node_id,
                verified=False,
                confidence=1.0,
                n_checked=0,
                n_suspicious=0,
                reason="Hash mismatch: internal inconsistency",
            )

        # Level 2: Cut value spot-check
        cuts_ok, n_bad = self.verify_cut_values(result, task.problem)
        n_checked = max(1, int(len(result.bitstrings) * self.check_fraction))

        if not cuts_ok:
            return VerificationResult(
                task_id=result.task_id,
                node_id=result.node_id,
                verified=False,
                confidence=1.0,
                n_checked=n_checked,
                n_suspicious=n_bad,
                reason=f"Cut value mismatch: {n_bad}/{n_checked} samples wrong",
            )

        # Level 3: Distribution check (expensive, only if deep=True)
        if deep:
            dist_ok, p_value = self.verify_distribution(result, task)
            if not dist_ok:
                return VerificationResult(
                    task_id=result.task_id,
                    node_id=result.node_id,
                    verified=False,
                    confidence=1.0 - p_value,
                    n_checked=n_checked,
                    n_suspicious=0,
                    reason=f"Distribution mismatch: KS test p={p_value:.4f}",
                )

        # Confidence based on sample size
        # With check_fraction=0.05, checking 5% of samples gives ~95% detection rate
        # for a cheater who fabricates >10% of results
        confidence = 1.0 - (1.0 - self.check_fraction) ** n_checked

        return VerificationResult(
            task_id=result.task_id,
            node_id=result.node_id,
            verified=True,
            confidence=confidence,
            n_checked=n_checked,
            n_suspicious=0,
        )


class CheatingNode:
    """A dishonest solver node for testing the verifier.

    Three cheating strategies:
    1. Fabricate: Generate random results without running the simulation
    2. Inflate: Run the simulation but report inflated cut values
    3. Freeload: Copy results from another node
    """

    def __init__(self, strategy: str = "fabricate"):
        self.strategy = strategy
        self.rng = np.random.default_rng()

    def fabricate_result(self, task: TaskSpec) -> TaskResult:
        """Generate fake results."""
        if self.strategy == "fabricate":
            # Random bitstrings, plausible cut values
            bitstrings = [int(self.rng.integers(0, 2 ** task.problem.n_nodes))
                         for _ in range(task.n_shots)]
            cut_values = [task.problem.cut_value(bs) for bs in bitstrings]
            # This is hard to detect because cut values are correct for the bitstrings!
            # The giveaway is the distribution doesn't match quantum sampling

        elif self.strategy == "inflate":
            # Actually run but inflate cut values by 20%
            engine = MatchStrikeEngine(task.problem, n_layers=task.n_layers)
            bitstrings = []
            cut_values = []
            for _ in range(task.n_shots):
                result = engine.single_strike(task.gamma, task.beta)
                bitstrings.append(result.bitstring)
                cut_values.append(result.cut_value * 1.2)  # Inflate!

        elif self.strategy == "random_values":
            # Correct bitstrings but wrong cut values
            engine = MatchStrikeEngine(task.problem, n_layers=task.n_layers)
            bitstrings = []
            cut_values = []
            for _ in range(task.n_shots):
                result = engine.single_strike(task.gamma, task.beta)
                bitstrings.append(result.bitstring)
                # Report a random cut value instead of actual
                cut_values.append(float(self.rng.uniform(0, 30)))
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        task_result = TaskResult(
            task_id=task.task_id,
            node_id="cheater",
            bitstrings=bitstrings,
            cut_values=cut_values,
            sigma_values=[0.5] * task.n_shots,
            wall_time_ms=100.0,
        )
        task_result.result_hash = task_result.compute_hash()
        return task_result


def test_verifier():
    """Test the verifier against honest and dishonest nodes."""
    from solver_node import SolverNode

    problem = MaxCutProblem.random(n_nodes=10, edge_prob=0.5, seed=42)
    task = TaskSpec(
        task_id="verify-test",
        problem=problem,
        gamma=np.pi * 0.7,
        beta=np.pi * 0.3,
        n_layers=3,
        n_shots=200,
    )

    verifier = Verifier(check_fraction=0.1)
    print("Verifier Test Suite")
    print("=" * 60)

    # Honest node
    print("\n1. Honest solver:")
    honest_node = SolverNode()
    honest_result = honest_node.execute(task)
    v = verifier.verify(honest_result, task, deep=True)
    print(f"   Verified: {v.verified}, Confidence: {v.confidence:.2%}")
    print(f"   Checked: {v.n_checked}, Suspicious: {v.n_suspicious}")

    # Fabricator (hard to catch without deep verification)
    print("\n2. Fabricating solver (random bitstrings, correct cut values):")
    cheater = CheatingNode(strategy="fabricate")
    fake_result = cheater.fabricate_result(task)
    v = verifier.verify(fake_result, task, deep=False)
    print(f"   Shallow: Verified={v.verified} (cheater passes shallow check)")
    v = verifier.verify(fake_result, task, deep=True)
    print(f"   Deep:    Verified={v.verified}, p-value in reason: {v.reason}")

    # Inflator
    print("\n3. Inflating solver (real bitstrings, 20% inflated cut values):")
    cheater = CheatingNode(strategy="inflate")
    inflated_result = cheater.fabricate_result(task)
    v = verifier.verify(inflated_result, task, deep=False)
    print(f"   Shallow: Verified={v.verified}, Reason: {v.reason}")

    # Random values
    print("\n4. Random value solver (real bitstrings, wrong cut values):")
    cheater = CheatingNode(strategy="random_values")
    bad_result = cheater.fabricate_result(task)
    v = verifier.verify(bad_result, task, deep=False)
    print(f"   Shallow: Verified={v.verified}, Reason: {v.reason}")

    # Tampered hash
    print("\n5. Hash tampering:")
    tampered = honest_node.execute(task)
    tampered.result_hash = "tampered"
    v = verifier.verify(tampered, task)
    print(f"   Verified: {v.verified}, Reason: {v.reason}")


if __name__ == "__main__":
    test_verifier()
