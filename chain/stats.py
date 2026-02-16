"""
Stats — Born Rule compliance testing for quantum sample verification.

The core insight: to fake Born-rule-compliant samples without running the
simulation, a cheater needs to know the output probability distribution.
But computing that distribution IS the simulation. No shortcut.

Tests:
  1. Chi-squared goodness-of-fit against theoretical distribution
  2. Entropy test: quantum samples have characteristic entropy
  3. Correlation test: sequential samples should be independent
  4. Cut value distribution test: KS test against expected distribution
"""

import numpy as np
from scipy import stats as scipy_stats
from dataclasses import dataclass
from chain.merkle import QuantumSample
from core.quantum_vm import QuantumVM
from core.match_strike import MatchStrikeEngine, MaxCutProblem


@dataclass
class StatisticalTestResult:
    """Result of a statistical validity test."""
    test_name: str
    passed: bool
    statistic: float
    p_value: float
    threshold: float
    detail: str = ""


class BornRuleValidator:
    """Tests whether quantum samples comply with Born rule predictions.

    This is the statistical heart of the verification system. If samples
    pass these tests, they were (almost certainly) produced by an honest
    quantum simulation.
    """

    def __init__(self, problem: MaxCutProblem, n_layers: int = 3,
                 significance: float = 0.01):
        self.problem = problem
        self.n_layers = n_layers
        self.significance = significance  # p-value threshold
        self._ref_cache: dict[tuple[float, float], np.ndarray] = {}  # (gamma, beta) → distribution

    def _get_reference_distribution(self, gamma: float, beta: float,
                                    n_reference: int = 10000) -> np.ndarray:
        """Compute reference distribution by running the circuit.

        Returns probability distribution over measurement outcomes.
        For small qubit counts, this is exact. For large, it's sampled.
        Caches results by (gamma, beta) to avoid redundant simulations.
        """
        cache_key = (round(gamma, 6), round(beta, 6))
        if cache_key in self._ref_cache:
            return self._ref_cache[cache_key]

        dist = self._compute_reference_distribution(gamma, beta, n_reference)
        self._ref_cache[cache_key] = dist
        return dist

    def _compute_reference_distribution(self, gamma: float, beta: float,
                                        n_reference: int = 10000) -> np.ndarray:
        """Actually compute the reference distribution (uncached)."""
        if self.problem.n_nodes <= 20:
            # Exact: compute full state vector
            vm = QuantumVM(self.problem.n_nodes)
            costs = self.problem.all_cut_values()
            max_cost = costs.max() if costs.max() > 0 else 1.0
            phase_costs = (costs / max_cost) * np.pi

            vm.superpose_all()
            for _ in range(self.n_layers):
                vm.apply_phase_oracle(gamma * phase_costs)
                vm.apply_mixer(beta)
            return vm.probabilities()
        else:
            # Sample-based reference
            engine = MatchStrikeEngine(
                self.problem, n_layers=self.n_layers, backend="tensor"
            )
            samples = []
            for _ in range(n_reference):
                result = engine.single_strike(gamma, beta)
                samples.append(result.bitstring)
            hist = np.bincount(samples, minlength=2**self.problem.n_nodes)
            return hist / hist.sum()

    def chi_squared_test(self, samples: list[QuantumSample],
                         gamma: float, beta: float) -> StatisticalTestResult:
        """Chi-squared goodness-of-fit test against Born rule distribution.

        Tests whether observed sample frequencies match theoretical probabilities.
        """
        dim = 2 ** self.problem.n_nodes
        expected_probs = self._get_reference_distribution(gamma, beta)

        # Build observed histogram
        bitstrings = [s.bitstring for s in samples]
        observed = np.bincount(bitstrings, minlength=dim).astype(float)
        expected = expected_probs * len(samples)

        # Only test bins with expected count >= 5 (chi-squared requirement)
        mask = expected >= 5
        if mask.sum() < 2:
            # Not enough bins with sufficient expected counts
            # Fall back to KS test on cut values
            return self.cut_value_ks_test(samples, gamma, beta)

        chi2_stat = np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask])
        dof = mask.sum() - 1
        p_value = float(1 - scipy_stats.chi2.cdf(chi2_stat, dof))

        return StatisticalTestResult(
            test_name="chi_squared_born_rule",
            passed=p_value >= self.significance,
            statistic=float(chi2_stat),
            p_value=p_value,
            threshold=self.significance,
            detail=f"dof={dof}, bins_tested={mask.sum()}"
        )

    def cut_value_ks_test(self, samples: list[QuantumSample],
                          gamma: float, beta: float,
                          n_reference: int = 1000) -> StatisticalTestResult:
        """Kolmogorov-Smirnov test on the distribution of cut values.

        Doesn't require knowing the full quantum state — just checks that
        the cut value distribution looks right.
        """
        # Generate reference cut values (cached by params)
        cache_key = ("ks", round(gamma, 6), round(beta, 6))
        if cache_key in self._ref_cache:
            reference_cuts = self._ref_cache[cache_key]
        else:
            engine = MatchStrikeEngine(self.problem, n_layers=self.n_layers)
            reference_cuts = []
            for _ in range(n_reference):
                result = engine.single_strike(gamma, beta)
                reference_cuts.append(result.cut_value)
            self._ref_cache[cache_key] = reference_cuts

        claimed_cuts = [s.cut_value for s in samples]
        ks_stat, p_value = scipy_stats.ks_2samp(claimed_cuts, reference_cuts)

        return StatisticalTestResult(
            test_name="cut_value_ks",
            passed=p_value >= self.significance,
            statistic=float(ks_stat),
            p_value=float(p_value),
            threshold=self.significance,
            detail=f"n_claimed={len(claimed_cuts)}, n_reference={n_reference}"
        )

    def entropy_test(self, samples: list[QuantumSample]) -> StatisticalTestResult:
        """Test that sample entropy is consistent with quantum computation.

        Random bitstrings have maximum entropy (~n_qubits bits).
        Quantum circuit outputs typically have lower entropy due to
        interference patterns. Suspiciously high entropy suggests fake samples.
        """
        dim = 2 ** self.problem.n_nodes
        bitstrings = [s.bitstring for s in samples]
        hist = np.bincount(bitstrings, minlength=dim).astype(float)
        hist = hist / hist.sum()
        hist = hist[hist > 0]

        entropy = -np.sum(hist * np.log2(hist))
        max_entropy = self.problem.n_nodes  # log2(2^n) = n bits

        # Quantum circuits with problem Hamiltonians should have entropy
        # significantly below max. If entropy is > 95% of max, suspicious.
        entropy_ratio = entropy / max_entropy
        # This is a soft test — not all circuits reduce entropy
        passed = entropy_ratio < 0.98

        return StatisticalTestResult(
            test_name="entropy",
            passed=passed,
            statistic=float(entropy),
            p_value=float(1 - entropy_ratio),  # Not a true p-value
            threshold=0.98,
            detail=f"entropy={entropy:.2f}/{max_entropy:.2f} bits, "
                   f"ratio={entropy_ratio:.3f}"
        )

    def independence_test(self, samples: list[QuantumSample]) -> StatisticalTestResult:
        """Test that sequential samples are independent.

        Quantum measurements should produce independent samples (each strike
        resets the VM). If samples show autocorrelation, the solver might
        be reusing/copying results.
        """
        if len(samples) < 20:
            return StatisticalTestResult(
                test_name="independence",
                passed=True,
                statistic=0.0,
                p_value=1.0,
                threshold=self.significance,
                detail="Too few samples for autocorrelation test"
            )

        cut_values = np.array([s.cut_value for s in samples])
        # Normalize
        if cut_values.std() == 0:
            return StatisticalTestResult(
                test_name="independence",
                passed=False,
                statistic=1.0,
                p_value=0.0,
                threshold=self.significance,
                detail="All cut values identical — not independent"
            )

        cv_norm = (cut_values - cut_values.mean()) / cut_values.std()

        # Lag-1 autocorrelation
        n = len(cv_norm)
        autocorr = np.sum(cv_norm[:-1] * cv_norm[1:]) / (n - 1)

        # Under independence, autocorrelation ~ N(0, 1/n)
        z_stat = autocorr * np.sqrt(n)
        p_value = float(2 * (1 - scipy_stats.norm.cdf(abs(z_stat))))

        return StatisticalTestResult(
            test_name="independence",
            passed=p_value >= self.significance,
            statistic=float(autocorr),
            p_value=p_value,
            threshold=self.significance,
            detail=f"lag1_autocorr={autocorr:.4f}, z={z_stat:.2f}"
        )

    def full_validation(self, samples: list[QuantumSample],
                        gamma: float, beta: float) -> list[StatisticalTestResult]:
        """Run all statistical tests on a batch of samples."""
        results = []

        # Group by (gamma, beta) if mixed
        results.append(self.chi_squared_test(samples, gamma, beta))
        results.append(self.entropy_test(samples))
        results.append(self.independence_test(samples))

        return results

    def batch_verdict(self, samples: list[QuantumSample],
                      gamma: float, beta: float) -> tuple[bool, list[StatisticalTestResult]]:
        """Run all tests, return overall pass/fail verdict."""
        results = self.full_validation(samples, gamma, beta)
        # Pass only if all critical tests pass
        critical_tests = ["chi_squared_born_rule", "cut_value_ks"]
        critical_results = [r for r in results if r.test_name in critical_tests]

        if critical_results:
            all_pass = all(r.passed for r in critical_results)
        else:
            all_pass = all(r.passed for r in results)

        return all_pass, results


def benchmark_stats():
    """Benchmark statistical validation tests."""
    print("Statistical Validation Benchmark")
    print("=" * 60)

    problem = MaxCutProblem.random(n_nodes=10, edge_prob=0.5, seed=42)
    validator = BornRuleValidator(problem, n_layers=3)

    import time

    # Honest samples
    print("\nHonest solver:")
    engine = MatchStrikeEngine(problem, n_layers=3, seed=42)
    gamma, beta = np.pi * 0.7, np.pi * 0.3
    honest_samples = []
    for i in range(500):
        result = engine.single_strike(gamma, beta)
        honest_samples.append(QuantumSample(
            strike_index=i, gamma=gamma, beta=beta, n_layers=3,
            bitstring=result.bitstring, cut_value=result.cut_value,
            random_seed=i, timestamp=time.time(),
        ))

    t0 = time.perf_counter()
    passed, results = validator.batch_verdict(honest_samples, gamma, beta)
    val_time = (time.perf_counter() - t0) * 1000
    print(f"  Verdict: {'PASS' if passed else 'FAIL'} ({val_time:.1f}ms)")
    for r in results:
        print(f"    {r.test_name}: {'PASS' if r.passed else 'FAIL'} "
              f"(stat={r.statistic:.4f}, p={r.p_value:.4f}) {r.detail}")

    # Fake samples (random bitstrings)
    print("\nFake solver (random bitstrings):")
    rng = np.random.default_rng(99)
    fake_samples = []
    for i in range(500):
        bs = int(rng.integers(0, 2**problem.n_nodes))
        fake_samples.append(QuantumSample(
            strike_index=i, gamma=gamma, beta=beta, n_layers=3,
            bitstring=bs, cut_value=problem.cut_value(bs),
            random_seed=i, timestamp=time.time(),
        ))

    t0 = time.perf_counter()
    passed, results = validator.batch_verdict(fake_samples, gamma, beta)
    val_time = (time.perf_counter() - t0) * 1000
    print(f"  Verdict: {'PASS' if passed else 'FAIL'} ({val_time:.1f}ms)")
    for r in results:
        print(f"    {r.test_name}: {'PASS' if r.passed else 'FAIL'} "
              f"(stat={r.statistic:.4f}, p={r.p_value:.4f}) {r.detail}")


if __name__ == "__main__":
    benchmark_stats()
