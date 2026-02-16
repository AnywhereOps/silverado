"""
Cross-Validator — Free verification through solver agreement.

When multiple solvers work on the same problem with the same circuit
parameters, their sample distributions should converge to the same
Born-rule distribution. Any solver whose distribution significantly
deviates is either cheating, buggy, or astronomically unlucky.

This is "free" verification: no extra computation, just comparing
what solvers already produced.
"""

import numpy as np
from dataclasses import dataclass, field
from commit import BatchCommitment
from merkle import QuantumSample
from match_strike import MaxCutProblem


@dataclass
class CrossValidationResult:
    """Result of cross-validating one solver against the group."""
    solver_id: str
    batch_id: str
    avg_kl_divergence: float
    max_kl_divergence: float
    n_peers_compared: int
    is_outlier: bool
    outlier_threshold: float
    detail: str = ""


class CrossValidator:
    """Compares solver distributions to detect outliers.

    Uses KL-divergence between solver output distributions.
    If one solver's distribution significantly differs from the majority,
    it's flagged for explicit verification.
    """

    def __init__(self, problem: MaxCutProblem, outlier_threshold: float | None = None):
        self.problem = problem
        self.dim = 2 ** problem.n_nodes
        # Adaptive threshold: KL divergence scales with log(dim)
        # For honest solvers, expected KL ~ dim / (2 * n_samples) due to sampling noise
        # Use a generous default that scales with problem size
        self.outlier_threshold = outlier_threshold if outlier_threshold is not None else float(np.log(self.dim))

        # Collect distributions from all solvers
        self.distributions: dict[str, np.ndarray] = {}  # solver_id → distribution
        self.batch_map: dict[str, str] = {}  # solver_id → batch_id

    def add_solver_samples(self, solver_id: str, batch_id: str,
                           samples: list[QuantumSample]):
        """Register a solver's sample distribution."""
        bitstrings = [s.bitstring for s in samples]
        hist = np.bincount(bitstrings, minlength=self.dim).astype(float)
        hist += 1e-10  # Laplace smoothing to avoid log(0)
        hist /= hist.sum()
        self.distributions[solver_id] = hist
        self.batch_map[solver_id] = batch_id

    def add_solver_distribution(self, solver_id: str, batch_id: str,
                                distribution: np.ndarray):
        """Register a pre-computed distribution."""
        dist = distribution + 1e-10
        dist /= dist.sum()
        self.distributions[solver_id] = dist
        self.batch_map[solver_id] = batch_id

    def kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """KL(P || Q) — asymmetric divergence."""
        # Only compute over non-zero support of P
        mask = p > 1e-15
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

    def symmetric_kl(self, p: np.ndarray, q: np.ndarray) -> float:
        """Symmetric KL divergence: (KL(P||Q) + KL(Q||P)) / 2."""
        return (self.kl_divergence(p, q) + self.kl_divergence(q, p)) / 2

    def validate_all(self) -> list[CrossValidationResult]:
        """Cross-validate all registered solvers. Returns per-solver results."""
        solver_ids = list(self.distributions.keys())
        n = len(solver_ids)

        if n < 2:
            return [CrossValidationResult(
                solver_id=sid,
                batch_id=self.batch_map.get(sid, ""),
                avg_kl_divergence=0.0,
                max_kl_divergence=0.0,
                n_peers_compared=0,
                is_outlier=False,
                outlier_threshold=self.outlier_threshold,
                detail="Not enough solvers for cross-validation"
            ) for sid in solver_ids]

        # Compute pairwise KL divergences
        kl_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    kl_matrix[i, j] = self.symmetric_kl(
                        self.distributions[solver_ids[i]],
                        self.distributions[solver_ids[j]],
                    )

        results = []
        for i, sid in enumerate(solver_ids):
            peers = [kl_matrix[i, j] for j in range(n) if j != i]
            avg_kl = float(np.mean(peers))
            max_kl = float(np.max(peers))

            # Relative outlier detection using modified Z-score
            # Robust to high absolute KL from sampling noise
            all_avg_kls = []
            for ii in range(len(solver_ids)):
                pp = [kl_matrix[ii, jj] for jj in range(len(solver_ids)) if jj != ii]
                all_avg_kls.append(np.mean(pp))

            median_kl = np.median(all_avg_kls)
            mad = np.median(np.abs(np.array(all_avg_kls) - median_kl))
            # Modified Z-score: how many MADs above median
            if mad > 1e-10:
                z_score = 0.6745 * (avg_kl - median_kl) / mad
            else:
                z_score = 0.0

            # Outlier if modified Z-score > 3.5 (standard threshold)
            is_outlier = z_score > 3.5

            results.append(CrossValidationResult(
                solver_id=sid,
                batch_id=self.batch_map.get(sid, ""),
                avg_kl_divergence=avg_kl,
                max_kl_divergence=max_kl,
                n_peers_compared=n - 1,
                is_outlier=is_outlier,
                outlier_threshold=self.outlier_threshold,
                detail=f"pairwise_kl={[f'{kl:.4f}' for kl in peers]}"
            ))

        return results

    def find_outliers(self) -> list[str]:
        """Return list of solver IDs flagged as outliers."""
        results = self.validate_all()
        return [r.solver_id for r in results if r.is_outlier]

    def consensus_distribution(self) -> np.ndarray | None:
        """Compute the consensus distribution (median of non-outlier solvers)."""
        results = self.validate_all()
        honest = [self.distributions[r.solver_id] for r in results if not r.is_outlier]
        if not honest:
            return None
        return np.median(honest, axis=0)


def demo_cross_validation():
    """Demo cross-validation with honest and dishonest solvers."""
    from match_strike import MatchStrikeEngine
    import time

    print("Cross-Validation Demo")
    print("=" * 60)

    problem = MaxCutProblem.random(n_nodes=10, edge_prob=0.5, seed=42)
    gamma, beta = np.pi * 0.7, np.pi * 0.3

    cv = CrossValidator(problem, outlier_threshold=0.1)

    # 4 honest solvers
    print("\nGenerating solver samples...")
    for i in range(4):
        engine = MatchStrikeEngine(problem, n_layers=3, seed=100 + i)
        samples = []
        for j in range(1000):
            result = engine.single_strike(gamma, beta)
            samples.append(QuantumSample(
                strike_index=j, gamma=gamma, beta=beta, n_layers=3,
                bitstring=result.bitstring, cut_value=result.cut_value,
                random_seed=j, timestamp=time.time(),
            ))
        cv.add_solver_samples(f"honest-{i}", f"batch-{i}", samples)
        print(f"  honest-{i}: 1000 samples")

    # 1 dishonest solver (random bitstrings)
    rng = np.random.default_rng(99)
    fake_samples = []
    for j in range(1000):
        bs = int(rng.integers(0, 2 ** problem.n_nodes))
        fake_samples.append(QuantumSample(
            strike_index=j, gamma=gamma, beta=beta, n_layers=3,
            bitstring=bs, cut_value=problem.cut_value(bs),
            random_seed=j, timestamp=time.time(),
        ))
    cv.add_solver_samples("cheater", "batch-fake", fake_samples)
    print(f"  cheater: 1000 fake samples")

    # Cross-validate
    print("\nCross-validation results:")
    t0 = time.perf_counter()
    results = cv.validate_all()
    cv_time = (time.perf_counter() - t0) * 1000

    for r in results:
        status = "OUTLIER" if r.is_outlier else "OK"
        print(f"  {r.solver_id:<12s}: avg_KL={r.avg_kl_divergence:.4f}, "
              f"max_KL={r.max_kl_divergence:.4f} [{status}]")

    outliers = cv.find_outliers()
    print(f"\nOutliers detected: {outliers}")
    print(f"Cross-validation time: {cv_time:.1f}ms")

    # All honest
    print("\n--- All honest solvers ---")
    cv2 = CrossValidator(problem, outlier_threshold=0.1)
    for i in range(5):
        engine = MatchStrikeEngine(problem, n_layers=3, seed=200 + i)
        samples = []
        for j in range(1000):
            result = engine.single_strike(gamma, beta)
            samples.append(QuantumSample(
                strike_index=j, gamma=gamma, beta=beta, n_layers=3,
                bitstring=result.bitstring, cut_value=result.cut_value,
                random_seed=j, timestamp=time.time(),
            ))
        cv2.add_solver_samples(f"solver-{i}", f"batch-{i}", samples)

    results2 = cv2.validate_all()
    for r in results2:
        status = "OUTLIER" if r.is_outlier else "OK"
        print(f"  {r.solver_id:<12s}: avg_KL={r.avg_kl_divergence:.4f} [{status}]")
    print(f"  Outliers: {cv2.find_outliers()}")


if __name__ == "__main__":
    demo_cross_validation()
