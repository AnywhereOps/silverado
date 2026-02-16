"""
Benchmark — Side-by-side comparison of solution strategies on MaxCut.

Compares:
  1. Brute force (ground truth)
  2. Classical random search
  3. Match-strike with random parameters
  4. Match-strike with parameter sweep
  5. Match-strike with greedy narrowing

Across multiple graph sizes and random instances.
"""

import time
import numpy as np
from dataclasses import dataclass
from match_strike import MaxCutProblem, MatchStrikeEngine, classical_random_search


@dataclass
class BenchmarkResult:
    method: str
    n_nodes: int
    n_edges: int
    n_samples: int
    best_cut: float
    optimal_cut: float
    ratio: float  # best_cut / optimal_cut
    wall_time_ms: float
    convergence_at_90pct: int | None  # samples needed to reach 90% of optimal


def run_single_benchmark(problem: MaxCutProblem, n_strikes: int = 500,
                         seed: int = 42) -> list[BenchmarkResult]:
    """Run all methods on a single problem instance."""
    results = []
    optimal_bs, optimal_cut = problem.brute_force_optimal()

    if optimal_cut == 0:
        return results  # degenerate graph

    def convergence_point(history: list[float], threshold: float) -> int | None:
        target = threshold * optimal_cut
        for i, val in enumerate(history):
            if val >= target:
                return i + 1
        return None

    # 1. Classical random search
    t0 = time.perf_counter()
    classical_log = classical_random_search(problem, n_samples=n_strikes, seed=seed)
    t1 = time.perf_counter()
    results.append(BenchmarkResult(
        method="classical_random",
        n_nodes=problem.n_nodes,
        n_edges=len(problem.edges),
        n_samples=n_strikes,
        best_cut=classical_log.best_cut,
        optimal_cut=optimal_cut,
        ratio=classical_log.best_cut / optimal_cut,
        wall_time_ms=(t1 - t0) * 1000,
        convergence_at_90pct=convergence_point(classical_log.convergence_history, 0.9),
    ))

    # 2-4. Match-strike strategies
    for strategy in ["random", "sweep", "greedy"]:
        engine = MatchStrikeEngine(problem, n_layers=3, seed=seed)
        t0 = time.perf_counter()
        log = engine.run(n_strikes=n_strikes, strategy=strategy)
        t1 = time.perf_counter()
        results.append(BenchmarkResult(
            method=f"match_strike_{strategy}",
            n_nodes=problem.n_nodes,
            n_edges=len(problem.edges),
            n_samples=n_strikes,
            best_cut=log.best_cut,
            optimal_cut=optimal_cut,
            ratio=log.best_cut / optimal_cut,
            wall_time_ms=(t1 - t0) * 1000,
            convergence_at_90pct=convergence_point(log.convergence_history, 0.9),
        ))

    return results


def run_benchmark_suite(node_sizes: list[int] | None = None,
                        n_instances: int = 5,
                        n_strikes: int = 500) -> list[BenchmarkResult]:
    """Run benchmarks across multiple graph sizes and instances."""
    if node_sizes is None:
        node_sizes = [8, 10, 12]

    all_results = []
    for n in node_sizes:
        print(f"\n{'='*60}")
        print(f"  MaxCut benchmark: {n} nodes, {n_instances} random instances")
        print(f"{'='*60}")

        for instance in range(n_instances):
            seed = 1000 * n + instance
            problem = MaxCutProblem.random(n_nodes=n, edge_prob=0.5, seed=seed)
            results = run_single_benchmark(problem, n_strikes=n_strikes, seed=seed)
            all_results.extend(results)

            # Print per-instance results
            for r in results:
                conv = f"{r.convergence_at_90pct:>4d}" if r.convergence_at_90pct else "   -"
                print(f"  {r.method:<24s} | cut {r.best_cut:6.2f}/{r.optimal_cut:6.2f} "
                      f"({r.ratio:5.1%}) | {r.wall_time_ms:7.1f}ms | 90%@{conv}")

    return all_results


def print_summary(results: list[BenchmarkResult]):
    """Print aggregate summary across all instances."""
    methods = sorted(set(r.method for r in results))
    node_sizes = sorted(set(r.n_nodes for r in results))

    print(f"\n{'='*70}")
    print(f"  AGGREGATE SUMMARY")
    print(f"{'='*70}")

    for n in node_sizes:
        print(f"\n  {n} nodes:")
        print(f"  {'Method':<24s} | {'Avg Ratio':>9s} | {'Avg Time':>9s} | {'Avg 90% Conv':>12s}")
        print(f"  {'-'*24}-+-{'-'*9}-+-{'-'*9}-+-{'-'*12}")

        for method in methods:
            subset = [r for r in results if r.method == method and r.n_nodes == n]
            if not subset:
                continue
            avg_ratio = np.mean([r.ratio for r in subset])
            avg_time = np.mean([r.wall_time_ms for r in subset])
            conv_vals = [r.convergence_at_90pct for r in subset if r.convergence_at_90pct]
            avg_conv = f"{np.mean(conv_vals):>8.0f}" if conv_vals else "       -"
            print(f"  {method:<24s} | {avg_ratio:>8.1%} | {avg_time:>7.1f}ms | {avg_conv}")

    # Overall winner
    print(f"\n  Overall by method:")
    for method in methods:
        subset = [r for r in results if r.method == method]
        avg_ratio = np.mean([r.ratio for r in subset])
        print(f"    {method:<24s}: {avg_ratio:.1%} avg optimality")


if __name__ == "__main__":
    print("Quantum Carburetor — Phase 1 Benchmark")
    print("MaxCut on random graphs: classical vs. match-strike")
    print()

    results = run_benchmark_suite(
        node_sizes=[8, 10, 12],
        n_instances=5,
        n_strikes=500,
    )
    print_summary(results)
