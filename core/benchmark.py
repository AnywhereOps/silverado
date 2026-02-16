"""
Benchmark — Phase 1 proof that the quantum carburetor works.

Side-by-side comparison across multiple strategies, backends, and graph sizes.
This is the gate: if these numbers don't show quantum advantage, nothing
downstream (distribution, blockchain, incentives) matters.

Compares:
  1. Classical random search (baseline)
  2. Match-strike with random parameters
  3. Match-strike with parameter sweep
  4. Match-strike with greedy narrowing
  5. Adaptive controller (UCB)
  6. Multi-scale adaptive controller
  7. (Optional) Tensor network backend

Across multiple graph sizes and random instances.
"""

import time
import numpy as np
from dataclasses import dataclass
from core.match_strike import MaxCutProblem, MatchStrikeEngine, classical_random_search


@dataclass
class BenchmarkResult:
    method: str
    backend: str
    n_nodes: int
    n_edges: int
    n_samples: int
    best_cut: float
    optimal_cut: float
    ratio: float
    wall_time_ms: float
    convergence_at_90pct: int | None


def convergence_point(history: list[float], optimal: float, threshold: float) -> int | None:
    target = threshold * optimal
    for i, val in enumerate(history):
        if val >= target:
            return i + 1
    return None


def run_single_benchmark(problem: MaxCutProblem, n_strikes: int = 500,
                         seed: int = 42, include_adaptive: bool = True,
                         include_tensor: bool = False) -> list[BenchmarkResult]:
    """Run all methods on a single problem instance."""
    results = []
    optimal_bs, optimal_cut = problem.brute_force_optimal()

    if optimal_cut == 0:
        return results

    def record(method, backend, log, wall_ms):
        results.append(BenchmarkResult(
            method=method,
            backend=backend,
            n_nodes=problem.n_nodes,
            n_edges=len(problem.edges),
            n_samples=len(log.convergence_history),
            best_cut=log.best_cut,
            optimal_cut=optimal_cut,
            ratio=log.best_cut / optimal_cut,
            wall_time_ms=wall_ms,
            convergence_at_90pct=convergence_point(
                log.convergence_history, optimal_cut, 0.9),
        ))

    # 1. Classical random search
    t0 = time.perf_counter()
    log = classical_random_search(problem, n_samples=n_strikes, seed=seed)
    record("classical_random", "none", log, (time.perf_counter() - t0) * 1000)

    # 2-4. Match-strike strategies (state vector)
    for strategy in ["random", "sweep", "greedy"]:
        engine = MatchStrikeEngine(problem, n_layers=3, backend="statevector", seed=seed)
        t0 = time.perf_counter()
        log = engine.run(n_strikes=n_strikes, strategy=strategy)
        record(f"ms_{strategy}", "statevector", log,
               (time.perf_counter() - t0) * 1000)

    # 5-6. Adaptive controllers
    if include_adaptive:
        from core.adaptive_controller import AdaptiveController, MultiScaleController

        # UCB adaptive
        engine = MatchStrikeEngine(problem, n_layers=3, backend="statevector", seed=seed)
        controller = AdaptiveController(engine, grid_size=15, exploration_weight=2.0)
        t0 = time.perf_counter()
        log = controller.run(n_strikes=n_strikes)
        record("adaptive_ucb", "statevector", log,
               (time.perf_counter() - t0) * 1000)

        # Multi-scale adaptive
        engine = MatchStrikeEngine(problem, n_layers=3, backend="statevector", seed=seed)
        multi = MultiScaleController(engine, coarse_grid=8, fine_grid=12)
        t0 = time.perf_counter()
        log = multi.run(n_strikes=n_strikes)
        record("adaptive_multi", "statevector", log,
               (time.perf_counter() - t0) * 1000)

    # 7. Tensor backend (if requested and makes sense for this size)
    if include_tensor and problem.n_nodes >= 10:
        engine = MatchStrikeEngine(
            problem, n_layers=3, backend="tensor", max_bond=32, seed=seed)
        t0 = time.perf_counter()
        log = engine.run(n_strikes=n_strikes, strategy="greedy")
        record("ms_greedy", "tensor", log,
               (time.perf_counter() - t0) * 1000)

    return results


def run_benchmark_suite(node_sizes: list[int] | None = None,
                        n_instances: int = 5,
                        n_strikes: int = 500,
                        include_adaptive: bool = True,
                        include_tensor: bool = False) -> list[BenchmarkResult]:
    """Run benchmarks across multiple graph sizes and instances."""
    if node_sizes is None:
        node_sizes = [8, 10, 12, 14]

    all_results = []
    for n in node_sizes:
        print(f"\n{'='*70}")
        print(f"  MaxCut benchmark: {n} nodes, {n_instances} instances, {n_strikes} strikes")
        print(f"{'='*70}")

        for instance in range(n_instances):
            seed = 1000 * n + instance
            problem = MaxCutProblem.random(n_nodes=n, edge_prob=0.5, seed=seed)
            results = run_single_benchmark(
                problem, n_strikes=n_strikes, seed=seed,
                include_adaptive=include_adaptive,
                include_tensor=include_tensor,
            )
            all_results.extend(results)

            for r in results:
                conv = f"{r.convergence_at_90pct:>4d}" if r.convergence_at_90pct else "   -"
                be = f"({r.backend[:3]})" if r.backend != "none" else "     "
                print(f"  {r.method:<16s} {be} | "
                      f"cut {r.best_cut:6.2f}/{r.optimal_cut:6.2f} "
                      f"({r.ratio:5.1%}) | {r.wall_time_ms:7.1f}ms | 90%@{conv}")

    return all_results


def print_summary(results: list[BenchmarkResult]):
    """Print aggregate summary — the numbers that matter."""
    methods = sorted(set((r.method, r.backend) for r in results))
    node_sizes = sorted(set(r.n_nodes for r in results))

    print(f"\n{'='*80}")
    print(f"  AGGREGATE SUMMARY")
    print(f"{'='*80}")

    for n in node_sizes:
        print(f"\n  {n} nodes:")
        print(f"  {'Method':<22s} | {'Avg Ratio':>9s} | {'Avg Time':>9s} | {'Avg 90% Conv':>12s}")
        print(f"  {'-'*22}-+-{'-'*9}-+-{'-'*9}-+-{'-'*12}")

        for method, backend in methods:
            subset = [r for r in results
                      if r.method == method and r.backend == backend and r.n_nodes == n]
            if not subset:
                continue
            avg_ratio = np.mean([r.ratio for r in subset])
            avg_time = np.mean([r.wall_time_ms for r in subset])
            conv_vals = [r.convergence_at_90pct for r in subset if r.convergence_at_90pct]
            avg_conv = f"{np.mean(conv_vals):>8.0f}" if conv_vals else "       -"
            be = f"({backend[:3]})" if backend != "none" else ""
            label = f"{method} {be}".strip()
            print(f"  {label:<22s} | {avg_ratio:>8.1%} | {avg_time:>7.1f}ms | {avg_conv}")

    # Overall ranking
    print(f"\n  {'='*60}")
    print(f"  OVERALL RANKING (all sizes)")
    print(f"  {'='*60}")
    for method, backend in sorted(methods):
        subset = [r for r in results if r.method == method and r.backend == backend]
        avg_ratio = np.mean([r.ratio for r in subset])
        be = f"({backend[:3]})" if backend != "none" else ""
        label = f"{method} {be}".strip()
        print(f"    {label:<22s}: {avg_ratio:.1%} avg optimality")

    # Crossover analysis
    print(f"\n  {'='*60}")
    print(f"  CROSSOVER: where does match-strike beat classical?")
    print(f"  {'='*60}")
    for n in node_sizes:
        classical = [r for r in results
                     if r.method == "classical_random" and r.n_nodes == n]
        if not classical:
            continue
        cl_avg = np.mean([r.ratio for r in classical])

        best_label = "none"
        best_avg = 0.0
        for method, backend in methods:
            if method == "classical_random":
                continue
            subset = [r for r in results
                      if r.method == method and r.backend == backend and r.n_nodes == n]
            if not subset:
                continue
            avg = np.mean([r.ratio for r in subset])
            if avg > best_avg:
                best_avg = avg
                be = f"({backend[:3]})" if backend != "none" else ""
                best_label = f"{method} {be}".strip()

        delta = best_avg - cl_avg
        winner = "QUANTUM" if delta > 0.005 else "CLASSICAL" if delta < -0.005 else "TIE"
        print(f"    {n:>2d} nodes: classical={cl_avg:.1%}  "
              f"best_quantum={best_avg:.1%} [{best_label}]  "
              f"delta={delta:+.1%}  -> {winner}")


if __name__ == "__main__":
    print("Quantum Carburetor — Phase 1 Benchmark")
    print("Proving the engine before distributing it")
    print()

    results = run_benchmark_suite(
        node_sizes=[8, 10, 12, 14],
        n_instances=5,
        n_strikes=500,
        include_adaptive=True,
        include_tensor=False,  # set True to also test tensor backend
    )
    print_summary(results)
