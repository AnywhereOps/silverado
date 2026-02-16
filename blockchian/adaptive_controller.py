"""
Adaptive Controller — Bayesian update loop for the Quantum Carburetor.

Each quantum measurement updates a classical probability model of the
parameter space (γ, β). Next query targets the highest-expected-improvement
region. The controller gets smarter every cycle.

This is where the real value lives. The quantum VM is the carburetor.
The adaptive controller is the engine.
"""

import numpy as np
from dataclasses import dataclass, field
from match_strike import MatchStrikeEngine, MaxCutProblem, StrikeResult, CycleLog


@dataclass
class ParameterPoint:
    """A point in (γ, β) space with observed performance."""
    gamma: float
    beta: float
    cut_values: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return np.mean(self.cut_values) if self.cut_values else 0.0

    @property
    def std(self) -> float:
        return np.std(self.cut_values) if len(self.cut_values) > 1 else float('inf')

    @property
    def n_samples(self) -> int:
        return len(self.cut_values)


class AdaptiveController:
    """Bayesian optimization over (γ, β) parameter space.

    Uses a discretized grid with UCB (Upper Confidence Bound) acquisition
    to balance exploration vs exploitation. No scipy dependency.
    """

    def __init__(self, engine: MatchStrikeEngine,
                 gamma_range: tuple[float, float] = (0.1, 2 * np.pi),
                 beta_range: tuple[float, float] = (0.1, np.pi),
                 grid_size: int = 20,
                 exploration_weight: float = 2.0):
        self.engine = engine
        self.gamma_range = gamma_range
        self.beta_range = beta_range
        self.grid_size = grid_size
        self.exploration_weight = exploration_weight
        self.rng = np.random.default_rng()

        # Discretize parameter space
        gammas = np.linspace(*gamma_range, grid_size)
        betas = np.linspace(*beta_range, grid_size)
        self.grid: dict[tuple[int, int], ParameterPoint] = {}
        for i, g in enumerate(gammas):
            for j, b in enumerate(betas):
                self.grid[(i, j)] = ParameterPoint(gamma=g, beta=b)

        self.gammas = gammas
        self.betas = betas

    def _ucb_score(self, point: ParameterPoint) -> float:
        """Upper Confidence Bound: mean + c * std / sqrt(n).

        Unvisited points get infinite score (explore first).
        """
        if point.n_samples == 0:
            return float('inf')
        return point.mean + self.exploration_weight * point.std / np.sqrt(point.n_samples)

    def _select_next_point(self) -> tuple[int, int]:
        """Select the grid point with highest UCB score."""
        best_score = -float('inf')
        best_key = (0, 0)
        candidates = []

        for key, point in self.grid.items():
            score = self._ucb_score(point)
            if score == float('inf'):
                candidates.append(key)
            elif score > best_score:
                best_score = score
                best_key = key

        if candidates:
            # Random among unvisited
            return candidates[int(self.rng.integers(len(candidates)))]
        return best_key

    def _update_neighbors(self, key: tuple[int, int], cut_value: float):
        """Soft update: propagate information to neighboring grid points.

        This acts as a simple kernel smoothing — nearby parameters
        are likely to produce similar results.
        """
        i, j = key
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if (ni, nj) in self.grid:
                    # Add with reduced weight (as if it were a noisy observation)
                    neighbor = self.grid[(ni, nj)]
                    # Discount factor based on distance
                    discount = 0.5 if abs(di) + abs(dj) == 1 else 0.25
                    neighbor.cut_values.append(cut_value * discount + neighbor.mean * (1 - discount)
                                               if neighbor.n_samples > 0 else cut_value * discount)

    def run(self, n_strikes: int = 500, warmup_fraction: float = 0.1) -> CycleLog:
        """Run the adaptive match-strike cycle.

        First warmup_fraction of strikes are random (explore).
        Rest use UCB acquisition (exploit + explore).
        """
        log = CycleLog()
        n_warmup = max(int(n_strikes * warmup_fraction), self.grid_size)

        for i in range(n_strikes):
            if i < n_warmup:
                # Warmup: random sampling
                gi = int(self.rng.integers(self.grid_size))
                bi = int(self.rng.integers(self.grid_size))
            else:
                # UCB acquisition
                gi, bi = self._select_next_point()

            point = self.grid[(gi, bi)]
            result = self.engine.single_strike(point.gamma, point.beta)
            point.cut_values.append(result.cut_value)
            log.record(result)

            # Soft-propagate to neighbors
            if i >= n_warmup:
                self._update_neighbors((gi, bi), result.cut_value)

        return log

    def best_parameters(self) -> tuple[float, float, float]:
        """Return the best (γ, β) found and their average cut value."""
        best_point = max(self.grid.values(), key=lambda p: p.mean if p.n_samples > 0 else -1)
        return best_point.gamma, best_point.beta, best_point.mean

    def parameter_landscape(self) -> np.ndarray:
        """Return the observed performance landscape as a 2D array."""
        landscape = np.zeros((self.grid_size, self.grid_size))
        for (i, j), point in self.grid.items():
            landscape[i, j] = point.mean if point.n_samples > 0 else np.nan
        return landscape

    def uncertainty_landscape(self) -> np.ndarray:
        """Return the uncertainty (std) landscape."""
        landscape = np.full((self.grid_size, self.grid_size), np.nan)
        for (i, j), point in self.grid.items():
            if point.n_samples > 1:
                landscape[i, j] = point.std
        return landscape


class MultiScaleController:
    """Two-stage adaptive controller: coarse grid → fine grid.

    Stage 1: Coarse grid over full parameter space, UCB exploration.
    Stage 2: Fine grid around the best region from Stage 1, exploitation.
    """

    def __init__(self, engine: MatchStrikeEngine,
                 coarse_grid: int = 10, fine_grid: int = 15,
                 exploration_weight: float = 2.0):
        self.engine = engine
        self.coarse_grid = coarse_grid
        self.fine_grid = fine_grid
        self.exploration_weight = exploration_weight

    def run(self, n_strikes: int = 500) -> CycleLog:
        """Run coarse + fine adaptive search."""
        n_coarse = n_strikes // 3
        n_fine = n_strikes - n_coarse

        # Stage 1: Coarse exploration
        coarse = AdaptiveController(
            self.engine,
            grid_size=self.coarse_grid,
            exploration_weight=self.exploration_weight,
        )
        coarse_log = coarse.run(n_strikes=n_coarse, warmup_fraction=0.3)

        # Find best region
        best_gamma, best_beta, _ = coarse.best_parameters()
        gamma_step = (2 * np.pi) / self.coarse_grid
        beta_step = np.pi / self.coarse_grid

        # Stage 2: Fine exploitation around best region
        fine = AdaptiveController(
            self.engine,
            gamma_range=(best_gamma - gamma_step, best_gamma + gamma_step),
            beta_range=(best_beta - beta_step, best_beta + beta_step),
            grid_size=self.fine_grid,
            exploration_weight=self.exploration_weight * 0.5,  # Less exploration
        )
        fine_log = fine.run(n_strikes=n_fine, warmup_fraction=0.1)

        # Merge logs
        combined = CycleLog()
        for result in coarse_log.strikes + fine_log.strikes:
            combined.record(result)

        return combined


def benchmark_adaptive(n_nodes: int = 12, n_strikes: int = 500, n_instances: int = 5):
    """Benchmark adaptive vs. non-adaptive controllers."""
    print(f"Benchmarking adaptive controller: {n_nodes} nodes, {n_strikes} strikes, "
          f"{n_instances} instances")
    print()

    results = {"random": [], "greedy": [], "adaptive": [], "multiscale": [], "classical": []}

    for inst in range(n_instances):
        seed = 2000 + inst
        problem = MaxCutProblem.random(n_nodes=n_nodes, edge_prob=0.5, seed=seed)
        optimal_bs, optimal_cut = problem.brute_force_optimal()

        if optimal_cut == 0:
            continue

        # Classical random
        from match_strike import classical_random_search
        cl_log = classical_random_search(problem, n_samples=n_strikes, seed=seed)
        results["classical"].append(cl_log.best_cut / optimal_cut)

        # Match-strike random
        engine = MatchStrikeEngine(problem, n_layers=3, seed=seed)
        rnd_log = engine.run(n_strikes=n_strikes, strategy="random")
        results["random"].append(rnd_log.best_cut / optimal_cut)

        # Match-strike greedy
        engine = MatchStrikeEngine(problem, n_layers=3, seed=seed)
        gr_log = engine.run(n_strikes=n_strikes, strategy="greedy")
        results["greedy"].append(gr_log.best_cut / optimal_cut)

        # Adaptive UCB
        engine = MatchStrikeEngine(problem, n_layers=3, seed=seed)
        adaptive = AdaptiveController(engine, grid_size=15, exploration_weight=2.0)
        ad_log = adaptive.run(n_strikes=n_strikes)
        results["adaptive"].append(ad_log.best_cut / optimal_cut)

        # Multi-scale adaptive
        engine = MatchStrikeEngine(problem, n_layers=3, seed=seed)
        multi = MultiScaleController(engine, coarse_grid=8, fine_grid=12)
        ms_log = multi.run(n_strikes=n_strikes)
        results["multiscale"].append(ms_log.best_cut / optimal_cut)

        print(f"  Instance {inst}: optimal={optimal_cut:.2f}  "
              f"classical={cl_log.best_cut / optimal_cut:.1%}  "
              f"random={rnd_log.best_cut / optimal_cut:.1%}  "
              f"greedy={gr_log.best_cut / optimal_cut:.1%}  "
              f"adaptive={ad_log.best_cut / optimal_cut:.1%}  "
              f"multiscale={ms_log.best_cut / optimal_cut:.1%}")

    print()
    print("  Averages:")
    for method, ratios in results.items():
        if ratios:
            print(f"    {method:<12s}: {np.mean(ratios):.1%} ± {np.std(ratios):.1%}")


if __name__ == "__main__":
    benchmark_adaptive(n_nodes=12, n_strikes=500, n_instances=5)
