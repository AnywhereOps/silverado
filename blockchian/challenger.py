"""
Challenger — Automated fraud detection and challenge submission.

Monitors the DAG for suspicious work units and automatically challenges
them. Uses all available verification methods:
  1. Merkle leaf spot-checking
  2. Born rule compliance testing
  3. Cross-validation against peer distributions
  4. Cut value verification

The challenger is economically incentivized: wins challenger deposit +
50% of fraudster's slashed stake when fraud is proven.
"""

import time
import hashlib
import numpy as np
from dataclasses import dataclass, field
from merkle import MerkleTree, QuantumSample
from commit import SolverCommitter, BatchCommitment
from spot_checker import SpotChecker, BatchCheckResult
from cross_validator import CrossValidator
from stats import BornRuleValidator
from reputation import ReputationSystem, ReputationEvent
from dag import QuantumDAG, WorkUnit
from finality import FinalityManager, Challenge
from match_strike import MaxCutProblem


@dataclass
class ChallengeDecision:
    """Whether and why to challenge a work unit."""
    unit_id: str
    should_challenge: bool
    confidence: float           # How confident we are fraud exists
    evidence: list[str]         # List of evidence reasons
    estimated_reward: float     # Expected value of challenging
    estimated_risk: float       # Expected value of losing challenge deposit


class AutoChallenger:
    """Automated fraud detection and challenge bot.

    Monitors submissions and challenges suspicious ones when the
    expected value of challenging is positive.

    EV(challenge) = P(fraud) * reward - P(honest) * deposit_loss
    Challenge when EV > 0.
    """

    def __init__(self, challenger_id: str, problem: MaxCutProblem,
                 reputation: ReputationSystem,
                 challenge_deposit: float = 100.0,
                 slash_reward_pct: float = 0.50,
                 confidence_threshold: float = 0.7):
        self.challenger_id = challenger_id
        self.problem = problem
        self.reputation = reputation
        self.challenge_deposit = challenge_deposit
        self.slash_reward_pct = slash_reward_pct
        self.confidence_threshold = confidence_threshold

        self.spot_checker = SpotChecker(
            challenger_id, problem, check_fraction=0.1, do_statistical=True
        )
        self.cross_validator = CrossValidator(problem)
        self.stats_validator = BornRuleValidator(problem)

        # Track challenges
        self.challenges_submitted: list[ChallengeDecision] = []
        self.challenges_won: int = 0
        self.challenges_lost: int = 0

    def evaluate_submission(self, commitment: BatchCommitment,
                            samples: list[QuantumSample],
                            tree: MerkleTree,
                            solver_stake: float = 1000.0) -> ChallengeDecision:
        """Evaluate whether a submission should be challenged."""
        evidence = []
        fraud_probability = 0.0

        # 1. Spot-check Merkle proofs and cut values
        check_result = self.spot_checker.check_batch(commitment, samples, tree)
        if check_result.n_merkle_failures > 0:
            evidence.append(f"Merkle proof failures: {check_result.n_merkle_failures}")
            fraud_probability = max(fraud_probability, 0.99)
        if check_result.n_cut_value_failures > 0:
            evidence.append(f"Cut value mismatches: {check_result.n_cut_value_failures}")
            fraud_probability = max(fraud_probability, 0.99)

        # 2. Statistical tests
        for test in check_result.statistical_tests:
            if not test.passed:
                evidence.append(f"Statistical test failed: {test.test_name} (p={test.p_value:.4f})")
                if test.test_name in ("chi_squared_born_rule", "cut_value_ks"):
                    fraud_probability = max(fraud_probability, 0.9)
                else:
                    fraud_probability = max(fraud_probability, 0.5)

        # 3. Reputation check
        solver_rep = self.reputation.get_reputation(commitment.solver_id)
        if solver_rep:
            if solver_rep.trust_level == "untrusted":
                evidence.append(f"Untrusted solver (rep={solver_rep.score:.0f})")
                fraud_probability = max(fraud_probability, fraud_probability + 0.1)
            elif solver_rep.slash_count > 0:
                evidence.append(f"Previously slashed ({solver_rep.slash_count} times)")
                fraud_probability = max(fraud_probability, fraud_probability + 0.2)

        # 4. Suspiciously high energy
        if commitment.best_energy > 0:
            # Compare against expected distribution
            mean_energy = commitment.mean_energy
            if commitment.best_energy > mean_energy * 3:
                evidence.append(f"Suspiciously high best energy: "
                              f"{commitment.best_energy:.2f} vs mean {mean_energy:.2f}")
                fraud_probability = max(fraud_probability, 0.6)

        # Calculate expected value
        reward = solver_stake * self.slash_reward_pct
        ev = fraud_probability * reward - (1 - fraud_probability) * self.challenge_deposit

        should_challenge = (fraud_probability >= self.confidence_threshold and ev > 0)

        decision = ChallengeDecision(
            unit_id=commitment.batch_id,
            should_challenge=should_challenge,
            confidence=fraud_probability,
            evidence=evidence,
            estimated_reward=reward if should_challenge else 0,
            estimated_risk=self.challenge_deposit if should_challenge else 0,
        )

        self.challenges_submitted.append(decision)
        return decision

    def monitor_and_challenge(self, commitments: list[tuple[BatchCommitment, list[QuantumSample], MerkleTree]],
                              solver_stakes: dict[str, float] | None = None) -> list[ChallengeDecision]:
        """Monitor a batch of commitments and challenge suspicious ones."""
        if solver_stakes is None:
            solver_stakes = {}

        decisions = []
        for commitment, samples, tree in commitments:
            stake = solver_stakes.get(commitment.solver_id, 1000.0)
            decision = self.evaluate_submission(commitment, samples, tree, stake)
            decisions.append(decision)

            if decision.should_challenge:
                # Record challenge attempt
                self.reputation.record_event(
                    commitment.solver_id,
                    ReputationEvent.FAILED_SPOT_CHECK,
                    context=f"Challenged by {self.challenger_id}: {', '.join(decision.evidence[:3])}"
                )

        return decisions

    @property
    def success_rate(self) -> float:
        total = self.challenges_won + self.challenges_lost
        if total == 0:
            return 0.0
        return self.challenges_won / total

    @property
    def stats(self) -> dict:
        return {
            "total_evaluations": len(self.challenges_submitted),
            "challenges_recommended": sum(1 for d in self.challenges_submitted if d.should_challenge),
            "challenges_won": self.challenges_won,
            "challenges_lost": self.challenges_lost,
            "success_rate": self.success_rate,
        }


def demo_challenger():
    """Demonstrate the automated challenger."""
    from match_strike import MatchStrikeEngine

    print("Auto-Challenger Demo")
    print("=" * 60)

    problem = MaxCutProblem.random(n_nodes=10, edge_prob=0.5, seed=42)
    _, optimal = problem.brute_force_optimal()

    reputation = ReputationSystem()
    challenger = AutoChallenger("challenger-bot", problem, reputation)

    # Generate honest commitments
    print("\nEvaluating honest commitments...")
    honest_submissions = []
    for i in range(5):
        solver_id = f"honest-{i}"
        reputation.register_peer(solver_id)
        # Build up some reputation
        for _ in range(10):
            reputation.record_event(solver_id, ReputationEvent.HONEST_COMMITMENT)

        solver = SolverCommitter(solver_id, problem)
        gamma, beta = float(np.random.uniform(0.5, 5)), float(np.random.uniform(0.2, 2.5))
        solver.run_strikes(200, gamma=gamma, beta=beta)
        commitment = solver.commit_batch()
        tree = solver.trees[commitment.batch_id]
        samples = solver.samples_by_batch[commitment.batch_id]
        honest_submissions.append((commitment, samples, tree))

    decisions = challenger.monitor_and_challenge(honest_submissions)
    for d in decisions:
        status = "CHALLENGE" if d.should_challenge else "PASS"
        print(f"  {d.unit_id}: {status} (confidence={d.confidence:.2f}, "
              f"evidence={len(d.evidence)})")

    # Generate dishonest commitments
    print("\nEvaluating dishonest commitments...")
    rng = np.random.default_rng(99)
    dishonest_submissions = []

    # Type 1: Random bitstrings
    fake_solver_id = "faker-random"
    reputation.register_peer(fake_solver_id)
    fake_samples = []
    for i in range(200):
        bs = int(rng.integers(0, 2 ** problem.n_nodes))
        fake_samples.append(QuantumSample(
            strike_index=i, gamma=np.pi, beta=np.pi/2, n_layers=3,
            bitstring=bs, cut_value=problem.cut_value(bs),
            random_seed=i, timestamp=time.time(),
        ))
    fake_tree = MerkleTree(fake_samples)
    fake_commitment = BatchCommitment(
        batch_id="faker-random-batch-001",
        solver_id=fake_solver_id,
        problem_hash="fake",
        merkle_root=fake_tree.root,
        n_samples=200,
        best_bitstring=fake_samples[0].bitstring,
        best_energy=max(s.cut_value for s in fake_samples),
        mean_energy=float(np.mean([s.cut_value for s in fake_samples])),
        timestamp=time.time(),
        circuit_params=[(np.pi, np.pi/2)],
    )
    dishonest_submissions.append((fake_commitment, fake_samples, fake_tree))

    # Type 2: Inflated cut values
    inflater_id = "faker-inflate"
    reputation.register_peer(inflater_id)
    inflate_solver = SolverCommitter(inflater_id, problem)
    inflate_solver.run_strikes(200, gamma=np.pi, beta=np.pi/2)
    for s in inflate_solver.sample_buffer:
        s.cut_value *= 2.0  # Double the cuts
    inflate_commit = inflate_solver.commit_batch()
    inflate_tree = inflate_solver.trees[inflate_commit.batch_id]
    inflate_samples = inflate_solver.samples_by_batch[inflate_commit.batch_id]
    dishonest_submissions.append((inflate_commit, inflate_samples, inflate_tree))

    decisions = challenger.monitor_and_challenge(dishonest_submissions)
    for d in decisions:
        status = "CHALLENGE" if d.should_challenge else "PASS"
        print(f"  {d.unit_id}: {status} (confidence={d.confidence:.2f})")
        for e in d.evidence:
            print(f"    → {e}")

    print(f"\nChallenger stats: {challenger.stats}")


if __name__ == "__main__":
    demo_challenger()
