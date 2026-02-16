"""
Reputation — Solver and verifier reliability tracking.

Reputation accumulates from:
  + Verified honest commitments
  + Accurate verification verdicts
  + Consistent cross-validation
  - Failed spot-checks
  - Outlier detection in cross-validation
  - Slashing events

Higher reputation = priority in task assignment, lower verification burden.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class ReputationEvent(Enum):
    # Positive events
    HONEST_COMMITMENT = "honest_commitment"       # +10
    ACCURATE_VERDICT = "accurate_verdict"          # +5
    CONSISTENT_CROSS_VAL = "consistent_crossval"   # +3
    FAST_RESPONSE = "fast_response"                # +1

    # Negative events
    FAILED_SPOT_CHECK = "failed_spot_check"        # -50
    OUTLIER_DETECTED = "outlier_detected"          # -30
    SLOW_RESPONSE = "slow_response"                # -2
    SLASHED = "slashed"                            # -100


REPUTATION_WEIGHTS = {
    ReputationEvent.HONEST_COMMITMENT: 10,
    ReputationEvent.ACCURATE_VERDICT: 5,
    ReputationEvent.CONSISTENT_CROSS_VAL: 3,
    ReputationEvent.FAST_RESPONSE: 1,
    ReputationEvent.FAILED_SPOT_CHECK: -50,
    ReputationEvent.OUTLIER_DETECTED: -30,
    ReputationEvent.SLOW_RESPONSE: -2,
    ReputationEvent.SLASHED: -100,
}


@dataclass
class ReputationRecord:
    """A single reputation event."""
    peer_id: str
    event: ReputationEvent
    delta: int
    timestamp: float
    context: str = ""


@dataclass
class PeerReputation:
    """Aggregate reputation for a peer."""
    peer_id: str
    score: float = 500.0        # Starting reputation (0-1000 range)
    events: list[ReputationRecord] = field(default_factory=list)
    total_commitments: int = 0
    successful_commitments: int = 0
    total_verifications: int = 0
    accurate_verifications: int = 0
    total_cross_vals: int = 0
    consistent_cross_vals: int = 0
    slash_count: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_commitments == 0:
            return 0.0
        return self.successful_commitments / self.total_commitments

    @property
    def verification_accuracy(self) -> float:
        if self.total_verifications == 0:
            return 0.0
        return self.accurate_verifications / self.total_verifications

    @property
    def trust_level(self) -> str:
        if self.score >= 800:
            return "trusted"
        elif self.score >= 500:
            return "standard"
        elif self.score >= 200:
            return "probation"
        else:
            return "untrusted"

    @property
    def verification_discount(self) -> float:
        """How much to reduce verification burden for trusted nodes.

        trusted: only 2% spot-check
        standard: 5% spot-check
        probation: 10% spot-check
        untrusted: 20% spot-check
        """
        if self.score >= 800:
            return 0.02
        elif self.score >= 500:
            return 0.05
        elif self.score >= 200:
            return 0.10
        else:
            return 0.20


class ReputationSystem:
    """Manages reputation scores for all network participants."""

    def __init__(self, initial_score: float = 500.0,
                 min_score: float = 0.0, max_score: float = 1000.0,
                 decay_rate: float = 0.999):
        self.initial_score = initial_score
        self.min_score = min_score
        self.max_score = max_score
        self.decay_rate = decay_rate  # Per-event decay toward mean
        self.peers: dict[str, PeerReputation] = {}

    def register_peer(self, peer_id: str):
        """Register a new peer with initial reputation."""
        if peer_id not in self.peers:
            self.peers[peer_id] = PeerReputation(
                peer_id=peer_id,
                score=self.initial_score,
            )

    def record_event(self, peer_id: str, event: ReputationEvent,
                     context: str = ""):
        """Record a reputation event for a peer."""
        if peer_id not in self.peers:
            self.register_peer(peer_id)

        peer = self.peers[peer_id]
        delta = REPUTATION_WEIGHTS[event]

        # Apply event
        peer.score = np.clip(peer.score + delta, self.min_score, self.max_score)

        # Light decay toward initial score (prevents runaway scores)
        peer.score = peer.score * self.decay_rate + self.initial_score * (1 - self.decay_rate)

        # Record
        record = ReputationRecord(
            peer_id=peer_id,
            event=event,
            delta=delta,
            timestamp=time.time(),
            context=context,
        )
        peer.events.append(record)

        # Update counters
        if event == ReputationEvent.HONEST_COMMITMENT:
            peer.total_commitments += 1
            peer.successful_commitments += 1
        elif event == ReputationEvent.FAILED_SPOT_CHECK:
            peer.total_commitments += 1
        elif event == ReputationEvent.ACCURATE_VERDICT:
            peer.total_verifications += 1
            peer.accurate_verifications += 1
        elif event == ReputationEvent.CONSISTENT_CROSS_VAL:
            peer.total_cross_vals += 1
            peer.consistent_cross_vals += 1
        elif event == ReputationEvent.OUTLIER_DETECTED:
            peer.total_cross_vals += 1
        elif event == ReputationEvent.SLASHED:
            peer.slash_count += 1

    def get_reputation(self, peer_id: str) -> PeerReputation | None:
        return self.peers.get(peer_id)

    def get_check_fraction(self, peer_id: str) -> float:
        """How much to spot-check this peer based on reputation."""
        peer = self.peers.get(peer_id)
        if peer is None:
            return 0.20  # Unknown peer: maximum checking
        return peer.verification_discount

    def rank_peers(self, role: str = "solver") -> list[tuple[str, float]]:
        """Rank peers by reputation score."""
        peers = list(self.peers.values())
        peers.sort(key=lambda p: p.score, reverse=True)
        return [(p.peer_id, p.score) for p in peers]

    def get_trusted_peers(self, min_score: float = 500.0) -> list[str]:
        """Get peers with reputation above threshold."""
        return [pid for pid, p in self.peers.items() if p.score >= min_score]

    def summary(self) -> dict:
        """Network-wide reputation summary."""
        if not self.peers:
            return {"total_peers": 0}

        scores = [p.score for p in self.peers.values()]
        levels = {}
        for p in self.peers.values():
            lvl = p.trust_level
            levels[lvl] = levels.get(lvl, 0) + 1

        return {
            "total_peers": len(self.peers),
            "avg_score": float(np.mean(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "trust_levels": levels,
            "total_slashes": sum(p.slash_count for p in self.peers.values()),
        }


def simulate_reputation():
    """Simulate reputation dynamics over many rounds."""
    print("Reputation System Simulation")
    print("=" * 60)

    system = ReputationSystem()

    # 10 honest solvers, 2 cheaters
    honest_ids = [f"honest-{i}" for i in range(10)]
    cheater_ids = ["cheater-1", "cheater-2"]
    all_ids = honest_ids + cheater_ids

    for pid in all_ids:
        system.register_peer(pid)

    rng = np.random.default_rng(42)

    print("\nSimulating 100 rounds...")
    for round_num in range(100):
        for pid in honest_ids:
            # Honest solvers: 95% pass, 5% slow
            if rng.random() < 0.95:
                system.record_event(pid, ReputationEvent.HONEST_COMMITMENT)
            else:
                system.record_event(pid, ReputationEvent.SLOW_RESPONSE)

            # Cross-validation: 98% consistent
            if rng.random() < 0.98:
                system.record_event(pid, ReputationEvent.CONSISTENT_CROSS_VAL)

        for pid in cheater_ids:
            # Cheaters: 70% appear honest, 20% caught by spot-check, 10% outlier
            r = rng.random()
            if r < 0.70:
                system.record_event(pid, ReputationEvent.HONEST_COMMITMENT)
            elif r < 0.90:
                system.record_event(pid, ReputationEvent.FAILED_SPOT_CHECK,
                                   context=f"round {round_num}")
            else:
                system.record_event(pid, ReputationEvent.OUTLIER_DETECTED,
                                   context=f"round {round_num}")

            # Cross-validation: 60% consistent (cheaters diverge more)
            if rng.random() < 0.60:
                system.record_event(pid, ReputationEvent.CONSISTENT_CROSS_VAL)
            else:
                system.record_event(pid, ReputationEvent.OUTLIER_DETECTED)

    # Print results
    print("\nFinal Reputation Scores:")
    ranking = system.rank_peers()
    for pid, score in ranking:
        peer = system.get_reputation(pid)
        check = system.get_check_fraction(pid)
        print(f"  {pid:<12s}: score={score:>6.1f} [{peer.trust_level:<10s}] "
              f"check={check:.0%} success={peer.success_rate:.0%}")

    print(f"\nSummary: {system.summary()}")

    # Verify cheaters have lower scores
    honest_scores = [system.get_reputation(pid).score for pid in honest_ids]
    cheater_scores = [system.get_reputation(pid).score for pid in cheater_ids]
    print(f"\nAvg honest score: {np.mean(honest_scores):.1f}")
    print(f"Avg cheater score: {np.mean(cheater_scores):.1f}")
    print(f"Separation: {np.mean(honest_scores) - np.mean(cheater_scores):.1f} points")


if __name__ == "__main__":
    simulate_reputation()
