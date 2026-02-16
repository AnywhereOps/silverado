"""
Finality — Determines when work units are final, handles conflicts,
and manages the challenge/dispute resolution process.

Three finality levels:
  1. Provisional: accepted by peer spot-checkers (~500ms)
  2. Soft: referenced by enough subsequent work units (~5 seconds)
  3. Hard: past challenge window with no successful disputes (~5 minutes)

Challenge flow:
  1. Anyone can challenge a work unit within the challenge window
  2. Challenger specifies which samples to re-verify
  3. If challenge succeeds: unit is rejected, solver slashed
  4. If challenge fails: challenger loses deposit
"""

import time
import hashlib
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from dag import QuantumDAG, WorkUnit, WorkUnitStatus


class ChallengeStatus(Enum):
    OPEN = "open"
    RESOLVED_VALID = "resolved_valid"       # Original unit was honest
    RESOLVED_FRAUD = "resolved_fraud"       # Original unit was fraudulent
    EXPIRED = "expired"                     # Challenge window passed without resolution


@dataclass
class Challenge:
    """A dispute against a work unit."""
    challenge_id: str
    target_unit_id: str
    challenger_id: str
    sample_indices: list[int]       # Which samples to re-verify
    timestamp: float
    status: ChallengeStatus = ChallengeStatus.OPEN
    resolution_proof: str = ""      # Evidence for resolution
    resolution_time: float = 0.0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp


@dataclass
class FinalityStatus:
    """Current finality status of a work unit."""
    unit_id: str
    provisional: bool = False
    soft_final: bool = False
    hard_final: bool = False
    challenged: bool = False
    rejected: bool = False
    confirmation_depth: int = 0
    age_seconds: float = 0.0
    active_challenges: int = 0


class FinalityManager:
    """Manages the finality pipeline for the DAG.

    Integrates with the DAG consensus to:
    - Track confirmation progress
    - Handle challenge/dispute resolution
    - Determine payment readiness
    - Detect and resolve conflicts
    """

    def __init__(self, dag: QuantumDAG,
                 provisional_verifiers: int = 2,
                 challenge_window_sec: float = 300.0,
                 challenge_deposit: float = 100.0):
        self.dag = dag
        self.provisional_verifiers = provisional_verifiers
        self.challenge_window = challenge_window_sec
        self.challenge_deposit = challenge_deposit

        self.challenges: dict[str, Challenge] = {}
        self.unit_challenges: dict[str, list[str]] = {}  # unit_id → [challenge_ids]
        self._challenge_counter = 0

    def get_finality(self, unit_id: str) -> FinalityStatus:
        """Get the current finality status of a work unit."""
        if unit_id not in self.dag.units:
            return FinalityStatus(unit_id=unit_id)

        unit = self.dag.units[unit_id]
        now = time.time()
        age = now - unit.timestamp

        # Count verification votes
        approvals = sum(1 for v in unit.verification_votes.values() if v)
        rejections = sum(1 for v in unit.verification_votes.values() if not v)

        # Active challenges
        active = [cid for cid in self.unit_challenges.get(unit_id, [])
                  if self.challenges[cid].status == ChallengeStatus.OPEN]

        provisional = (approvals >= self.provisional_verifiers and rejections == 0
                       and not active)
        soft_final = unit.status in (WorkUnitStatus.SOFT_CONFIRMED,
                                     WorkUnitStatus.HARD_CONFIRMED)
        hard_final = (unit.status == WorkUnitStatus.HARD_CONFIRMED and
                      age >= self.challenge_window and not active)
        rejected = unit.status == WorkUnitStatus.REJECTED

        return FinalityStatus(
            unit_id=unit_id,
            provisional=provisional,
            soft_final=soft_final,
            hard_final=hard_final,
            challenged=len(active) > 0,
            rejected=rejected,
            confirmation_depth=unit.confirmation_depth,
            age_seconds=age,
            active_challenges=len(active),
        )

    def submit_challenge(self, target_unit_id: str, challenger_id: str,
                         sample_indices: list[int]) -> Challenge:
        """Submit a challenge against a work unit."""
        unit = self.dag.units.get(target_unit_id)
        if not unit:
            raise ValueError(f"Unit {target_unit_id} not found")

        age = time.time() - unit.timestamp
        if age > self.challenge_window:
            raise ValueError(f"Challenge window expired ({age:.0f}s > {self.challenge_window:.0f}s)")

        if unit.status == WorkUnitStatus.REJECTED:
            raise ValueError("Unit already rejected")

        self._challenge_counter += 1
        challenge_id = f"challenge-{self._challenge_counter:06d}"

        challenge = Challenge(
            challenge_id=challenge_id,
            target_unit_id=target_unit_id,
            challenger_id=challenger_id,
            sample_indices=sample_indices,
            timestamp=time.time(),
        )

        self.challenges[challenge_id] = challenge
        self.unit_challenges.setdefault(target_unit_id, []).append(challenge_id)

        return challenge

    def resolve_challenge(self, challenge_id: str, fraud_proven: bool,
                          proof: str = ""):
        """Resolve a challenge."""
        challenge = self.challenges.get(challenge_id)
        if not challenge:
            raise ValueError(f"Challenge {challenge_id} not found")

        if challenge.status != ChallengeStatus.OPEN:
            raise ValueError(f"Challenge already resolved: {challenge.status}")

        if fraud_proven:
            challenge.status = ChallengeStatus.RESOLVED_FRAUD
            # Reject the work unit
            self.dag.reject_unit(challenge.target_unit_id)
        else:
            challenge.status = ChallengeStatus.RESOLVED_VALID

        challenge.resolution_proof = proof
        challenge.resolution_time = time.time()

    def expire_old_challenges(self):
        """Expire challenges that have been open too long without resolution."""
        for cid, challenge in self.challenges.items():
            if (challenge.status == ChallengeStatus.OPEN and
                challenge.age_seconds > self.challenge_window):
                challenge.status = ChallengeStatus.EXPIRED

    def detect_conflicts(self, problem_id: str) -> list[tuple[str, str]]:
        """Detect conflicting work units for the same problem.

        Conflicts: two units claiming very different best energies
        for the same problem, both confirmed.
        """
        confirmed_units = [
            u for u in self.dag.units.values()
            if u.problem_id == problem_id and
            u.status in (WorkUnitStatus.SOFT_CONFIRMED, WorkUnitStatus.HARD_CONFIRMED)
        ]

        conflicts = []
        for i, u1 in enumerate(confirmed_units):
            for u2 in confirmed_units[i+1:]:
                # If energies differ by > 20%, flag as conflict
                max_e = max(abs(u1.best_energy), abs(u2.best_energy), 1e-10)
                diff = abs(u1.best_energy - u2.best_energy) / max_e
                if diff > 0.2:
                    conflicts.append((u1.unit_id, u2.unit_id))

        return conflicts

    def get_payable_units(self) -> list[str]:
        """Get work units that have achieved hard finality and are ready for payment."""
        payable = []
        for uid, unit in self.dag.units.items():
            if uid == "genesis":
                continue
            f = self.get_finality(uid)
            if f.hard_final and not f.rejected and not f.challenged:
                payable.append(uid)
        return payable

    def summary(self) -> dict:
        """Summary of finality states across the DAG."""
        statuses = {
            "provisional": 0,
            "soft_final": 0,
            "hard_final": 0,
            "challenged": 0,
            "rejected": 0,
            "pending": 0,
        }
        for uid in self.dag.units:
            if uid == "genesis":
                continue
            f = self.get_finality(uid)
            if f.rejected:
                statuses["rejected"] += 1
            elif f.hard_final:
                statuses["hard_final"] += 1
            elif f.challenged:
                statuses["challenged"] += 1
            elif f.soft_final:
                statuses["soft_final"] += 1
            elif f.provisional:
                statuses["provisional"] += 1
            else:
                statuses["pending"] += 1

        return {
            "finality_states": statuses,
            "total_challenges": len(self.challenges),
            "open_challenges": sum(
                1 for c in self.challenges.values()
                if c.status == ChallengeStatus.OPEN
            ),
            "payable_units": len(self.get_payable_units()),
        }


def simulate_finality():
    """Simulate the finality pipeline with challenges."""
    import hashlib

    print("Finality Pipeline Simulation")
    print("=" * 60)

    # Use a large challenge window so we can actually test challenges
    dag = QuantumDAG(soft_threshold=3, hard_threshold=8, challenge_window_sec=9999.0)
    fm = FinalityManager(dag, challenge_window_sec=9999.0)
    rng = np.random.default_rng(42)

    # Build up the DAG
    unit_ids = []
    print("\nBuilding DAG with 50 honest units + 5 fraudulent...")
    for i in range(55):
        solver_id = f"solver-{i % 10}"
        is_fraud = i >= 50  # Last 5 are fraudulent

        energy = float(rng.uniform(10, 35))
        if is_fraud:
            energy *= 2  # Suspiciously high

        merkle_root = hashlib.sha256(f"{solver_id}{i}".encode()).digest()
        unit = dag.add_unit(
            solver_id=solver_id,
            problem_id="maxcut-42",
            merkle_root=merkle_root,
            best_energy=energy,
            total_samples=500,
        )

        # Add verification votes
        for v in range(3):
            dag.add_verification(unit.unit_id, f"verifier-{v}", not is_fraud)

        unit_ids.append(unit.unit_id)

    # Challenge the fraudulent units
    print("\nChallenging fraudulent units...")
    for uid in unit_ids[50:]:
        try:
            challenge = fm.submit_challenge(uid, "challenger-1", list(range(50)))
            fm.resolve_challenge(challenge.challenge_id, fraud_proven=True,
                                proof="cut values don't match bitstrings")
            print(f"  Challenged {uid[:12]}: FRAUD PROVEN → rejected")
        except ValueError as e:
            print(f"  Challenge failed: {e}")

    # Check finality states
    summary = fm.summary()
    print(f"\nFinality Summary:")
    for state, count in summary["finality_states"].items():
        print(f"  {state:<15s}: {count}")
    print(f"  Payable units: {summary['payable_units']}")
    print(f"  Total challenges: {summary['total_challenges']}")

    # Verify payable units are all honest
    payable = fm.get_payable_units()
    fraudulent_payable = [uid for uid in payable if uid in unit_ids[50:]]
    print(f"\n  Fraudulent units in payable: {len(fraudulent_payable)} (should be 0)")

    # Detect conflicts
    conflicts = fm.detect_conflicts("maxcut-42")
    print(f"  Conflicts detected: {len(conflicts)}")


if __name__ == "__main__":
    simulate_finality()
