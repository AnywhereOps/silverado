"""
Peer Network — Speed 2 layer for gossip and coordination.

In production this would use libp2p. For now, simulates an in-process
peer network with realistic latency modeling.

Peers gossip Merkle roots, request spot-checks, and broadcast
verification verdicts. The network reaches provisional consensus
in ~500ms without any blockchain involvement.
"""

import time
import hashlib
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from chain.commit import BatchCommitment


class MessageType(Enum):
    COMMITMENT = "commitment"       # Solver broadcasts Merkle root
    SPOT_CHECK_REQ = "spot_check"   # Verifier requests leaves
    SPOT_CHECK_RESP = "spot_resp"   # Solver provides leaves + proofs
    VERDICT = "verdict"             # Verifier broadcasts pass/fail
    CROSS_VALIDATE = "cross_val"    # Cross-validation request/result


@dataclass
class PeerMessage:
    """A message in the gossip network."""
    msg_type: MessageType
    sender_id: str
    payload: dict
    timestamp: float = 0.0
    msg_id: str = ""

    def __post_init__(self):
        if not self.msg_id:
            data = f"{self.sender_id}{self.msg_type.value}{self.timestamp}{id(self)}"
            self.msg_id = hashlib.sha256(data.encode()).hexdigest()[:12]
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class PeerInfo:
    """Information about a network peer."""
    peer_id: str
    is_solver: bool = False
    is_verifier: bool = False
    latency_ms: float = 10.0     # Simulated network latency
    reputation: float = 0.5       # 0-1 trust score
    connected: bool = True


class PeerNetwork:
    """Simulated peer-to-peer network for Speed 2 coordination.

    Handles:
    - Peer registration and discovery
    - Message gossip with latency simulation
    - Commitment broadcasting
    - Verification coordination
    """

    def __init__(self, base_latency_ms: float = 10.0):
        self.peers: dict[str, PeerInfo] = {}
        self.message_log: list[PeerMessage] = []
        self.subscriptions: dict[str, list[MessageType]] = {}  # peer_id → subscribed types
        self.handlers: dict[str, callable] = {}  # peer_id → message handler
        self.base_latency_ms = base_latency_ms
        self._rng = np.random.default_rng()

        # Commitment tracking
        self.pending_commitments: dict[str, BatchCommitment] = {}  # batch_id → commitment
        self.verdicts: dict[str, list[tuple[str, bool, float]]] = {}  # batch_id → [(verifier, pass, confidence)]
        self.provisional_accepts: dict[str, bool] = {}  # batch_id → accepted?

    def register_peer(self, peer_id: str, is_solver: bool = False,
                      is_verifier: bool = False, latency_ms: float | None = None):
        """Register a peer on the network."""
        lat = latency_ms or self.base_latency_ms + self._rng.exponential(5)
        self.peers[peer_id] = PeerInfo(
            peer_id=peer_id,
            is_solver=is_solver,
            is_verifier=is_verifier,
            latency_ms=lat,
        )
        self.subscriptions[peer_id] = list(MessageType)

    def broadcast(self, msg: PeerMessage, exclude: set[str] | None = None):
        """Broadcast a message to all subscribed peers."""
        self.message_log.append(msg)
        exclude = exclude or set()

        for peer_id, subs in self.subscriptions.items():
            if peer_id in exclude or peer_id == msg.sender_id:
                continue
            if msg.msg_type in subs:
                if peer_id in self.handlers:
                    self.handlers[peer_id](msg)

    def submit_commitment(self, commitment: BatchCommitment) -> float:
        """A solver broadcasts a batch commitment. Returns simulated time to acceptance.

        Flow:
        1. Solver broadcasts Merkle root
        2. Verifiers receive and begin spot-checking
        3. After enough verdicts, provisional acceptance
        """
        batch_id = commitment.batch_id

        # Store commitment
        self.pending_commitments[batch_id] = commitment
        self.verdicts[batch_id] = []

        # Broadcast
        msg = PeerMessage(
            msg_type=MessageType.COMMITMENT,
            sender_id=commitment.solver_id,
            payload=commitment.to_dict(),
        )
        self.broadcast(msg)

        # Simulate verifier latencies
        verifiers = [p for p in self.peers.values()
                     if p.is_verifier and p.connected and p.peer_id != commitment.solver_id]

        if not verifiers:
            # No verifiers — auto-accept (not secure, but functional)
            self.provisional_accepts[batch_id] = True
            return 0.0

        # Simulate verification round-trips
        total_latency = 0.0
        for v in verifiers[:5]:  # Cap at 5 verifiers per commitment
            # Time = network latency + verification compute
            rt_latency = v.latency_ms * 2  # Round trip
            compute_time = self._rng.uniform(1, 5)  # ms for spot-check
            total_latency = max(total_latency, rt_latency + compute_time)

        return total_latency

    def submit_verdict(self, batch_id: str, verifier_id: str,
                       passed: bool, confidence: float):
        """A verifier submits their verdict on a commitment."""
        if batch_id not in self.verdicts:
            return

        self.verdicts[batch_id].append((verifier_id, passed, confidence))

        # Check for provisional consensus
        verdicts = self.verdicts[batch_id]
        if len(verdicts) >= 2:  # Minimum 2 verifiers
            accepts = sum(1 for _, p, _ in verdicts if p)
            rejects = sum(1 for _, p, _ in verdicts if not p)

            if accepts >= 2 and rejects == 0:
                self.provisional_accepts[batch_id] = True
            elif rejects >= 2:
                self.provisional_accepts[batch_id] = False

        # Broadcast verdict
        msg = PeerMessage(
            msg_type=MessageType.VERDICT,
            sender_id=verifier_id,
            payload={
                "batch_id": batch_id,
                "passed": passed,
                "confidence": confidence,
            },
        )
        self.broadcast(msg)

    def is_provisionally_accepted(self, batch_id: str) -> bool | None:
        """Check if a commitment has provisional acceptance. None if pending."""
        return self.provisional_accepts.get(batch_id)

    def get_solvers(self) -> list[PeerInfo]:
        return [p for p in self.peers.values() if p.is_solver and p.connected]

    def get_verifiers(self) -> list[PeerInfo]:
        return [p for p in self.peers.values() if p.is_verifier and p.connected]

    def stats(self) -> dict:
        return {
            "total_peers": len(self.peers),
            "solvers": len(self.get_solvers()),
            "verifiers": len(self.get_verifiers()),
            "messages": len(self.message_log),
            "pending_commitments": len(self.pending_commitments),
            "accepted": sum(1 for v in self.provisional_accepts.values() if v),
            "rejected": sum(1 for v in self.provisional_accepts.values() if not v),
        }


if __name__ == "__main__":
    # Quick demo
    net = PeerNetwork(base_latency_ms=10)
    net.register_peer("solver-1", is_solver=True)
    net.register_peer("solver-2", is_solver=True)
    net.register_peer("verifier-1", is_verifier=True)
    net.register_peer("verifier-2", is_verifier=True)
    net.register_peer("verifier-3", is_verifier=True)

    print(f"Network: {net.stats()}")
