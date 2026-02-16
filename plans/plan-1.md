# Fast Crypto for Quantum Simulation

## Solving the Speed Problem in Proof of Quantum Work

**Authors:** Drew Kemp-Dahlberg & Claude
**Date:** February 2026

---

## The Problem in One Sentence

Quantum simulation needs microsecond feedback loops. Blockchain consensus takes seconds to minutes. How do you verify quantum work on-chain without killing the speed that makes it useful?

---

## The Insight That Changes Everything

Quantum mechanics is already probabilistic. You don't verify that a quantum simulation produced the RIGHT answer (there is no single right answer). You verify that the DISTRIBUTION of answers matches what quantum mechanics predicts. This is a statistical test, not a deterministic check. And statistical tests are inherently:

- Fast (chi-squared test on a sample is O(n) where n is sample size)
- Parallelizable (every verifier can test independently)
- Tunable (more samples = more confidence, choose your threshold)
- Naturally asymmetric (generating 100,000 samples is expensive, testing 500 of them is cheap)

This means you don't need traditional blockchain consensus at all. You need statistical consensus, which is fundamentally faster.

---

## The Architecture: Three Speeds

```
SPEED 1: MICROSECONDS (Local Compute, No Consensus)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Match-strike cycle runs locally on solver node.
No network communication during computation.
Thousands of quantum VM fires per second.
Results accumulate in local sample buffer.

         ↓ batch every ~100ms

SPEED 2: MILLISECONDS (Statistical Pre-Verification)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Solver commits a Merkle root of sample batch.
2-3 peer nodes spot-check random samples.
Statistical test: does this batch look quantum?
If yes: provisional acceptance (100-500ms total).

         ↓ commit every ~5 seconds

SPEED 3: SECONDS (On-Chain Finality)  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Batch commitments roll up to chain.
Optimistic finality: accepted unless challenged.
Challenge window: 5 minutes.
Slashing if fraud proven.
```

The solver never waits for consensus to continue computing. It fires continuously at Speed 1, batches at Speed 2, and settles at Speed 3. The feedback loop that matters (adaptive controller updating the next quantum query) runs entirely at Speed 1, locally, with zero network latency.

---

## Component 1: The Sample Commitment Scheme

**Problem:** A solver claims it ran circuit C and got measurement outcomes {m1, m2, ... mn}. How do you prove this without re-running all n samples?

**Solution: Merkle Tree of Quantum Samples**

```
                    Root Hash (32 bytes, goes on-chain)
                   /                                    \
            H(left)                                H(right)
           /      \                               /        \
     H(s1,s2)   H(s3,s4)                   H(s5,s6)    H(s7,s8)
      /    \      /    \                     /    \       /    \
    s1     s2   s3     s4                  s5    s6     s7    s8
```

Each leaf si contains:
- Circuit parameters (angles, gates) for that strike
- Measurement outcome (bitstring)
- Random seed used (for reproducibility)
- Timestamp

The solver publishes the Merkle root. A verifier can request any leaf, re-run that specific circuit with the given seed, and check if the outcome matches. If ANY leaf fails, the entire batch is fraudulent.

**Why this is fast:**
- Solver builds the tree locally: O(n log n) hashes, milliseconds for 10,000 samples
- Merkle root is 32 bytes: trivial to publish
- Verification of one leaf: one circuit simulation + one Merkle proof = microseconds to milliseconds
- Probability of detecting a cheater who faked k% of samples, checking m random leaves: 1 - (1-k/100)^m

Checking 50 random leaves catches a 10% cheater with 99.5% probability.
Checking 100 random leaves catches a 5% cheater with 99.4% probability.

---

## Component 2: Statistical Validity Testing

**Problem:** Even if individual samples are honestly computed, the solver might be submitting garbage circuits (not actually solving the problem). How do you verify the computation is USEFUL?

**Solution: Born Rule Compliance Test**

For any quantum circuit, the Born rule predicts the exact probability distribution over measurement outcomes. A solver submitting samples from a well-run simulation will produce outcomes matching this distribution. A cheater submitting random bitstrings will not.

**The test (runs in milliseconds):**

1. Take the solver's submitted samples for a given circuit
2. Compute the empirical frequency distribution
3. Compare against the theoretical distribution using chi-squared test
4. If p-value < threshold: flag as suspicious

**Why this is hard to fake:**

To fake Born-rule-compliant samples without running the simulation, a cheater would need to know the output probability distribution. But computing that distribution IS the simulation. There's no shortcut. You either run the circuit honestly or you can't produce statistically valid samples.

**Edge case: What if the cheater runs the simulation but lies about which circuit they ran?**

The circuit parameters are committed in the Merkle tree. The verifier re-runs with those exact parameters. If the cheater changed the circuit (to run an easier one), the re-run produces different statistics.

**Edge case: What if the cheater pre-computes results for easy circuits and claims they were hard ones?**

The circuit selection is deterministic from the problem contract + block hash + solver address. The solver can't choose which circuit to run. It's assigned by the protocol, like how Bitcoin miners can't choose an easy hash target.

---

## Component 3: Optimistic Rollup for Quantum Work

**This is the key speed unlock.**

Instead of waiting for consensus before accepting results, use optimistic execution:

1. **Solver submits:** Merkle root + best solution found + claimed energy value
2. **Immediate provisional acceptance:** The result is treated as valid immediately
3. **Challenge window opens:** Any node has 5 minutes to challenge
4. **If no challenge:** Result finalizes. Payment settles. Done.
5. **If challenged:** On-chain dispute resolution triggers:
   - Challenger picks random leaf indices
   - Solver must reveal those Merkle leaves
   - On-chain contract re-verifies the samples
   - Loser pays gas costs + gets slashed

**Why optimistic works here:**

The economics make cheating irrational:
- Cheater must stake tokens to participate (slashed if caught)
- Challenger earns a portion of the cheater's stake if fraud is proven
- The statistical detection rate makes fraud extremely risky
- Even a small chance of being checked makes the expected value of cheating negative

**Speed achieved:**
- Solver gets provisional credit in ~500ms (time to publish Merkle root + peer spot-check)
- Final settlement in 5 minutes if unchallenged
- Dispute resolution in ~30 seconds (a few on-chain transactions)

Compare to Bitcoin: 10 minutes per block, 6 confirmations for finality = 60 minutes. This is 120x faster to provisional acceptance and 12x faster to finality.

---

## Component 4: DAG Consensus Instead of Linear Chain

**Problem:** Linear blockchain (one block at a time) creates a bottleneck. Many solvers produce results simultaneously. Why serialize them?

**Solution: Directed Acyclic Graph (DAG)**

Instead of blocks in a chain, each quantum work unit is its own "micro-transaction" that references 2-3 previous transactions. No mining, no block intervals. Work units flow continuously.

```
    [work-A] ← [work-D] ← [work-G]
        ↑          ↑          ↑
    [work-B] ← [work-E] ← [work-H]
        ↑          ↑
    [work-C] ← [work-F]
```

Each work unit contains:
- Solver's Merkle root for a batch of samples
- References to 2-3 previous work units (provides ordering)
- Cumulative solution quality (best energy found so far)
- Statistical validity attestations from peer nodes

**Consensus emerges from accumulation:** A work unit is "confirmed" when enough subsequent work units reference it (directly or transitively). No explicit voting. No block production delay. Throughput scales with number of active solvers.

**This is similar to IOTA's Tangle but purpose-built for quantum work verification.**

---

## Component 5: The Verification Circuit (ZK Approach)

**For maximum speed and minimum on-chain cost, wrap the whole thing in a ZK proof.**

The claim: "I correctly simulated quantum circuit C for n strikes and the best solution I found has energy E."

The ZK proof proves:
1. Each gate application was a valid unitary operation
2. Measurements were sampled according to Born rule
3. The reported energy E corresponds to an actual measurement outcome
4. The Merkle root correctly commits to all samples

**Why quantum circuits are GOOD candidates for ZK proofs:**

Quantum gates are small, structured matrices (2x2 or 4x4). Applying them is a sequence of matrix multiplications. This is exactly the kind of structured, repetitive arithmetic that ZK-SNARKs handle efficiently. The circuit for "verify n applications of 2x2 unitary matrices" is:

- Highly regular (same gate structure repeated)
- Bounded depth (shallow circuits = short ZK proofs)
- Algebraically clean (complex number arithmetic maps to field arithmetic)

**Prover cost:** O(n * q^2) where n = number of strikes, q = number of qubits. For 10,000 strikes on a 15-qubit system, this is ~2.25 billion field operations. With modern ZK provers (Plonky2, Halo2), this generates a proof in 10-60 seconds on a GPU.

**Verifier cost:** O(log n) regardless of circuit size. Verification takes ~50ms on-chain. One on-chain transaction verifies 10,000 quantum simulation runs.

**Proof size:** ~200-500 KB. Cheap to store and transmit.

---

## Component 6: Cross-Validation as Free Verification

**The elegant trick: when multiple solvers work on the same problem, they verify each other for free.**

If 5 solvers independently run match-strike on the same Hamiltonian, their sample distributions should converge to the same Born-rule distribution. Any solver whose distribution significantly deviates from the group is either:
- Cheating (submitting fake samples)
- Buggy (incorrect simulation)
- Extremely unlucky (statistically improbable with enough samples)

**Implementation:**

```
Solver A submits: distribution D_A over 10,000 samples
Solver B submits: distribution D_B over 10,000 samples  
Solver C submits: distribution D_C over 10,000 samples

Pairwise KL-divergence:
  KL(D_A || D_B) = 0.003  (very similar: both honest)
  KL(D_A || D_C) = 0.002  (very similar: both honest)  
  KL(D_B || D_C) = 0.004  (very similar: both honest)

All three distributions agree → high confidence all are honest
No explicit verification needed. Consensus emerges from agreement.
```

**If one solver deviates:**

```
KL(D_A || D_B) = 0.003
KL(D_A || D_C) = 0.847  ← outlier
KL(D_B || D_C) = 0.831  ← outlier

C disagrees with both A and B → C is flagged
Trigger explicit verification of C's Merkle leaves
```

**Why this is fast:** KL-divergence computation is O(2^q) where q is qubit count. For 15 qubits, that's 32,768 operations. Microseconds. And it's computed at the coordination layer (Speed 2), not on-chain.

---

## How It All Fits Together

```
SOLVER NODE (Speed 1: microseconds)
├── Runs match-strike cycle continuously
├── Accumulates samples in local buffer
├── Every 100ms: builds Merkle tree, computes root
│
├── PEER VERIFICATION (Speed 2: milliseconds)
│   ├── Publishes Merkle root to peer network
│   ├── 2-3 peers spot-check random leaves
│   ├── Cross-validation against other solvers' distributions
│   ├── Statistical validity test (Born rule compliance)
│   └── Provisional acceptance: ~500ms
│
├── DAG COMMITMENT (Speed 2-3: seconds)
│   ├── Work unit published to DAG
│   ├── References 2-3 previous work units
│   ├── Accumulates confirmations as more units reference it
│   └── Soft finality: ~5 seconds
│
└── ON-CHAIN SETTLEMENT (Speed 3: minutes)
    ├── Optimistic rollup: accepted unless challenged
    ├── ZK proof available for instant on-chain verification
    ├── Challenge window: 5 minutes
    ├── Payment settles after finality
    └── Hard finality: ~5 minutes
```

**Net result:**
- Solver NEVER stops computing to wait for verification
- Provisional results available in 500ms
- Soft finality in 5 seconds
- Hard finality in 5 minutes
- On-chain cost: one verification transaction per batch of 10,000+ samples

---

## The Numbers

| Metric | Bitcoin | Ethereum L2 | This System |
|---|---|---|---|
| Time to first confirmation | 10 min | ~2 sec | ~500ms |
| Hard finality | 60 min | ~15 min | 5 min |
| Useful work per block | 0 | 0 | 10,000+ quantum sim runs |
| On-chain verification cost | N/A | ~50K gas | ~200K gas (ZK verify) |
| Throughput (tx/sec) | 7 | ~4,000 | Limited by solver count |
| Energy wasted on consensus | ~150 TWh/yr | ~0.01 TWh/yr | 0 (all energy does useful work) |

---

## Implementation Plan

### Step 1: Merkle Commitment Library (Week 1)

Build the sample commitment scheme:
- `merkle.py`: Build Merkle trees from quantum samples
- `commit.py`: Solver-side batch commitment
- `verify.py`: Verifier-side leaf spot-checking
- `stats.py`: Born rule compliance testing (chi-squared)

Benchmark: time to build tree for 10K/100K/1M samples, time to verify 50/100/500 random leaves.

### Step 2: Peer Verification Network (Week 2-3)

Build the Speed 2 layer:
- `peer_network.py`: libp2p-based gossip for Merkle roots
- `spot_checker.py`: Automated random leaf verification
- `cross_validator.py`: KL-divergence between solver distributions
- `reputation.py`: Track solver reliability scores

Benchmark: end-to-end time from solver commitment to provisional acceptance across 10-50 nodes.

### Step 3: DAG Consensus (Week 4-5)

Build the DAG layer:
- `dag.py`: Work unit creation, parent selection, traversal
- `confirmation.py`: Confirmation depth tracking
- `conflict_resolution.py`: Handle contradictory work units
- `finality.py`: Determine when a work unit is final

Benchmark: throughput (work units per second), time to finality, behavior under adversarial conditions (20% dishonest nodes).

### Step 4: ZK Verification Circuit (Week 6-8)

Build the on-chain verification:
- `circuit.circom`: ZK circuit for quantum simulation verification
  - Unitary gate application (2x2 complex matrix multiply in field arithmetic)
  - Born rule sampling verification
  - Merkle root computation
- `prover.py`: Generate ZK proof from solver's computation trace
- `verifier.sol`: On-chain Solidity verifier contract

Benchmark: proof generation time (target: <60 seconds for 10K strikes on GPU), proof size (target: <500KB), on-chain verification gas cost (target: <300K gas).

### Step 5: Optimistic Rollup Integration (Week 9-10)

Connect everything:
- `rollup.sol`: Optimistic rollup contract with challenge mechanism
- `challenger.py`: Automated fraud detection and challenge submission
- `settlement.sol`: Payment settlement after finality
- Integration tests: full end-to-end flow from problem submission to payment

### Step 6: Load Testing and Security Audit (Week 11-12)

- Simulate 100+ solver nodes
- Adversarial testing: 10%, 20%, 33% dishonest nodes
- Measure: false positive rate, false negative rate, time to detect fraud
- Economic simulation: is cheating profitable under any parameter choice?

---

## What Makes This Different From Every Other "Blockchain + X" Project

Most proof-of-useful-work projects bolt arbitrary computation onto blockchain consensus. The useful work and the consensus mechanism are separate things awkwardly married.

Here, the verification IS the consensus. The statistical properties of quantum mechanics provide the asymmetry (hard to generate, easy to verify) that consensus protocols need. The Born rule is doing double duty: it governs the physics AND provides the verification scheme. The math that makes quantum simulation correct is the same math that makes the consensus secure.

You're not adding quantum simulation to a blockchain. You're using the statistical structure of quantum mechanics as the foundation of a new consensus mechanism.

That's the difference between "blockchain for X" and "X as blockchain."

---

## Risk Factors

1. **ZK proof generation might be too slow.** If proving 10K strikes takes 10 minutes instead of 60 seconds, the ZK path becomes impractical. Fallback: rely on statistical spot-checking without ZK. Still works, just less elegant on-chain.

2. **Cross-validation fails for unique problems.** If only one solver works on a problem, there's no cross-validation. Fallback: rely on Merkle spot-checking and Born rule compliance. Still works, just requires more explicit verification.

3. **Economic attack: solver collusion.** If all solvers on a problem collude to submit the same fake results, cross-validation won't catch them. Mitigation: require diverse solver sets (different geographic regions, different staking pools). Also: the problem submitter can independently verify the final solution by running it once.

4. **Quantum simulation might not map cleanly to ZK arithmetic circuits.** Complex number arithmetic in finite fields has precision issues. Mitigation: use fixed-point representation with sufficient precision, or abandon ZK and rely on statistical verification alone.