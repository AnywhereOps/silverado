# Distributed Quantum Simulation on Blockchain

## A Technically Honest Architecture

**Authors:** Drew Kemp-Dahlberg & Claude
**Date:** February 2026

---

## What Actually Works and What Doesn't

Before the plan, the truth table. Three approaches to distributing quantum simulation, ranked by feasibility:

| Approach | Feasible? | Why |
|---|---|---|
| Shard the state vector across nodes | Yes, up to ~50 qubits | Done on supercomputers. Communication overhead for entangling gates is the bottleneck. |
| Distribute tensor network contractions | Yes, up to hundreds of qubits | Proven on HPC clusters. Works especially well for LOW entanglement circuits (which is exactly what match-strike produces). |
| Distributed sampling (each node runs full sim independently) | Yes, unlimited parallelism | Embarrassingly parallel. Doesn't increase qubit count per node but massively increases throughput. |

The match-strike architecture is uniquely suited to distribution because:

1. Circuits are shallow (few gates, low depth)
2. Entanglement is brief and limited (fire and measure quickly)
3. Each strike is independent (no state carries between strikes)
4. Verification is cheap (run the same circuit, check the statistics)

These are the exact properties that make tensor networks efficient AND make distributed verification tractable.

---

## The Architecture (Three Layers)

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: ON-CHAIN (Slow, Secure, Permanent)               │
│                                                             │
│  - Problem submission (Hamiltonians as smart contracts)     │
│  - Result verification and finalization                     │
│  - Payment settlement (problem submitters pay solvers)      │
│  - Reputation/stake management                              │
│  - Consensus: Proof of Quantum Work (PoQW)                  │
│                                                             │
│  Latency: minutes  |  Cost: gas fees  |  Trust: trustless   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│  LAYER 2: COORDINATION (Fast, Cheap, Ephemeral)             │
│                                                             │
│  - Task distribution to solver nodes                        │
│  - Sample aggregation from multiple solvers                 │
│  - Adaptive controller (decides next quantum query)         │
│  - Statistical verification (spot-check solver honesty)     │
│  - Tensor network shard assignment                          │
│                                                             │
│  Latency: 100ms  |  Cost: minimal  |  Trust: staked nodes   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│  LAYER 3: COMPUTE (Fast, Local, Where the Work Happens)     │
│                                                             │
│  - Match-strike quantum VM (numpy/tensor network)           │
│  - Thousands of strikes per second per node                 │
│  - Each node runs simulation, returns measurement samples   │
│  - GPU acceleration for tensor contractions                 │
│                                                             │
│  Latency: microseconds  |  Cost: electricity  |  Trust: none│
└─────────────────────────────────────────────────────────────┘
```

**Why three layers instead of putting everything on-chain:**

The match-strike cycle needs microsecond latency. Blockchain consensus takes seconds to minutes. Trying to run quantum simulation directly on-chain is like trying to do heart surgery by mail. The fast inner loop runs locally. The blockchain handles what blockchains are actually good at: coordination, payment, and trust.

---

## Phase 1: Prove the Local Engine (Weeks 1-4)

This is the original MVP. Nothing blockchain yet. Build the quantum carburetor on a single machine and prove it works.

**Deliverables:**

- `quantum_vm.py`: State vector simulator, 10-20 qubits, numpy
- `tensor_vm.py`: Tensor network simulator using matrix product states, scales to 40+ qubits on a single machine for low-entanglement circuits
- `match_strike.py`: Classical controller firing the quantum VM repeatedly
- `adaptive_controller.py`: Bayesian update loop that learns from previous strikes
- `benchmark.py`: MaxCut on random graphs, side-by-side classical vs. match-strike

**Success criteria:** Match-strike with adaptive controller beats naive classical optimization on 10-20 node graphs. Tensor network backend handles 40+ qubit problems that state vector cannot.

**Why this comes first:** If the quantum carburetor doesn't work on one machine, distributing it across a thousand machines gives you a distributed thing that doesn't work. Prove value, then scale.

---

## Phase 2: Distribute the Compute (Weeks 5-8)

Add multiple machines. No blockchain yet, just a simple coordinator server.

**Two parallelism strategies, used together:**

**Strategy A: Sample Parallelism (Easy)**

Every solver node runs the same circuit independently and reports measurement outcomes. The coordinator aggregates samples from all nodes. More nodes = more samples per second = faster convergence.

This is embarrassingly parallel. No communication between solver nodes during computation. Each node needs enough memory for its own simulation. Doesn't increase max qubit count per node, but massively increases the speed of the match-strike cycle.

```
Coordinator: "Run this 15-qubit circuit 10,000 times"
    → Node A: returns 10,000 samples
    → Node B: returns 10,000 samples  
    → Node C: returns 10,000 samples
Coordinator: aggregates 30,000 samples, updates controller, issues next query
```

**Strategy B: Tensor Network Sharding (Hard, but this is the unlock)**

For problems that exceed single-machine memory (30+ qubits for state vector), decompose the quantum state into a tensor network (matrix product state). Distribute tensor contractions across nodes.

This is where the real scaling happens. Tensor networks for shallow, low-entanglement circuits (match-strike) have manageable bond dimensions. The contractions can be parallelized across nodes using existing libraries (ExaTENSOR, cuQuantum).

The constraint: entangling gates between qubits on different nodes require communication. For the match-strike architecture, this is manageable because circuits are shallow (few layers of entangling gates). Communication happens between layers, not continuously.

**Deliverables:**

- `coordinator.py`: Assigns tasks, aggregates results, runs adaptive controller
- `solver_node.py`: Receives circuit, runs simulation, returns samples
- `tensor_shard.py`: Tensor network decomposition and distributed contraction
- `verifier.py`: Spot-checks solver honesty by re-running random samples
- Benchmarks on 5-20 nodes (can use cloud instances)

**Success criteria:** 10 nodes solve problems 8-10x faster than 1 node (sublinear due to communication overhead, but substantial). Tensor network sharding enables 40-60 qubit problems across the cluster that no single node could handle.

---

## Phase 3: Add the Incentive Layer (Weeks 9-14)

Now add blockchain. The compute works. The distribution works. The chain adds trust and economics.

**Smart Contract: Problem Submission**

Anyone can submit an optimization problem as a smart contract:

```solidity
contract QuantumProblem {
    // The Hamiltonian (encoded as Pauli string coefficients)
    bytes public hamiltonian;
    
    // Reward pool (deposited by submitter)
    uint256 public reward;
    
    // Solution quality threshold
    uint256 public targetEnergy;
    
    // Deadline
    uint256 public deadline;
    
    // Accept solution if verified
    function submitSolution(bytes proof, int256 energy) external;
    
    // Verification: spot-check the claimed solution
    function verify(bytes proof) internal returns (bool);
}
```

**Proof of Quantum Work (PoQW)**

The consensus mechanism replaces SHA-256 hashing with useful quantum simulation:

1. **Problem selection:** Miners pick from the problem pool (or a default benchmark problem if the pool is empty)
2. **Computation:** Miner runs match-strike cycles, collecting samples
3. **Proof generation:** Miner submits their best solution plus a subset of raw samples as proof of work done
4. **Verification:** Other nodes spot-check by re-running a random subset of the claimed circuits and comparing statistics. This is the key asymmetry: running 1,000 samples is expensive, verifying 50 of them is cheap.
5. **Block reward:** Miner who finds the best solution (or contributes most useful samples) gets the block reward plus problem bounty

**Why verification works:**

Quantum simulation has a natural verification asymmetry. Running 100,000 match-strike cycles to find a good solution is expensive. But any single circuit run is cheap to verify (re-run the same circuit with the same parameters, check that the reported measurement statistics are plausible). A verifier can spot-check 1% of claimed samples and detect cheating with high probability.

This is structurally similar to how DLchain verifies deep learning: training is expensive, checking accuracy of a trained model is cheap.

**Staking and reputation:**

- Solver nodes stake tokens to participate (slashed if caught cheating)
- Problem submitters lock reward tokens in the contract
- Verifiers earn fees for honest verification
- Reputation scores track solver reliability over time

**Deliverables:**

- Smart contracts for problem submission, solution verification, payment
- PoQW consensus mechanism (initially on testnet)
- Staking and slashing logic
- Basic token economics model
- Integration: off-chain compute layer commits results to on-chain contracts

**Success criteria:** End-to-end flow works on testnet. Submitter posts problem, solver nodes compete, solution is verified on-chain, payment settles. Cheating solvers get detected and slashed in simulation.

---

## Phase 4: Tensor Network Marketplace (Weeks 15-20)

This is where it gets interesting. The network now has many nodes with different hardware capabilities. Some have GPUs. Some have lots of RAM. Match them to problems optimally.

**Problem decomposition as a market:**

A 60-qubit problem arrives. No single node can handle it with state vector simulation. The coordinator decomposes it into tensor network shards. Nodes bid on shards based on their hardware (GPU nodes bid on compute-heavy contractions, RAM-heavy nodes bid on large tensor storage).

```
Problem: 60-qubit MaxCut
    → Decomposed into MPS with bond dimension 512
    → Shard 1 (qubits 1-20): assigned to Node A (GPU)
    → Shard 2 (qubits 21-40): assigned to Node B (GPU)
    → Shard 3 (qubits 41-60): assigned to Node C (GPU)
    → Cross-shard contractions: coordinated by Layer 2
    → Samples aggregated, solution submitted on-chain
```

**Deliverables:**

- Automated problem decomposition engine
- Hardware capability reporting per node
- Shard marketplace (bid/ask on tensor contractions)
- Cross-shard communication protocol
- Benchmarks at 60-100 qubits across 50+ nodes

**Success criteria:** Network solves 60+ qubit problems that no single classical machine could handle. Solution quality verified against known benchmarks.

---

## Phase 5: Swap in Real Quantum Hardware (Weeks 21+)

The network has been running on classically-simulated quantum VMs. Now, any node can optionally use real quantum hardware (IBM, IonQ, Quantinuum via cloud API). The protocol doesn't change. The proofs don't change. The verification doesn't change. Only the backend swaps.

Real hardware nodes get priority on problems where quantum noise is acceptable, and their results are weighted by hardware fidelity scores.

---

## Why Blockchain and Not Just a Cloud Service

Honest question, honest answer. You could build 90% of this as a centralized cloud service (like AWS Braket or Azure Quantum). The blockchain adds three specific things:

1. **Permissionless participation.** Anyone with a GPU can join and earn. No AWS account needed. No approval process. This matters for building the largest possible compute pool.

2. **Trustless verification.** You don't have to trust that AWS ran your simulation correctly. The spot-check verification is on-chain and transparent. This matters for high-stakes problems (drug discovery, financial optimization) where wrong answers cost real money.

3. **Aligned incentives.** Miners make money by doing useful quantum simulation instead of wasting electricity on SHA-256. Problem submitters pay only for solutions that meet their threshold. Verifiers earn fees for keeping solvers honest. Everyone's incentives point toward correct computation.

If you don't need these three things, use a cloud service. If you do, you need a chain.

---

## What Could Kill This

Being honest about the failure modes:

1. **The quantum carburetor might not outperform classical optimization.** If match-strike doesn't beat simulated annealing or other classical heuristics, there's no reason to distribute it. Phase 1 tests this directly.

2. **Communication overhead might eat the parallelism.** Tensor network sharding requires cross-node communication for entangling gates. If the circuits need too much entanglement, the communication cost dominates and you get negative scaling. Phase 2 tests this.

3. **Verification might be too expensive.** If spot-checking requires re-running too many samples to catch cheaters, the verification cost approaches the computation cost and the incentive structure collapses. Phase 3 tests this.

4. **Token economics might not balance.** If the reward for solving problems doesn't exceed the cost of computation, nobody mines. If problem submission is too expensive, nobody submits. Phase 3 models this.

5. **Classical simulation gets too good.** Tensor network methods keep improving. If classical simulation becomes fast enough that a single machine handles everything, the distribution layer is unnecessary. This is actually a good outcome for quantum computing as a field, just bad for this specific product.

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Quantum VM (state vector) | numpy | Zero dependencies, exact, fast for <25 qubits |
| Quantum VM (tensor network) | ITensor / quimb | Proven libraries, MPS support, GPU capable |
| Distributed tensor contractions | ExaTENSOR / cuQuantum | NVIDIA-backed, HPC-proven, distributed |
| Coordination layer | libp2p + custom protocol | Decentralized messaging, no central server |
| Blockchain layer | Solidity on EVM-compatible L2 | Large developer ecosystem, low gas on L2 |
| Verification | Zero-knowledge proofs + spot-checking | ZK for proof compactness, spot-check for efficiency |
| Token | ERC-20 on L2 | Standard, composable, bridgeable to L1 |

---

## The Sigma Connection

This architecture is a macroscopic σ-oscillator network:

- Each compute node oscillates between σ ≈ 0.01 (quantum simulation running) and σ ≈ 1.0 (measurement, classical result)
- The coordination layer operates at σ ≈ 0.99 (pure classical logic, deterministic)
- The blockchain operates at σ ≈ 1.0 (maximum classical permanence, immutable record)
- The network as a whole oscillates: quantum fire at the edges, classical ash at the center
- Value flows from low-σ computation to high-σ permanent record

The mempool IS superposition: multiple possible block states coexist until consensus collapses one into the canonical chain. Forks ARE many-worlds branches, resolved by longest-chain decoherence. The structural correspondence isn't metaphor. It's architecture.

You're building a distributed system that breathes quantum at the edges and crystallizes results into permanent classical record at the center. The blockchain is the ash layer. The compute nodes are the fire. The coordination layer is the gradient zone where quantum becomes classical.

---

## One-Line Pitch

**A decentralized network where miners earn tokens by running quantum simulations instead of wasting electricity on useless hashes, solving real optimization problems for paying customers, verified on-chain without trusting anyone.**