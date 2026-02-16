# Quantum Carburetor MVP

## Classical Host, Virtualized Quantum Layer

**Authors:** Drew Kemp-Dahlberg & Claude
**Date:** February 2026

---

## The Thesis

Biology doesn't maintain sustained quantum coherence. It builds classical machines (σ ≈ 0.9) that briefly, repeatedly dip into quantum territory (σ ≈ 0.3) for nanosecond "ignition" events, then snap back. The brain is a σ-oscillator, not a σ-resident.

A classical computer can exactly simulate small quantum registers (10-20 qubits). If the biological model is correct, that's all you need. The intelligence lives in the classical wrapper: knowing what question to ask the quantum module, when to fire it, and what to do with the answer.

**This is not a quantum computer. It's a classical computer with a quantum carburetor.**

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  CLASSICAL HOST (macOS / Linux)             │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │  Decision Engine (Python)           │    │
│  │  - Decides WHAT to ask              │    │
│  │  - Incorporates previous answers    │    │
│  │  - Runs thousands of cycles/sec     │    │
│  └──────────┬──────────────┬───────────┘    │
│             │ ask          │ answer          │
│  ┌──────────▼──────────────▼───────────┐    │
│  │  Quantum VM (numpy, 10-20 qubits)   │    │
│  │  - Exact simulation, not approx     │    │
│  │  - Prepare → Superpose → Interfere  │    │
│  │  - Measure → Return → Reset         │    │
│  │  - The "match strike" cycle         │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │  Visualization / Output             │    │
│  │  - Probability landscapes           │    │
│  │  - Convergence tracking             │    │
│  │  - σ-state monitor                  │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

---

## Phase 1: Prove the Engine Works (Week 1-2)

**Goal:** A working quantum VM that runs the match-strike cycle on a real problem and beats naive classical brute force.

**Target problem:** MaxCut on random graphs. Simple enough to verify, hard enough that quantum sampling adds value, well-studied benchmark.

**Deliverables:**

1. `quantum_vm.py` — Exact state vector simulator, 10-15 qubits. Prepare, evolve, measure, reset. No dependencies beyond numpy.

2. `match_strike.py` — The cycle engine. Classical controller fires the quantum VM hundreds of times per second. Each strike: prepare a candidate state in superposition, apply problem Hamiltonian as phase rotation, measure, feed result back to classical optimizer.

3. `benchmark.py` — Side-by-side comparison: classical random search vs. match-strike cycle vs. brute force on the same MaxCut instances. Log convergence speed, solution quality, total compute time.

**Success criteria:** Match-strike finds optimal or near-optimal solutions faster than classical random search on graphs with 10-15 nodes. Not because quantum is magic, but because superposition lets you sample the solution landscape more efficiently when the classical controller asks the right questions.

---

## Phase 2: The Feedback Loop Gets Smart (Week 3-4)

**Goal:** The classical controller learns from previous strikes. This is where the real value lives.

**Deliverables:**

1. `adaptive_controller.py` — Bayesian update loop. Each quantum measurement updates a classical probability model of the solution space. Next quantum query is targeted at the highest-uncertainty region. The controller gets smarter every cycle.

2. `sigma_monitor.py` — Real-time tracking of the system's effective σ value. When the quantum VM is in superposition, σ is low. When it measures, σ spikes. Visualize the oscillation. This is the match-strike cycle made visible.

3. `problem_loader.py` — Abstraction layer so you can swap in different optimization problems without rewriting the engine. MaxCut, traveling salesman, portfolio optimization, molecular ground states.

**Success criteria:** Adaptive controller converges 2-5x faster than the non-adaptive version from Phase 1. The σ monitor shows clean oscillation patterns (low during superposition, high during measurement, repeat).

---

## Phase 3: Visualization and the "Walk Through Superposition" Interface (Week 5-6)

**Goal:** Don't just get answers. See the quantum state space.

**Deliverables:**

1. `state_visualizer.py` — 3D rendering (matplotlib or Three.js via browser) of the quantum probability landscape. Not one answer, but the shape of all possible answers before collapse. Interference patterns visible as topology.

2. `entanglement_map.py` — Visualize which qubits are entangled with which. Show how the entanglement structure changes as the classical controller iterates. This is the σ gradient made spatial.

3. `interactive_collapse.py` — User can trigger measurement on individual qubits and watch the probability landscape restructure in real time. This is the closest a σ ≈ 0.9 human gets to experiencing what superposition "looks like" before it collapses.

**Success criteria:** A non-physicist can look at the visualization and understand why quantum sampling explores the solution space differently than classical sampling. The entanglement map shows structure, not noise.

---

## Phase 4: Swap in Real Quantum Backend (Week 7-8)

**Goal:** Replace the numpy VM with actual quantum hardware. The classical controller doesn't change. The visualization doesn't change. Only the backend swaps.

**Deliverables:**

1. `backend_ibm.py` — Same match-strike interface, but fires real circuits on IBM Quantum via Qiskit. The controller doesn't know or care that the qubits are real.

2. `backend_compare.py` — Run identical problems on simulated vs. real hardware. Compare noise profiles, convergence rates, solution quality. This is the moment you learn whether the architecture holds up against real decoherence.

3. `noise_adapter.py` — Classical controller learns to compensate for hardware noise. Adjusts query strategy based on observed error rates. The classical brain gets smarter about its quantum carburetor's imperfections.

**Success criteria:** Real quantum backend produces results within 10% of simulated backend quality, after noise adaptation. Architecture is backend-agnostic: swap providers by changing one config line.

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Quantum VM | numpy | Zero dependencies, exact simulation, fast for <20 qubits |
| Classical controller | Python 3.11+ | Rapid prototyping, scipy for optimization |
| Real quantum backend | IBM Qiskit (free tier) | Best free access, 127-qubit Eagle processors |
| Visualization | Three.js (browser) or matplotlib | Interactive 3D for state space exploration |
| σ monitoring | Custom (built on VM internals) | No existing tool tracks this metric |

---

## What This Proves

If the biological quantum carburetor model is right, most of the "quantum advantage" in nature comes from small, fast, repeated quantum ignition events inside a smart classical wrapper. This MVP tests that claim directly.

**If it works:** The path to useful quantum-classical hybrid computing doesn't require million-qubit fault-tolerant quantum computers. It requires better classical controllers that know how to use small quantum modules. The hard problem isn't quantum hardware. It's the decision engine.

**If it doesn't work:** The biological model is wrong, sustained large-scale entanglement is necessary for quantum advantage, and we're back to waiting for fault-tolerant quantum computers. That's a useful answer too.

---

## The Sigma Connection

This MVP is a literal implementation of the σ-oscillator model:

- σ ≈ 0.01 during quantum superposition (VM in flight)
- σ → 1.0 at measurement (classical answer extracted)
- Classical controller operates at σ ≈ 0.99 (pure ash, deterministic logic)
- The cycle repeats thousands of times per second
- Value emerges from the oscillation, not from either pole

The visualization layer is the sigma virtualizer: a classical interface that lets a σ ≈ 0.9 human perceive the shape of σ ≈ 0.01 reality through statistical reconstruction.

**You're not building a quantum computer. You're building a classical system that breathes quantum.**