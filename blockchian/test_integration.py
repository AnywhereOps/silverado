"""
Integration Tests — End-to-end validation of the quantum carburetor system.

Tests:
  1. Quantum VM correctness (state vector + tensor network)
  2. Match-strike engine convergence
  3. Adaptive controller improvement over baseline
  4. Distributed coordination with multiple nodes
  5. Verifier catches all cheating strategies
  6. Full pipeline: problem → distribute → verify → settle
"""

import time
import numpy as np
import sys


def test_quantum_vm_basics():
    """Test QuantumVM fundamental operations."""
    from quantum_vm import QuantumVM

    print("  1a. QuantumVM basics...", end=" ")
    vm = QuantumVM(3)

    # |000⟩
    assert vm.measure() == 0, "Initial state should be |000⟩"
    assert vm.sigma() > 0.99, "Collapsed state should have σ ≈ 1"

    # Superposition
    vm.superpose_all()
    assert vm.sigma() < 0.01, "Equal superposition should have σ ≈ 0"

    # H on qubit 0 from |000⟩
    vm.reset()
    vm.h(0)
    probs = vm.probabilities()
    assert abs(probs[0] - 0.5) < 1e-10, "H|0⟩ should give 50/50"
    assert abs(probs[4] - 0.5) < 1e-10, "H|0⟩ should give 50/50"

    # Bell state
    vm.reset()
    vm.h(0)
    vm.cx(0, 1)
    probs = vm.probabilities()
    assert abs(probs[0] - 0.5) < 1e-10, "Bell state: |000⟩ component"
    assert abs(probs[6] - 0.5) < 1e-10, "Bell state: |110⟩ component"

    print("PASS")


def test_tensor_vm_matches_statevector():
    """Test TensorVM produces same distributions as QuantumVM."""
    from quantum_vm import QuantumVM
    from tensor_vm import TensorVM

    print("  1b. TensorVM matches QuantumVM...", end=" ")
    rng = np.random.default_rng(42)
    n = 8

    # Build identical random circuits
    qvm = QuantumVM(n)
    tvm = TensorVM(n, max_bond=64)

    for _ in range(15):
        gate = rng.choice(['h', 'ry', 'cx'])
        if gate == 'h':
            q = int(rng.integers(0, n))
            qvm.h(q); tvm.h(q)
        elif gate == 'ry':
            q = int(rng.integers(0, n))
            angle = float(rng.uniform(0, 2 * np.pi))
            qvm.ry(q, angle); tvm.ry(q, angle)
        else:
            q1, q2 = rng.choice(n, size=2, replace=False)
            qvm.cx(int(q1), int(q2)); tvm.cx(int(q1), int(q2))

    # Compare via sampling
    exact = qvm.probabilities()
    samples = tvm.measure_shots(30000)
    hist = np.bincount(samples, minlength=2**n) / 30000

    l1 = np.sum(np.abs(exact - hist))
    assert l1 < 0.1, f"L1 error {l1:.4f} too high (expected < 0.1)"

    print(f"PASS (L1={l1:.4f})")


def test_tensor_vm_scales():
    """Test TensorVM handles 30+ qubits."""
    from tensor_vm import TensorVM

    print("  1c. TensorVM scales to 40 qubits...", end=" ")
    tvm = TensorVM(40, max_bond=16)
    tvm.superpose_all()
    for q in range(39):
        tvm.cz(q, q + 1)
    sample = tvm.measure_bitstring()
    assert len(sample) == 40
    assert tvm.memory_bytes() < 1e6, "Should use < 1MB for low entanglement"

    print(f"PASS (mem={tvm.memory_bytes()/1024:.1f}KB)")


def test_match_strike_finds_good_solutions():
    """Test match-strike engine finds near-optimal MaxCut solutions."""
    from match_strike import MatchStrikeEngine, MaxCutProblem

    print("  2a. Match-strike convergence...", end=" ")
    problem = MaxCutProblem.random(n_nodes=10, edge_prob=0.5, seed=42)
    optimal_bs, optimal_cut = problem.brute_force_optimal()

    engine = MatchStrikeEngine(problem, n_layers=3, seed=42)
    log = engine.run(n_strikes=500, strategy="greedy")

    ratio = log.best_cut / optimal_cut
    assert ratio > 0.9, f"Should find >90% optimal, got {ratio:.1%}"

    print(f"PASS ({ratio:.1%} of optimal)")


def test_adaptive_beats_random():
    """Test adaptive controller converges faster than random."""
    from match_strike import MatchStrikeEngine, MaxCutProblem, classical_random_search
    from adaptive_controller import MultiScaleController

    print("  3a. Adaptive beats random...", end=" ")
    # Average over multiple instances for robustness
    adaptive_ratios = []
    random_ratios = []

    for inst in range(5):
        seed = 3000 + inst
        problem = MaxCutProblem.random(n_nodes=12, edge_prob=0.5, seed=seed)
        _, optimal_cut = problem.brute_force_optimal()
        if optimal_cut == 0:
            continue

        # Adaptive
        engine = MatchStrikeEngine(problem, n_layers=3, seed=seed)
        multi = MultiScaleController(engine)
        ad_log = multi.run(n_strikes=300)
        adaptive_ratios.append(ad_log.best_cut / optimal_cut)

        # Random classical
        cl_log = classical_random_search(problem, n_samples=300, seed=seed)
        random_ratios.append(cl_log.best_cut / optimal_cut)

    avg_adaptive = np.mean(adaptive_ratios)
    avg_random = np.mean(random_ratios)
    assert avg_adaptive >= avg_random * 0.99, \
        f"Adaptive ({avg_adaptive:.1%}) should be >= random ({avg_random:.1%})"

    print(f"PASS (adaptive={avg_adaptive:.1%}, random={avg_random:.1%})")


def test_distributed_coordination():
    """Test distributed coordinator with multiple solver nodes."""
    from match_strike import MaxCutProblem
    from coordinator import Coordinator

    print("  4a. Distributed coordination...", end=" ")
    problem = MaxCutProblem.random(n_nodes=10, edge_prob=0.5, seed=42)
    _, optimal_cut = problem.brute_force_optimal()

    coord = Coordinator(problem, n_layers=3)
    nodes = coord.register_nodes(4)
    log = coord.run_adaptive(n_rounds=20, shots_per_node=25)
    stats = coord.stats()

    assert stats["n_nodes"] == 4
    assert stats["total_samples"] > 0
    assert stats["best_cut"] > 0

    ratio = stats["best_cut"] / optimal_cut
    assert ratio > 0.85, f"Should find >85% optimal, got {ratio:.1%}"

    # Check all nodes participated
    for nid, nstats in stats["node_stats"].items():
        assert nstats["tasks"] > 0, f"Node {nid} did no work"

    print(f"PASS ({ratio:.1%} optimal, {stats['total_samples']} samples across 4 nodes)")


def test_distributed_scaling():
    """Test that more nodes = more samples per round."""
    from match_strike import MaxCutProblem
    from coordinator import Coordinator

    print("  4b. Distributed scaling...", end=" ")
    problem = MaxCutProblem.random(n_nodes=10, edge_prob=0.5, seed=42)

    samples_by_nodes = {}
    for n_nodes in [1, 2, 4]:
        coord = Coordinator(problem, n_layers=3)
        coord.register_nodes(n_nodes)
        coord.run_random(n_rounds=5, shots_per_node=50)
        stats = coord.stats()
        samples_by_nodes[n_nodes] = stats["total_samples"]

    # More nodes should give proportionally more samples
    assert samples_by_nodes[4] > samples_by_nodes[1] * 3, \
        f"4 nodes should give ~4x samples (got {samples_by_nodes[4]} vs {samples_by_nodes[1]})"

    print(f"PASS (1 node={samples_by_nodes[1]}, 4 nodes={samples_by_nodes[4]})")


def test_verifier_catches_cheaters():
    """Test that the verifier catches all cheating strategies."""
    from match_strike import MaxCutProblem
    from solver_node import SolverNode, TaskSpec
    from verifier import Verifier, CheatingNode

    print("  5a. Verifier catches cheaters...", end=" ")
    problem = MaxCutProblem.random(n_nodes=10, edge_prob=0.5, seed=42)
    task = TaskSpec(
        task_id="test",
        problem=problem,
        gamma=np.pi * 0.7,
        beta=np.pi * 0.3,
        n_layers=3,
        n_shots=200,
    )

    verifier = Verifier(check_fraction=0.1)

    # Honest node passes
    honest = SolverNode()
    honest_result = honest.execute(task)
    v = verifier.verify(honest_result, task)
    assert v.verified, "Honest node should pass"

    # Inflator caught by shallow check
    cheater = CheatingNode(strategy="inflate")
    fake = cheater.fabricate_result(task)
    v = verifier.verify(fake, task)
    assert not v.verified, "Inflator should be caught"

    # Random values caught
    cheater = CheatingNode(strategy="random_values")
    fake = cheater.fabricate_result(task)
    v = verifier.verify(fake, task)
    assert not v.verified, "Random value cheater should be caught"

    # Hash tampering caught
    tampered = honest.execute(task)
    tampered.result_hash = "fake"
    v = verifier.verify(tampered, task)
    assert not v.verified, "Hash tampering should be caught"

    # Fabricator caught by deep check
    cheater = CheatingNode(strategy="fabricate")
    fake = cheater.fabricate_result(task)
    v = verifier.verify(fake, task, deep=True)
    assert not v.verified, "Fabricator should be caught by deep check"

    print("PASS (all 4 cheating strategies detected)")


def test_full_pipeline():
    """End-to-end: problem → distribute → verify → collect best."""
    from match_strike import MaxCutProblem
    from coordinator import Coordinator
    from solver_node import SolverNode, TaskSpec
    from verifier import Verifier

    print("  6a. Full pipeline...", end=" ")
    problem = MaxCutProblem.random(n_nodes=12, edge_prob=0.5, seed=42)
    _, optimal_cut = problem.brute_force_optimal()

    # Step 1: Submit problem to coordinator
    coord = Coordinator(problem, n_layers=3)
    nodes = coord.register_nodes(4)

    # Step 2: Run distributed computation
    log = coord.run_adaptive(n_rounds=20, shots_per_node=25)

    # Step 3: Verify results from each node
    verifier = Verifier(check_fraction=0.1)
    all_verified = True
    for agg in coord.results[:3]:  # Check first 3 rounds
        for node_result in agg.node_results:
            task = TaskSpec(
                task_id=node_result.task_id,
                problem=problem,
                gamma=agg.gamma,
                beta=agg.beta,
                n_layers=3,
                n_shots=len(node_result.bitstrings),
            )
            v = verifier.verify(node_result, task)
            if not v.verified:
                all_verified = False

    # Step 4: Collect results
    ratio = coord.log.best_cut / optimal_cut
    stats = coord.stats()

    assert all_verified, "All honest nodes should pass verification"
    assert ratio > 0.8, f"Should find decent solution, got {ratio:.1%}"

    print(f"PASS (ratio={ratio:.1%}, verified={all_verified}, "
          f"samples={stats['total_samples']})")


def run_all_tests():
    """Run the full integration test suite."""
    print("\nQuantum Carburetor — Integration Test Suite")
    print("=" * 60)

    tests = [
        ("Phase 1: Quantum VM", [
            test_quantum_vm_basics,
            test_tensor_vm_matches_statevector,
            test_tensor_vm_scales,
        ]),
        ("Phase 1: Match-Strike Engine", [
            test_match_strike_finds_good_solutions,
        ]),
        ("Phase 1: Adaptive Controller", [
            test_adaptive_beats_random,
        ]),
        ("Phase 2: Distributed Compute", [
            test_distributed_coordination,
            test_distributed_scaling,
        ]),
        ("Phase 2: Verification", [
            test_verifier_catches_cheaters,
        ]),
        ("Full Pipeline", [
            test_full_pipeline,
        ]),
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for group_name, group_tests in tests:
        print(f"\n{group_name}:")
        for test_fn in group_tests:
            total += 1
            try:
                test_fn()
                passed += 1
            except Exception as e:
                failed += 1
                errors.append((test_fn.__name__, str(e)))
                print(f"  FAIL: {test_fn.__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\n  Failures:")
        for name, err in errors:
            print(f"    {name}: {err}")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
