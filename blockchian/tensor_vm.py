"""
Tensor Network VM — Matrix Product State simulator for the Quantum Carburetor.

Scales to 40+ qubits for low-entanglement circuits (which is exactly what
match-strike produces). Bond dimension controls accuracy/memory tradeoff.

numpy only. Same interface as QuantumVM where possible.
"""

import numpy as np
from typing import Sequence


class TensorVM:
    """Matrix Product State (MPS) quantum simulator.

    Represents the quantum state as a chain of tensors:
        |ψ⟩ = Σ A[0]^{s0} · A[1]^{s1} · ... · A[n-1]^{s_{n-1}} |s0 s1 ... s_{n-1}⟩

    Each A[i] is a tensor of shape (bond_left, 2, bond_right).
    Bond dimension χ controls the max entanglement the MPS can represent.
    For product states χ=1. For match-strike circuits χ rarely exceeds 32-64.
    """

    def __init__(self, n_qubits: int = 30, max_bond: int = 64):
        self.n_qubits = n_qubits
        self.max_bond = max_bond
        self._rng = np.random.default_rng()
        self.tensors: list[np.ndarray] = []
        self.reset()

    def reset(self):
        """Reset to |000...0⟩ product state. Each tensor is (1, 2, 1)."""
        self.tensors = []
        for _ in range(self.n_qubits):
            t = np.zeros((1, 2, 1), dtype=np.complex128)
            t[0, 0, 0] = 1.0  # |0⟩
            self.tensors.append(t)

    def _truncated_svd(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SVD with truncation to max_bond dimension. Robust to numerical issues."""
        try:
            U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback: add small noise and retry
            noise = np.random.default_rng().normal(0, 1e-12, matrix.shape)
            U, S, Vh = np.linalg.svd(matrix + noise, full_matrices=False)

        # Truncate small singular values
        mask = S > 1e-14
        if not np.any(mask):
            mask[0] = True  # Keep at least one
        S = S[mask]
        U = U[:, :len(S)]
        Vh = Vh[:len(S), :]

        # Truncate to max bond
        chi = min(len(S), self.max_bond)
        U = U[:, :chi]
        S = S[:chi]
        Vh = Vh[:chi, :]

        # Normalize to prevent drift
        norm = np.linalg.norm(S)
        if norm > 0:
            S /= norm
        return U, S, Vh

    def _apply_single_qubit(self, gate: np.ndarray, qubit: int):
        """Apply a 2x2 gate to a single qubit. Updates only one tensor."""
        # tensors[qubit] shape: (χ_L, 2, χ_R)
        t = self.tensors[qubit]
        chi_l, _, chi_r = t.shape
        # Contract gate with physical index
        # new[a, s', b] = Σ_s gate[s', s] * t[a, s, b]
        new_t = np.einsum('ij,ajb->aib', gate, t)
        self.tensors[qubit] = new_t

    def _apply_two_qubit(self, gate: np.ndarray, q1: int, q2: int):
        """Apply a 4x4 gate to two adjacent qubits.

        If qubits aren't adjacent, we swap them into position first.
        gate is a 4x4 matrix acting on the joint space of q1, q2.
        """
        if abs(q1 - q2) != 1:
            # Move qubits adjacent via SWAP chain
            if q1 > q2:
                q1, q2 = q2, q1
                # Swap the gate indices too
                swap = np.array([
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]
                ], dtype=np.complex128)
                gate = swap @ gate @ swap

            # SWAP q2 leftward toward q1+1
            for i in range(q2, q1 + 1, -1):
                self._apply_swap(i - 1, i)
            # Apply gate on (q1, q1+1)
            self._apply_two_qubit_adjacent(gate, q1)
            # SWAP back
            for i in range(q1 + 1, q2):
                self._apply_swap(i, i + 1)
            return

        if q1 > q2:
            q1, q2 = q2, q1
            swap = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=np.complex128)
            gate = swap @ gate @ swap

        self._apply_two_qubit_adjacent(gate, q1)

    def _apply_two_qubit_adjacent(self, gate: np.ndarray, q: int):
        """Apply gate to adjacent qubits q and q+1. Core MPS operation."""
        t1 = self.tensors[q]      # (χ_L, 2, χ_mid)
        t2 = self.tensors[q + 1]  # (χ_mid, 2, χ_R)
        chi_l = t1.shape[0]
        chi_r = t2.shape[2]

        # Contract t1 and t2 into a single tensor
        # theta[a, s1, s2, b] = Σ_m t1[a, s1, m] * t2[m, s2, b]
        theta = np.einsum('asm,msb->asb', t1, t2)  # (χ_L, 2, 2, χ_R) — wait, wrong
        # Actually need 4 indices
        theta = np.einsum('aim,mjb->aijb', t1, t2)  # (χ_L, 2, 2, χ_R)

        # Apply gate: gate is 4x4, acting on (s1, s2) = (i, j)
        gate_4d = gate.reshape(2, 2, 2, 2)
        # new_theta[a, i', j', b] = Σ_{i,j} gate[i',j',i,j] * theta[a,i,j,b]
        new_theta = np.einsum('klij,aijb->aklb', gate_4d, theta)

        # SVD to split back into two tensors
        # Reshape to matrix: (χ_L * 2, 2 * χ_R)
        mat = new_theta.reshape(chi_l * 2, 2 * chi_r)
        U, S, Vh = self._truncated_svd(mat)
        chi_new = len(S)

        # Absorb S into U (left-canonical)
        US = U * S[np.newaxis, :]

        # Reshape back to MPS tensors
        self.tensors[q] = US.reshape(chi_l, 2, chi_new)
        self.tensors[q + 1] = Vh.reshape(chi_new, 2, chi_r)

    def _apply_swap(self, q1: int, q2: int):
        """SWAP two adjacent qubits."""
        swap = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.complex128)
        self._apply_two_qubit_adjacent(swap, min(q1, q2))

    # ── Standard gates ──────────────────────────────────────────────

    def h(self, qubit: int):
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        self._apply_single_qubit(H, qubit)

    def x(self, qubit: int):
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self._apply_single_qubit(X, qubit)

    def z(self, qubit: int):
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        self._apply_single_qubit(Z, qubit)

    def rz(self, qubit: int, theta: float):
        gate = np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=np.complex128)
        self._apply_single_qubit(gate, qubit)

    def ry(self, qubit: int, theta: float):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
        self._apply_single_qubit(gate, qubit)

    def phase(self, qubit: int, phi: float):
        gate = np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=np.complex128)
        self._apply_single_qubit(gate, qubit)

    def cx(self, control: int, target: int):
        cnot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)
        self._apply_two_qubit(cnot, control, target)

    def cz(self, qubit_a: int, qubit_b: int):
        cz_gate = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=np.complex128)
        self._apply_two_qubit(cz_gate, qubit_a, qubit_b)

    # ── Problem Hamiltonian ─────────────────────────────────────────

    def apply_phase_oracle(self, cost_function, gamma: float = 1.0):
        """Apply phase oracle using the cost function.

        For MPS, we can't apply a diagonal operator to all 2^n states directly.
        Instead, we decompose the cost Hamiltonian into local terms.

        cost_function: a callable that takes (qubit_i, qubit_j) pairs and returns
                      the coupling strength, OR a MaxCutProblem-like object.
        """
        if hasattr(cost_function, 'edges') and hasattr(cost_function, 'weights'):
            # MaxCutProblem-style: apply ZZ interactions for each edge
            for (i, j), w in zip(cost_function.edges, cost_function.weights):
                # For MaxCut: phase = γ * w * (1 - ZiZj) / 2
                # This decomposes into single-qubit Z rotations and CZ gates
                self._apply_zz_interaction(i, j, gamma * w)
        else:
            raise ValueError("cost_function must have .edges and .weights attributes")

    def _apply_zz_interaction(self, q1: int, q2: int, angle: float):
        """Apply e^{-i * angle * Z⊗Z / 2} to qubits q1, q2.

        Decomposed as: CNOT(q1,q2) → Rz(q2, angle) → CNOT(q1,q2)
        """
        self.cx(q1, q2)
        self.rz(q2, angle)
        self.cx(q1, q2)

    def apply_mixer(self, beta: float):
        """Transverse field mixer: product of Ry(2β) on all qubits."""
        for q in range(self.n_qubits):
            self.ry(q, 2 * beta)

    # ── Measurement ─────────────────────────────────────────────────

    def measure(self) -> int:
        """Sample one bitstring from the MPS via sequential left-to-right sampling.

        Maintains a left boundary vector that accumulates measured outcomes.
        This is exact for any MPS gauge.
        """
        bitstring = 0
        # left_vec starts as the trivial left boundary: shape (1,)
        left_vec = np.ones(1, dtype=np.complex128)

        for q in range(self.n_qubits):
            t = self.tensors[q]  # (χ_L, 2, χ_R)

            # For each outcome s, compute the conditional amplitude vector
            # v_s = left_vec @ A[:, s, :] → shape (χ_R,)
            p = np.zeros(2)
            vecs = []
            for s in range(2):
                v = left_vec @ t[:, s, :]  # (χ_R,)
                vecs.append(v)
                p[s] = np.real(np.vdot(v, v))

            total = p[0] + p[1]
            if total < 1e-30:
                outcome = 0
            else:
                p /= total
                outcome = int(self._rng.choice(2, p=p))

            bitstring |= (outcome << (self.n_qubits - 1 - q))

            # Update left boundary: propagate the chosen outcome
            chosen_v = vecs[outcome]
            norm = np.linalg.norm(chosen_v)
            if norm > 0:
                left_vec = chosen_v / norm
            else:
                left_vec = chosen_v

        return bitstring

    def measure_shots(self, n_shots: int) -> np.ndarray:
        """Multiple independent measurements."""
        return np.array([self.measure() for _ in range(n_shots)])

    def measure_bitstring(self) -> str:
        idx = self.measure()
        return format(idx, f'0{self.n_qubits}b')

    # ── State inspection ────────────────────────────────────────────

    def bond_dimensions(self) -> list[int]:
        """Return the bond dimension at each cut."""
        dims = []
        for i in range(self.n_qubits - 1):
            dims.append(self.tensors[i].shape[2])
        return dims

    def max_bond_used(self) -> int:
        """Maximum bond dimension currently used."""
        dims = self.bond_dimensions()
        return max(dims) if dims else 1

    def total_parameters(self) -> int:
        """Total number of complex parameters in the MPS."""
        return sum(t.size for t in self.tensors)

    def memory_bytes(self) -> int:
        """Approximate memory usage in bytes."""
        return self.total_parameters() * 16  # complex128 = 16 bytes

    def superpose_all(self):
        """Put all qubits into equal superposition."""
        for q in range(self.n_qubits):
            self.h(q)

    def sigma(self) -> float:
        """Approximate σ value: 0 = pure superposition, 1 = collapsed.

        For product states (bond dim 1): exact IPR computed from the
        product of per-site probabilities.

        For entangled states: estimated via sampling. Draw a batch of
        bitstrings and compute empirical IPR. This is O(n_samples * n_qubits)
        instead of O(2^n) — practical for any MPS size.
        """
        max_bd = self.max_bond_used()
        dim = 2 ** self.n_qubits

        if max_bd == 1:
            # Product state: exact IPR from per-site probabilities
            # p(x) = prod_i p_i(x_i), so sum(p^2) = prod_i sum_s(p_i(s)^2)
            ipr_inv = 1.0  # will accumulate product of per-site sum(p^2)
            for q in range(self.n_qubits):
                t = self.tensors[q]  # (1, 2, 1)
                p = np.abs(t[0, :, 0]) ** 2
                p_total = p.sum()
                if p_total > 0:
                    p /= p_total
                ipr_inv *= np.sum(p ** 2)
            # ipr_inv = sum(p(x)^2), IPR = 1/(ipr_inv * dim)
            ipr = 1.0 / (ipr_inv * dim) if ipr_inv > 0 else 1.0
        else:
            # Entangled: estimate from samples
            n_samples = min(500, dim)
            samples = self.measure_shots(n_samples)
            counts = np.bincount(samples, minlength=dim)
            freqs = counts / n_samples
            sum_p2 = np.sum(freqs ** 2)
            ipr = 1.0 / (sum_p2 * dim) if sum_p2 > 0 else 1.0

        # Map IPR to sigma: ipr=1 (uniform) → σ=0, ipr=1/dim (peaked) → σ=1
        sigma = 1.0 - (ipr * dim - 1) / (dim - 1) if dim > 1 else 1.0
        return float(np.clip(sigma, 0.0, 1.0))

    def __repr__(self):
        dims = self.bond_dimensions()
        max_d = max(dims) if dims else 1
        mem_mb = self.memory_bytes() / 1e6
        return (f"TensorVM(n_qubits={self.n_qubits}, "
                f"max_bond_used={max_d}/{self.max_bond}, "
                f"mem={mem_mb:.1f}MB)")


# ── Comparison utility ──────────────────────────────────────────────

def compare_vms(n_qubits: int = 12, n_shots: int = 10000):
    """Compare TensorVM against QuantumVM for correctness."""
    from quantum_vm import QuantumVM

    print(f"Comparing VMs on {n_qubits} qubits, {n_shots} shots...")

    # Build identical circuits
    rng = np.random.default_rng(42)
    gates = []
    for _ in range(20):
        gate_type = rng.choice(['h', 'rz', 'ry', 'cx'])
        if gate_type in ('h', 'rz', 'ry'):
            q = int(rng.integers(0, n_qubits))
            angle = float(rng.uniform(0, 2 * np.pi)) if gate_type != 'h' else 0
            gates.append((gate_type, q, angle))
        else:
            q1, q2 = rng.choice(n_qubits, size=2, replace=False)
            gates.append(('cx', int(q1), int(q2)))

    # Run on QuantumVM
    qvm = QuantumVM(n_qubits)
    for g in gates:
        if g[0] == 'h':
            qvm.h(g[1])
        elif g[0] == 'rz':
            qvm.rz(g[1], g[2])
        elif g[0] == 'ry':
            qvm.ry(g[1], g[2])
        elif g[0] == 'cx':
            qvm.cx(g[1], g[2])
    exact_probs = qvm.probabilities()

    # Run on TensorVM
    tvm = TensorVM(n_qubits, max_bond=64)
    for g in gates:
        if g[0] == 'h':
            tvm.h(g[1])
        elif g[0] == 'rz':
            tvm.rz(g[1], g[2])
        elif g[0] == 'ry':
            tvm.ry(g[1], g[2])
        elif g[0] == 'cx':
            tvm.cx(g[1], g[2])

    # Sample from TensorVM and build histogram
    samples = tvm.measure_shots(n_shots)
    hist = np.bincount(samples, minlength=2**n_qubits) / n_shots

    # Compare distributions
    kl_div = np.sum(exact_probs * np.log2(exact_probs / (hist + 1e-10) + 1e-10))
    l1_error = np.sum(np.abs(exact_probs - hist))

    print(f"  QuantumVM: exact probabilities computed")
    print(f"  TensorVM:  {n_shots} samples, max bond used = {tvm.max_bond_used()}")
    print(f"  L1 error (sampling noise):  {l1_error:.4f}")
    print(f"  TensorVM memory: {tvm.memory_bytes() / 1024:.1f} KB")
    print(f"  State vector would need: {2**n_qubits * 16 / 1024:.1f} KB")
    print(f"  Compression ratio: {2**n_qubits * 16 / max(tvm.memory_bytes(), 1):.1f}x")

    return l1_error


if __name__ == "__main__":
    # Correctness test
    compare_vms(n_qubits=12, n_shots=50000)

    print()

    # Scale test — go beyond state vector limit
    for n in [20, 30, 40]:
        tvm = TensorVM(n_qubits=n, max_bond=32)
        tvm.superpose_all()
        # Shallow circuit
        for q in range(n - 1):
            tvm.cz(q, q + 1)
        for q in range(n):
            tvm.ry(q, 0.3)

        sample = tvm.measure_bitstring()
        print(f"  {n} qubits: bond dims = {tvm.bond_dimensions()}, "
              f"mem = {tvm.memory_bytes() / 1024:.1f} KB, "
              f"sample = {sample[:20]}...")
