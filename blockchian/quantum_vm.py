"""
Quantum VM — Exact state vector simulator for the Quantum Carburetor.

10-20 qubits, numpy only. No approximations.
Prepare → Superpose → Interfere → Measure → Reset.
"""

import numpy as np


class QuantumVM:
    """Exact state vector quantum simulator."""

    def __init__(self, n_qubits: int = 10):
        if n_qubits > 20:
            raise ValueError(f"n_qubits={n_qubits} exceeds safe limit of 20 (2^20 = 1M amplitudes)")
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.state = np.zeros(self.dim, dtype=np.complex128)
        self.state[0] = 1.0
        self._rng = np.random.default_rng()

    def reset(self):
        """Snap back to |000...0⟩."""
        self.state[:] = 0.0
        self.state[0] = 1.0

    def _apply_single(self, gate: np.ndarray, qubit: int):
        """Apply a 2x2 gate to a single qubit via tensor reshaping."""
        shape = [2] * self.n_qubits
        psi = self.state.reshape(shape)
        psi = np.moveaxis(psi, qubit, 0)
        flat = psi.reshape(2, -1)
        flat = gate @ flat
        psi = flat.reshape(psi.shape)
        psi = np.moveaxis(psi, 0, qubit)
        self.state = psi.reshape(self.dim)

    def h(self, qubit: int):
        """Hadamard."""
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        self._apply_single(H, qubit)

    def x(self, qubit: int):
        """Pauli-X."""
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self._apply_single(X, qubit)

    def z(self, qubit: int):
        """Pauli-Z."""
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        self._apply_single(Z, qubit)

    def rz(self, qubit: int, theta: float):
        """Rotation around Z."""
        gate = np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=np.complex128)
        self._apply_single(gate, qubit)

    def ry(self, qubit: int, theta: float):
        """Rotation around Y."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
        self._apply_single(gate, qubit)

    def phase(self, qubit: int, phi: float):
        """Phase gate."""
        gate = np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=np.complex128)
        self._apply_single(gate, qubit)

    def cx(self, control: int, target: int):
        """CNOT."""
        shape = [2] * self.n_qubits
        psi = self.state.reshape(shape)
        idx_c1 = [slice(None)] * self.n_qubits
        idx_c1[control] = 1
        sub = psi[tuple(idx_c1)].copy()
        t_idx = target if target < control else target - 1
        sub = np.moveaxis(sub, t_idx, 0)
        sub_flat = sub.reshape(2, -1)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sub_flat = X @ sub_flat
        sub = sub_flat.reshape(sub.shape)
        sub = np.moveaxis(sub, 0, t_idx)
        psi[tuple(idx_c1)] = sub
        self.state = psi.reshape(self.dim)

    def cz(self, qubit_a: int, qubit_b: int):
        """Controlled-Z."""
        shape = [2] * self.n_qubits
        psi = self.state.reshape(shape)
        idx = [slice(None)] * self.n_qubits
        idx[qubit_a] = 1
        idx[qubit_b] = 1
        psi[tuple(idx)] *= -1
        self.state = psi.reshape(self.dim)

    def apply_phase_oracle(self, costs: np.ndarray):
        """Apply diagonal phase rotation: |x⟩ → e^(i·γ·C(x))|x⟩."""
        self.state *= np.exp(1j * costs)

    def apply_mixer(self, beta: float):
        """Apply transverse field mixer: product of Ry(2β) on all qubits."""
        for q in range(self.n_qubits):
            self.ry(q, 2 * beta)

    def probabilities(self) -> np.ndarray:
        """Born rule: |ψ|²."""
        return np.abs(self.state) ** 2

    def measure(self) -> int:
        """Collapse. Returns a basis state index sampled from |ψ|²."""
        probs = self.probabilities()
        return int(self._rng.choice(self.dim, p=probs))

    def measure_shots(self, n_shots: int) -> np.ndarray:
        """Multiple measurements without intermediate reset."""
        probs = self.probabilities()
        return self._rng.choice(self.dim, size=n_shots, p=probs)

    def measure_bitstring(self) -> str:
        """Measure and return as a binary string."""
        idx = self.measure()
        return format(idx, f'0{self.n_qubits}b')

    def sigma(self) -> float:
        """Effective σ value: 0 = pure superposition, 1 = collapsed."""
        probs = self.probabilities()
        ipr = 1.0 / (np.sum(probs ** 2) * self.dim)
        sigma = 1.0 - (ipr * self.dim - 1) / (self.dim - 1)
        return float(np.clip(sigma, 0.0, 1.0))

    def entropy(self) -> float:
        """Shannon entropy of the probability distribution (bits)."""
        probs = self.probabilities()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def superpose_all(self):
        """Put all qubits into equal superposition."""
        for q in range(self.n_qubits):
            self.h(q)

    def __repr__(self):
        return (f"QuantumVM(n_qubits={self.n_qubits}, "
                f"σ={self.sigma():.3f}, "
                f"entropy={self.entropy():.2f} bits)")
