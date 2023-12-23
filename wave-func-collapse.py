import numpy as np

def normalize(state):
    """ Normalize the quantum state vector. """
    return state / np.linalg.norm(state)

def matrix_exponential(matrix, t):
    """ Compute the matrix exponential using diagonalization, valid for normal matrices. """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvectors @ np.diag(np.exp(-1j * eigenvalues * t)) @ np.linalg.inv(eigenvectors)

def evolve(state, hamiltonian, time):
    """ Evolve the quantum state using the time-evolution operator. """
    U = matrix_exponential(hamiltonian, time)
    return normalize(np.dot(U, state))

# Example: 2 particles, each with 2 states (like spin up/down)
states = np.array([[1, 0], [0, 1]], dtype=complex)  # States for each particle
system_state = np.kron(states[0], states[1])  # Tensor product for system state
system_state = normalize(system_state)

# Observable Operators (e.g., spin in the Z direction)
spin_z = np.array([[1, 0], [0, -1]], dtype=complex)
observable = np.kron(spin_z, np.eye(2))  # Observable for the first particle

# Hamiltonian (4x4 Identity Matrix for simplicity)
hamiltonian = np.eye(4, dtype=complex)  # Example Hamiltonian

# Time Evolution (simplified)
system_state = evolve(system_state, hamiltonian, time=0.1)

# Measurement
probabilities = np.abs(np.dot(system_state.conj().T, observable))**2
outcome = np.random.choice(len(system_state), p=probabilities)

# Collapse the wave function
collapsed_state = np.zeros_like(system_state)
collapsed_state[outcome] = 1
collapsed_state = normalize(collapsed_state)

print("Collapsed state:", collapsed_state)
