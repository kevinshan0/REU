state based cutoff iis the method employed in https://arxiv.org/pdf/2207.06533

The idea is you precompute a $t_{cut}$ that ensures the sequence of
events that leads to the lowest end-to-end fidelity still results
in end-to-end fidelity above some threshold $F_{min}$. In addition,
the paper accounts for the existing parameters. These include: $\tau$,
which characterizes the exponential decay in fidelity of entangled
qubits, and $F_{new}$ which describes the fidelity of linked generated
by entanglement generation between adjacent nodes.

once $t_{cut}$ is computed, the paper discards the other parameters
with regards to the simulation (only $t_{cut}$ is considered as a
parameter for the MDP).

The key feature of this implementation is that it leans heavily into
the assumption of global knowledge. In a quantum repeater network,
the classical communication between links for entanglement swapping
introduces uncertainties about the age of links. A more realistic
approach is qubit retention-time based cutoffs. This cutoff implementation
simply looks at the age of a qubits, not the links, discards qubits once
they age past a certain threshold.

key assumptions for this cutoff implementation:
1. swapped links assume the age of the oldest link used to generate the
new link. It is possible to recompute the age of swapped links based on
post-swap fidelity.
>we assume that an entangled link generated as a result of
>entanglement swapping assumes the age of the oldest link
>that was involved in the swapping operation
2. global knowledge. This ensures instantaneous swaps with preserves the
time-slot and age calculation formulation used in the MDP