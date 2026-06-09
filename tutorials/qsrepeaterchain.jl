using QuantumSavory
using Graphs

# network parameters
NODES = 5 # number of nodes including alice and bob
MEMORY_DEPHASE_TIME = 1 #

# alice's register description
alice_qubit = Qubit()
alice_representation = CliffordRepr()
alice_background = T2Dephasing(T2)
alice_register = Register(alice_qubits, alice_representation, alice_background)

# bob's register description
bob_qubit = Qubit()
bob_representation = CliffordRepr()
bob_background = T2Dephasing(T2)
bob_register = Register(bob_qubit, bob_representation, bob_background)

# repeater nodes description
repeater_registers = Register[]

for _ in NODES - 2
    repeater_qubit_l = Qubit()
    repeater_qubit_r = Qubit()
    repeater_representation_l = CliffordRepr()
    repeater_representation_r = CliffordRepr()
    repeater_background_l = T2Dephasing(T2)
    repeater_background_r = T2Dephasing(T2)
    repeater_register = Register(
        [repeater_qubit_l, repeater_qubit_r],
        [repeater_representation_l, repeater_representation_r],
        [repeater_background_l, repeater_background_r]
    )
    push![repeater_registers, repeater_register]
end

# add nodes to register net
registers = Register[]
push![registers, alice_register]
push![registers, bob_register]

for repeater_register in repeater_registers
    push![registers, repeater_register]
end

linear_graph = path_graph(NODES)
register_network = RegisterNet(linear_graph, registers)
