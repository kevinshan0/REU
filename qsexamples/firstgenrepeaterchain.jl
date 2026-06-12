using QuantumSavory
using Graphs
using ConcurrentSim
using ResumableFunctions
using GLMakie

GLMakie.activate!(inline=false)

# parameters
SLOTS_PER_REG = [2, 3, 4, 4, 5, 3, 2]   # slots per node with alice and bob at each end
T2 = 100.0                      # T2 dephasing time for each node
F = 0.97                        # Fidelity of the raw Bell pairs
ENTANGLER_WAIT_TIME = 0.1       # How long to wait if all qubits are busy before retrying entangling
ENTANGLER_BUSY_TIME = 1.0       # How long it takes to establish a newly entangled pair
SWAPPER_WAIT_TIME = 0.1         # How long to wait if all qubits are unavailable for swapping
SWAPPER_BUSY_TIME = 0.15        # How long it takes to swap two qubits
PURIFIER_WAIT_TIME = 0.15       # How long to wait if there are no pairs to be purified
PURIFIER_BUSY_TIME = 0.2        # How long the purification circuit takes to execute

# sets up the simulation and the network
function setup()
    registers = Register[]

    # define each register
    for slots in SLOTS_PER_REG
        qubit = [Qubit() for _ in 1:slots]
        represenation = [QuantumOpticsRepr() for _ in 1:slots]
        background = [T2Dephasing(T2) for _ in 1:slots]
        register = Register(qubit, represenation, background)
        push!(registers, register)
    end

    # define network shape
    graph = path_graph(length(SLOTS_PER_REG))
    network = RegisterNet(graph, registers)

    # define simulation
    simulation = Simulation()

    # create the datastructure for entanglement tracker
    for v in vertices(network)
        # Create an array specifying whether a qubit is entangled with another qubit
        network[v, :enttrackers] = Any[nothing for i in 1:SLOTS_PER_REG[v]]
    end

    return simulation, network
end

simulation, network = setup()

# entanglement protocol
### register/nodes are interchangable
@resumable function entanglement_protocol(
    simulation,
    register_a_index,
    register_b_index,
)
    println("Entangler started for $register_a_index and $register_b_index")
    while true
        # checks for free slots. If none, waits then rechecks
        slot_a_index = find_free_slots(register_a_index)
        slot_b_index = find_free_slots(register_b_index)
        println("DEBUG: Entangler $register_a_index -> $register_b_index found slots $slot_a_index, $slot_b_index")
        if (isnothing(slot_a_index) || isnothing(slot_b_index))
            @simlog simulation "slots not available, waiting for: $(ENTANGLER_WAIT_TIME)"
            @yield timeout(simulation, ENTANGLER_WAIT_TIME)
            continue
        end

        # if free slots are found, requests the slots as resources
        slot_a = network[register_a_index, slot_a_index]
        slot_b = network[register_b_index, slot_b_index]
        println("DEBUG: Entangler $register_a_index -> $register_b_index acquired slot references")
        @yield request(slot_a) & request(slot_b) # ConcurrentSim uses & to wait for both resource to be locked before continuing
        println("DEBUG: Entangler $register_a_index -> $register_b_index locked slots")
        @simlog simulation "locked slots"

        # once slots are locked by this process, run the simulation forward, initialize the states of the slots, and update the entaglement tracker
        @simlog simulation "entangling slots, busy for: $(ENTANGLER_BUSY_TIME)"
        @yield timeout(simulation, ENTANGLER_BUSY_TIME)
        initialize!((slot_a, slot_b), create_noisy_bell_pair(F); time=now(simulation))
        network[register_a_index, :enttrackers][slot_a_index] = (node=register_b_index, slot=slot_b_index)
        network[register_b_index, :enttrackers][slot_b_index] = (node=register_a_index, slot=slot_a_index)
        @simlog simulation "entangled nodes slots: $(slot_a) in register $(register_a_index), $(slot_b) in register $(register_b_index)"
        unlock(slot_a)
        unlock(slot_b)
        @simlog simulation "unlocked slots"
    end
end

# helper to find free slots
function find_free_slots(register_index)
    register = network[register_index]
    register_size = nsubsystems(register)
    findfirst(slot_index -> !isassigned(register, slot_index) && !islocked(register[slot_index]), 1:register_size)
end

# helper function that creates bell pairs of a given fidelity
function create_noisy_bell_pair(F)
    phi_plus = (Z1 ⊗ Z1 + Z2 ⊗ Z2) / sqrt(2)
    phi_plus_density_matrix = SProjector(phi_plus)
    mixed_state_density_matrix = MixedState(phi_plus_density_matrix)
    return F * phi_plus_density_matrix + (1 - F) * mixed_state_density_matrix
end

# swapping protocol
@resumable function swapping_protocol(
    simulation,
    register_index
)
    while true
        # similar to entangler, checks for slots that are eligible for swapping
        slot_pair = find_swappable_slots(register_index)
            if isnothing(slot_pair)
                @simlog simulation "no available qubits to swap, waiting for: $(SWAPPER_WAIT_TIME)"
                @yield timeout(simulation, SWAPPER_WAIT_TIME)
                continue
            end

            # if swappable slots are found, requests the slots as resources
            slot_a_index, slot_b_index = slot_pair
            local_register = network[register_index]
            slot_a = local_register[slot_a_index]
            slot_b = local_register[slot_b_index]
            @yield request(slot_a) & request(slot_b)
            @simlog simulation "locked slots"

            @simlog simulation "swapping slots, busy for: $(SWAPPER_BUSY_TIME)"
            @yield timeout(simulation, SWAPPER_BUSY_TIME)
            entanglement_info_a = network[register_index, :enttrackers][slot_a_index]
            entanglement_info_b = network[register_index, :enttrackers][slot_b_index]
            remote_register_a = network[entanglement_info_a.node]
            remote_register_b = network[entanglement_info_b.node]

            # uptotime! updates the states to account for decoherence during the time passed in the simulation
            # QS uses lazy decoherence. use this after states are initialized and time has passed
            @simlog simulation "updating state to account for decoherence"
            uptotime!(
                (
                    slot_a,
                    slot_b,
                    remote_register_a[entanglement_info_a.slot],
                    remote_register_b[entanglement_info_b.slot]
                ),
                now(simulation)
            )

            # implement swap circuit using apply and measure by hand
            # can use swapcircuit exported by CircuitZoo instead
            @simlog simulation "perform swap circuit"
            apply!((slot_a, slot_b), CNOT)
            apply!(slot_a, H)
            measurement_a = project_traceout!(slot_a, Z)
            measurement_b = project_traceout!(slot_b, Z)

            # measurement table to update remote slots via a gate
            # measurement_a,    measurement_b,  outcome
            # Z1 (|0⟩)          Z1 (|0⟩)        nothing
            # Z2 (|1⟩)          Z1 (|0⟩)        Z gate on remote_b
            # Z1 (|0⟩)          Z2 (|1⟩)        X gate on remote_a
            # Z2 (|1⟩)          Z2 (|1⟩)        XZ gates
            if measurement_a == 2
                apply!(remote_register_a[entanglement_info_a.slot], Z)
            end
            if measurement_b == 2
                apply!(remote_register_b[entanglement_info_b.slot], X)
            end

            # update entanglement trackers
            network[entanglement_info_a.node, :enttrackers][entanglement_info_a.slot] = entanglement_info_b
            network[entanglement_info_b.node, :enttrackers][entanglement_info_b.slot] = entanglement_info_a
            network[register_index, :enttrackers][slot_a_index] = nothing
            network[register_index, :enttrackers][slot_b_index] = nothing
            @simlog simulation "swapped local slots $(slot_a_index) and $(slot_b_index) on register $(register_index)"
            @simlog simulation "now entangled: slot $(entanglement_info_a.slot) on register $(entanglement_info_a.node) with slot $(entanglement_info_b.slot) on register $(entanglement_info_b.node)"
            unlock(slot_a)
            unlock(slot_b)
    end
end

# helper function to find slots within register to swap
# this helper finds the slots that maximize the distance of the entangled pair after the swap
function find_swappable_slots(local_register_index)
    entanglement_trackers = network[local_register_index, :enttrackers]
    left_registers_info = []
    right_registers_info = []
    for (slot_index, entanglement_info) in enumerate(entanglement_trackers)
        # check if this slot contains an entangled qubit and it is it not locked by another process
        if !isnothing(entanglement_info) && !islocked(network[local_register_index, slot_index])
            # if the remote register has index less than the current register, push to left_register_indexes
            if entanglement_info.node < local_register_index
                push!(left_registers_info,
                    (
                        local_slot_index=slot_index,
                        remote_register_index=entanglement_info.node,
                        remote_slot_index=entanglement_info.slot
                    )
                )
            end

            # if the remote register has index greater than the current register, push right_register_indexes
            if entanglement_info.node > local_register_index
                push!(right_registers_info,
                    (
                        local_slot_index=slot_index,
                        remote_register_index=entanglement_info.node,
                        remote_slot_index=entanglement_info.slot
                    )
                )
            end
        end
    end

    # return nothing if no entanglement exists to the left or right of current register
    isempty(left_registers_info) && return nothing
    isempty(right_registers_info) && return nothing

    # find the index within left/right_registers_info that refers to the register that is furthest left and right of the current register
    _, farthest_left = findmin(info -> info.remote_register_index, left_registers_info)
    _, farthest_right = findmax(info -> info.remote_register_index, right_registers_info)

    return left_registers_info[farthest_left].local_slot_index, right_registers_info[farthest_right].local_slot_index
end

# purify protocol
@resumable function purify_protocol(
    simulation,
    register_a_index,
    register_b_index
)
    round = 0
    while true
        slot_pairs = find_purifiable_qubits(register_a_index, register_b_index)
            if isnothing(slot_pairs)
                @simlog simulation "no purifiable qubit pairs, waiting for: $(PURIFIER_WAIT_TIME)"
                @yield timeout(simulation, PURIFIER_WAIT_TIME)
                continue
            end

            pair_1_slot_a_index, pair_1_slot_b_index, pair_2_slot_a_index, pair_2_slot_b_index = slot_pairs
            pair_1_slot_a = network[register_a_index, pair_1_slot_a_index]
            pair_1_slot_b = network[register_b_index, pair_1_slot_b_index]
            pair_2_slot_a = network[register_a_index, pair_2_slot_a_index]
            pair_2_slot_b = network[register_b_index, pair_2_slot_b_index]
            @yield request(pair_1_slot_a) &
                   request(pair_1_slot_b) &
                   request(pair_2_slot_a) &
                   request(pair_2_slot_b)
            @simlog simulation "slots locked"

            # implement BBPSSW directly
            @simlog simulation "performing purification circuit"
            @yield timeout(simulation, PURIFIER_BUSY_TIME)

            if round % 2 == 0
                # purify X errors
                apply!((pair_2_slot_a, pair_1_slot_a), CNOT)
                apply!((pair_2_slot_b, pair_1_slot_b), CNOT)
                measurement_a = project_traceout!(pair_2_slot_a, X)
                measurement_b = project_traceout!(pair_2_slot_b, X)
            else
                # purify Z errors
                apply!((pair_2_slot_a, pair_1_slot_a), XCZ)
                apply!((pair_2_slot_b, pair_1_slot_b), XCZ)
                measurement_a = project_traceout!(pair_2_slot_a, Z)
                measurement_b = project_traceout!(pair_2_slot_b, Z)
            end

            success = (measurement_a == measurement_b)
            if success
                round += 1
                @simlog simulation "purification success at $(register_a_index):$(pair_1_slot_a_index) $(register_b_index):$(pair_1_slot_b_index) by sacrifice of $(register_a_index):$(pair_2_slot_a_index) $(register_b_index):$(pair_2_slot_b_index)"
            else
                @simlog simulation "purification failed at $(register_a_index):$(pair_1_slot_a_index)&$(pair_2_slot_a_index) and $(register_b_index):$(pair_1_slot_b_index)&$(pair_2_slot_b_index)"
                network[register_a_index, :enttrackers][pair_1_slot_a_index] = nothing
                network[register_b_index, :enttrackers][pair_1_slot_b_index] = nothing
                traceout!(pair_1_slot_a)
                traceout!(pair_1_slot_b)
            end

            # clear the sacrificed pair from entanglement trackers
            network[register_a_index, :enttrackers][pair_2_slot_a_index] = nothing
            network[register_b_index, :enttrackers][pair_2_slot_b_index] = nothing
            unlock(pair_1_slot_a)
            unlock(pair_1_slot_b)
            unlock(pair_2_slot_a)
            unlock(pair_2_slot_b)
    end
end

# helper function to find pairs of qubits
function find_purifiable_qubits(
    register_index_a,
    register_index_b
)
    register_a = network[register_index_a]
    register_b = network[register_index_b]
    entanglement_trackers = network[register_index_a, :enttrackers]

    entanglement_info = [(local_slot_index=i, remote_slot_index=n.slot) for (i, n) in enumerate(entanglement_trackers)
                         if !isnothing(n) && n.node == register_index_b && !islocked(register_a[i]) && !islocked(register_b[n.slot])
    ]

    if length(entanglement_info) < 2
        return nothing
    end

    return entanglement_info[1].local_slot_index,
    entanglement_info[1].remote_slot_index,
    entanglement_info[2].local_slot_index,
    entanglement_info[2].remote_slot_index
end


# begin processes
println("Number of edges: ", length(collect(edges(network))))
for (; src, dst) in edges(network)
    println("Spawning entangler for $src -> $dst")
    @process entanglement_protocol(simulation, src, dst)
end
for node in vertices(network)
    @process swapping_protocol(simulation, node)
end
for node_a in vertices(network)
    for node_b in vertices(network)
        if node_a < node_b
            @process purify_protocol(simulation, node_a, node_b)
        end
    end
end

# visualization
fig = Figure(size=(800, 400))
_, ax, _, obs = registernetplot_axis(fig[1, 1], network)

ts = Observable(Float64[0])
fidXX = Observable(Float64[0])
fidZZ = Observable(Float64[0])
ax_fid = Axis(fig[1, 2][1, 1], xlabel="time", ylabel="Entanglement Stabilizer\nExpectation")
lXX = stairs!(ax_fid, ts, fidXX, label="XX")
lZZ = stairs!(ax_fid, ts, fidZZ, label="ZZ")
xlims!(0, nothing)
ylims!(-0.05, 1.05)
Legend(fig[1, 2][2, 1], [lXX, lZZ], ["XX", "ZZ"],
    orientation=:horizontal, tellwidth=false, tellheight=true)

display(fig)

registers = [network[node] for node in vertices(network)]
last = length(registers)

step_ts = range(0, 100, step=0.1)
record(fig, "firstgenrepeater-07.observable.mp4", step_ts; framerate=10, visible=true) do t
    run(simulation, t)

    fXX = real(observable(registers[[1, last]], [2, 2], X ⊗ X; something=0.0, time=t))
    fZZ = real(observable(registers[[1, last]], [2, 2], Z ⊗ Z; something=0.0, time=t))
    push!(fidXX[], fXX)
    push!(fidZZ[], fZZ)
    push!(ts[], t)

    ax.title = "t=$(t)"
    notify(obs)
    notify(ts)
    xlims!(ax_fid, 0, t + 0.5)
end