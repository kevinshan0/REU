using QuantumSavory
using ResumableFunctions
using ConcurrentSim
using Graphs
using GLMakie
using QuantumSavory.ProtocolZoo
using QuantumSavory.StatesZoo

GLMakie.activate!()

struct Configuration
    sizes::Vector{Int}
    T1::Float64
    F::Float64
    cutoff_time::Float64
end

function setup(c::Configuration)
    registers = Register[]
    for s in c.sizes
        qubits = [Qubit() for _ in 1:s]
        representation = [QuantumOpticsRepr() for _ in 1:s]
        background = [T1Decay(c.T1) for _ in 1:s]
        push!(registers, Register(qubits, representation, background))
    end

    graph = path_graph(length(c.sizes))
    network = RegisterNet(graph, registers)
    simulation = get_time_tracker(network)

    # run protocols
    consumer = EntanglementConsumer(simulation, network, 1, length(c.sizes))
    @process consumer()

    for (;src, dst) in edges(network)
        entangler = EntanglerProt(simulation, network, src, dst; pairstate=DepolarizedBellPair(c.F))
        @process entangler()
    end

    for node in vertices(network)
        cutoff = CutoffProt(simulation, network, node; retention_time=c.cutoff_time)
        @process cutoff()
        swapper = SwapperProt(simulation, network, node; nodeL = <(node), nodeH = >(node), chooseL = argmin, chooseH = argmax)
        @process swapper()
        tracker = EntanglementTracker(simulation, network, node)
        @process tracker()
    end

    for nodea in vertices(network)
        for nodeb in vertices(network)
            if nodeb>nodea
                @process purifier(sim, network, nodea, nodeb, purifier_wait_time, purifier_busy_time)
            end
        end
    end
end

const XX = X⊗X
const ZZ = Z⊗Z
const YY = Y⊗Y

@resumable function purifier(
    sim::Environment,  # The scheduler for all simulation events
    network,           # The graph of quantum nodes
    nodea,             # One of the nodes on which the pairs to be purified rest
    nodeb,             # The other such node
    purifier_wait_time,# The wait time in case there are no pairs available for purification
    purifier_busy_time # The duration of the purification circuit
    )
    nround = 0
    while true
        pairs_of_bellpairs = findqubitstopurify(network, nodea, nodeb)
        if isnothing(pairs_of_bellpairs)
            @yield timeout(sim, purifier_wait_time)
            continue
        end
        # pairs_of_bellpairs = pairs_of_bellpairs::NTuple{4, QueryOnRegResult} # is this needed?
        qa1, qa2, qb1, qb2 = pairs_of_bellpairs
        @yield lock(qa1.slot) & lock(qa2.slot) & lock(qb1.slot) & lock(qb2.slot)
        @yield timeout(sim, purifier_busy_time)
        purifyerror = (:X, :Z)[nround%2+1]
        purificationcircuit = Purify2to1(purifyerror)
        success = purificationcircuit(qa1.slot, qb1.slot, qa2.slot, qb2.slot)
        if !success
            untag!(qa1.slot, qa1.id)
            untag!(qb1.slot, qb1.id)
            @info "$(round(now(sim), digits=2)): failed purification at $(nodea):$(qa1.slot.idx) & $(qa2.slot.idx) and $(nodeb):$(qb1.slot.idx) & $(qb2.slot.idx)"
        else
            nround += 1
            @info "$(round(now(sim), digits=2)): purification at $(nodea):$(qa1.slot.idx) $(nodeb):$(qb1.slot.idx) by sacrifice of $(nodea):$(qa2.slot.idx) $(nodeb):$(qb2.slot.idx)"
        end
        untag!(qa2.slot, qa2.id)
        untag!(qb2.slot, qb2.id)
        unlock(qa1.slot); unlock(qa2.slot); unlock(qb1.slot); unlock(qb2.slot)
    end
end

function findqubitstopurify(network, nodea, nodeb)
    rega = network[nodea]
    regb = network[nodeb]
    results_a = queryall(rega, EntanglementCounterpart, nodeb, ❓; locked=false, assigned=true)
    if length(results_a) >= 2
        qa1, qa2 = results_a[end-1], results_a[end]
        qb1 = query(regb, EntanglementCounterpart, nodea, qa1.slot.idx; locked=false, assigned=true)
        qb2 = query(regb, EntanglementCounterpart, nodea, qa2.slot.idx; locked=false, assigned=true)
        @assert !isnothing(qb1) && !isnothing(qb2)
        return qa1, qa2, qb1, qb2
    else
        return nothing
    end
end

config = Configuration([2, 3, 4, 3, 4], 10.0, 0.98, 3.0)
setup(config)