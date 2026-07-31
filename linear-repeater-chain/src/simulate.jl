# Physical (fidelity-aware) simulator for the optimal qubit-age cutoff policy
# computed by policy.py / environment.py.
#
# environment.py's MDP is a pure success/failure model with no notion of
# fidelity -- it optimizes expected delivery time only. This script drives
# QuantumSavory registers tick-by-tick under the *exact* state->action policy
# table handed to it (see simulate.py), rather than QuantumSavory's built-in
# ProtocolZoo protocols (SwapperProt/CutoffProt), which are generic
# age-limit/retry-loop heuristics that can't follow an arbitrary lookup
# table. A bare `Register` works standalone with explicit `time=` kwargs, so
# no ConcurrentSim/RegisterNet event loop is needed here -- this is a single
# synchronous controller, not a set of competing async protocols.
#
# Flat qubit indexing mirrors environment.py's State.initial(n) exactly
# (0-indexed, as plain Julia Ints -- only converted to 1-indexed Julia array
# access at the point of touching an array): index 0 is node 0's lone qubit;
# node j (1..n-2) owns flat indices 2j-1, 2j; index 2n-3 is node n-1's lone
# qubit. Swap action i (odd) means "the node owning qubits i,i+1 swaps them".
# Generation pairs are (2k, 2k+1), the two qubits facing each other across an
# edge.

using QuantumSavory
using QuantumSavory.CircuitZoo: EntanglementSwap
using QuantumSavory.StatesZoo: DepolarizedBellPair
using JSON3
using Statistics

const PERFECT_PAIR = (Z1 ⊗ Z1 + Z2 ⊗ Z2) / sqrt(2)

# --------------------------------------------------------------------------- #
# ------------------------  FLAT INDEX <-> REGISTER/SLOT  ------------------- #
# --------------------------------------------------------------------------- #

"""Julia (1-indexed) register number owning flat qubit `flatidx` (0-indexed),
matching environment.py's action_to_nodes convention (node = (i+1)/2)."""
node_of(flatidx) = (flatidx + 1) ÷ 2 + 1

"""Slot index (1 or 2) within that register."""
function slot_of(flatidx, n)
    if flatidx == 0 || flatidx == 2n - 3
        return 1
    elseif isodd(flatidx)
        return 1
    else
        return 2
    end
end

qslot(registers, flatidx, n) = registers[node_of(flatidx)][slot_of(flatidx, n)]

function build_registers(n, tau)
    registers = Register[]
    for node in 1:n
        slots = (node == 1 || node == n) ? 1 : 2
        traits = [Qubit() for _ in 1:slots]
        reprs = [QuantumOpticsRepr() for _ in 1:slots]
        bg = [Depolarization(tau) for _ in 1:slots]
        push!(registers, Register(traits, reprs, bg))
    end
    return registers
end

# --------------------------------------------------------------------------- #
# ------------------------  STATE <-> HASHABLE FORM  ------------------------ #
# --------------------------------------------------------------------------- #

"""Reconstruct links (pairs of flat indices) from age/hanging, mirroring
environment.py's State.links_of exactly (even flat index = pushed, odd =
completes a link with the top of the stack)."""
function links_of(age, hanging)
    links = Tuple{Int,Int}[]
    stack = Int[]
    for idx in 0:length(age)-1
        (age[idx+1] == -1 || hanging[idx+1]) && continue
        if iseven(idx)
            push!(stack, idx)
        else
            push!(links, (pop!(stack), idx))
        end
    end
    return links
end

function link_partner(age, hanging, q)
    age[q+1] == -1 && return nothing
    for (a, b) in links_of(age, hanging)
        a == q && return b
        b == q && return a
    end
    return nothing
end

function check_e2e(age, hanging)
    nq = length(age)
    (age[1] == -1 || age[nq] == -1) && return false
    return link_partner(age, hanging, 0) == nq - 1
end

"""Empty a qubit both classically (age/hanging bookkeeping) and physically.
traceout! is a no-op on an already-unassigned slot (e.g. the two qubits
EntanglementSwap() itself already project_traceout!'d as part of performing
the swap), so it's always safe to call here regardless of how the qubit
became free."""
function clear!(age, hanging, registers, n, q)
    age[q+1] = -1
    hanging[q+1] = false
    traceout!(qslot(registers, q, n))
end

# --------------------------------------------------------------------------- #
# --------------------------  MDP TRANSITION STEP  --------------------------- #
# --------------------------------------------------------------------------- #

"""Apply one swap action at flat index `flatidx` (its owning node swaps its
own qubits flatidx, flatidx+1), mirroring
environment.py's get_entanglement_swap_transitions_for_action for a single
sampled outcome instead of enumerating every combination: the classical
outcome (success / failure / guard) is decided first against age/hanging,
exactly matching that function's branches, and only then realized physically
on the registers."""
function apply_swap_action!(age, hanging, registers, n, flatidx, p_s, t)
    qL, qR = flatidx, flatidx + 1
    left_ok = age[qL+1] != -1 && !hanging[qL+1]
    right_ok = age[qR+1] != -1 && !hanging[qR+1]

    if left_ok && right_ok
        if rand() < p_s
            leftremote = link_partner(age, hanging, qL)
            rightremote = link_partner(age, hanging, qR)
            regL, regLeftRemote = qslot(registers, qL, n), qslot(registers, leftremote, n)
            regR, regRightRemote = qslot(registers, qR, n), qslot(registers, rightremote, n)
            uptotime!((regL, regLeftRemote, regR, regRightRemote), t)
            EntanglementSwap()(regL, regLeftRemote, regR, regRightRemote)
            clear!(age, hanging, registers, n, qL)
            clear!(age, hanging, registers, n, qR)
        else
            left = link_partner(age, hanging, qL)
            right = link_partner(age, hanging, qR)
            clear!(age, hanging, registers, n, qL)
            clear!(age, hanging, registers, n, qR)
            left !== nothing && clear!(age, hanging, registers, n, left)
            right !== nothing && clear!(age, hanging, registers, n, right)
        end
    else
        for q in (qL, qR)
            if age[q+1] != -1
                partner = link_partner(age, hanging, q)
                clear!(age, hanging, registers, n, q)
                partner !== nothing && clear!(age, hanging, registers, n, partner)
            end
        end
    end
end

"""Discard a single qubit, mirroring State.get_transitions_for_action's
discard handling: the qubit is cleared, its (former) partner becomes
hanging rather than being cleared itself."""
function apply_discard_action!(age, hanging, registers, n, q)
    partner = link_partner(age, hanging, q)
    clear!(age, hanging, registers, n, q)
    partner !== nothing && (hanging[partner+1] = true)
end

"""Empty any qubit whose age has reached `cutoff`, mirroring
environment.py's State.cutoff: a hanging qubit is discarded on its own; a
linked qubit is discarded along with its partner."""
function apply_cutoff!(age, hanging, registers, n, cutoff)
    links = links_of(age, hanging)
    involved = Int[]
    for idx in 0:length(age)-1
        if age[idx+1] == cutoff
            if hanging[idx+1]
                push!(involved, idx)
            else
                for (a, b) in links
                    (idx == a || idx == b) && append!(involved, (a, b))
                end
            end
        end
    end
    for idx in involved
        clear!(age, hanging, registers, n, idx)
    end
end

"""Fidelity of the end-to-end Bell pair once qubit 0 and the last qubit are
linked, relative to the standard |Phi+> Bell state DepolarizedBellPair
targets."""
function e2e_fidelity(registers, n, nq, t)
    node0, slot0 = node_of(0), slot_of(0, n)
    nodeL, slotL = node_of(nq - 1), slot_of(nq - 1, n)
    return real(observable([registers[node0], registers[nodeL]], [slot0, slotL],
                            SProjector(PERFECT_PAIR); time=Float64(t)))
end

"""Independent Bernoulli(p) entanglement generation attempt on every
currently-empty adjacent boundary pair (2k, 2k+1), mirroring
environment.py's get_entanglement_generation_transitions (which also only
checks age == -1 for eligibility: a hanging qubit is never age -1, so it's
excluded without an explicit hanging check)."""
function attempt_generation!(age, registers, n, p, F_new, t)
    for k in 0:(n-2)
        i, j = 2k, 2k + 1
        if age[i+1] == -1 && age[j+1] == -1 && rand() < p
            slotA, slotB = qslot(registers, i, n), qslot(registers, j, n)
            initialize!((slotA, slotB), DepolarizedBellPair(F=F_new); time=t)
            age[i+1] = 0
            age[j+1] = 0
        end
    end
end

# --------------------------------------------------------------------------- #
# ------------------------------  POLICY  ------------------------------------ #
# --------------------------------------------------------------------------- #

function sample_action(probs)
    r, cum = rand(), 0.0
    for (i, p) in enumerate(probs)
        cum += p
        r <= cum && return i
    end
    return length(probs)
end

function load_policy(path)
    data = JSON3.read(read(path, String))
    n = data.n
    policy_dict = Dict{NTuple,Tuple}()
    for s in data.states
        key = Tuple((Int(qh[1]), Bool(qh[2])) for qh in s.qubits)
        policy_dict[key] = (Float64.(s.policy), collect(s.actions))
    end
    return n, policy_dict
end

"""Flat qubit indices (0-indexed) owned by `node` (1-indexed, Julia register
convention), mirroring distill.py's node_qubits."""
function own_qubits(n, node)
    node == 1 && return (0,)
    node == n && return (2n - 3,)
    return (2node - 3, 2node - 2)
end

"""Load a distilled local per-node policy (see distill.py's
build_local_policy_json): one lookup table per node, each keyed on that
node's own qubits' (age, hanging) and mapping to a (probs, candidates) pair
-- candidates is a list of (swap, discard qubits) options, sampled from with
sample_action exactly like the global policy's (probs, actions). The greedy
distillation exports a single candidate at probability 1 per local state; the
probabilistic distillation generally exports several."""
function load_local_policy(path)
    data = JSON3.read(read(path, String))
    n = data.n
    node_tables = Dict{NTuple,Tuple}[]
    for node_entries in data.nodes
        table = Dict{NTuple,Tuple}()
        for e in node_entries
            key = Tuple((Int(qh[1]), Bool(qh[2])) for qh in e.qubits)
            probs = Float64[a.prob for a in e.actions]
            candidates = [(Bool(a.swap), collect(Int, a.discard)) for a in e.actions]
            table[key] = (probs, candidates)
        end
        push!(node_tables, table)
    end
    return n, node_tables
end

# --------------------------------------------------------------------------- #
# ---------------------------  EPISODE / MAIN  -------------------------------- #
# --------------------------------------------------------------------------- #

function run_episode!(registers, policy_dict, n, nq, p, p_s, cutoff, F_new, allow_discard)
    age = fill(-1, nq)
    hanging = fill(false, nq)

    t = 0
    while true
        t += 1
        key = Tuple((age[i], hanging[i]) for i in 1:nq)
        entry = get(policy_dict, key, nothing)
        entry === nothing && error("policy has no entry for state $key")
        probs, actions = entry
        chosen = actions[sample_action(probs)]

        swap_flats, discard_flats = allow_discard ? (chosen.swap, chosen.discard) : (chosen, ())

        for flatidx in swap_flats
            apply_swap_action!(age, hanging, registers, n, flatidx, p_s, Float64(t))
        end
        for q in discard_flats
            apply_discard_action!(age, hanging, registers, n, q)
        end

        check_e2e(age, hanging) && return t, e2e_fidelity(registers, n, nq, t)

        apply_cutoff!(age, hanging, registers, n, cutoff)
        for i in 1:nq
            age[i] != -1 && (age[i] += 1)
        end
        attempt_generation!(age, registers, n, p, F_new, Float64(t))

        # A link formed by generation itself (e.g. the n=2 case, or the last
        # swap-eligible pair collapsing directly to e2e) is terminal for free
        # -- environment.py's step() charges no extra time slot to notice it,
        # since the resulting state's terminality is checked lazily wherever
        # it's next visited, not by taking another action.
        check_e2e(age, hanging) && return t, e2e_fidelity(registers, n, nq, t)
    end
end

"""Like run_episode!, but each node's action is decided independently from
its own qubits' state (via node_tables), rather than looked up from a single
global state->action table: each node samples its own (swap, discard) from
its own local candidate list every tick, exactly like the global policy
samples a joint action from its own (probs, actions) (see sample_action) --
just one independent draw per node instead of one draw for the whole chain.
The per-node swap/discard decisions are simply unioned into the same
swap_flats/discard_flats lists run_episode! would have built -- ownership of
qubits never overlaps between nodes, so there's no conflict to resolve, only
an ordering to fix (sorted ascending, matching the global policy's
action-application order) for determinism."""
function run_episode_local!(registers, node_tables, n, nq, p, p_s, cutoff, F_new)
    age = fill(-1, nq)
    hanging = fill(false, nq)

    t = 0
    while true
        t += 1
        swap_flats = Int[]
        discard_flats = Int[]
        for node in 1:n
            own = own_qubits(n, node)
            key = Tuple((age[i+1], hanging[i+1]) for i in own)
            entry = get(node_tables[node], key, nothing)
            entry === nothing && error("local policy has no entry for node $node state $key")
            probs, candidates = entry
            swap, discard = candidates[sample_action(probs)]
            swap && push!(swap_flats, own[1])
            append!(discard_flats, discard)
        end
        sort!(swap_flats)
        sort!(discard_flats)

        for flatidx in swap_flats
            apply_swap_action!(age, hanging, registers, n, flatidx, p_s, Float64(t))
        end
        for q in discard_flats
            apply_discard_action!(age, hanging, registers, n, q)
        end

        check_e2e(age, hanging) && return t, e2e_fidelity(registers, n, nq, t)

        apply_cutoff!(age, hanging, registers, n, cutoff)
        for i in 1:nq
            age[i] != -1 && (age[i] += 1)
        end
        attempt_generation!(age, registers, n, p, F_new, Float64(t))

        check_e2e(age, hanging) && return t, e2e_fidelity(registers, n, nq, t)
    end
end

"""Runs `episodes` fresh episodes via `step_fn(registers)` (a closure over
whichever policy representation -- global or local -- is in play), and
collects (fidelity, 0-indexed delivery time) across all of them."""
function run_episodes(step_fn, n, tau, episodes)
    fidelities = Float64[]
    delivery_times = Int[]
    for _ in 1:episodes
        registers = build_registers(n, tau)
        t, fid = step_fn(registers)
        push!(fidelities, fid)
        # -1 to match policy.py / the paper's 0-indexed delivery-time
        # convention (delivering on the very first attempt is time slot 0,
        # not 1) -- see Agent's docstring in policy.py.
        push!(delivery_times, t - 1)
    end
    return fidelities, delivery_times
end

function parse_opts(argv)
    opts = Dict{String,String}()
    i = 1
    while i <= length(argv)
        opts[argv[i][3:end]] = argv[i+1]
        i += 2
    end
    return opts
end

function main()
    policy_json, result_json = ARGS[1], ARGS[2]
    opts = parse_opts(ARGS[3:end])

    p = parse(Float64, opts["p"])
    p_s = parse(Float64, opts["p_s"])
    cutoff = parse(Int, opts["cutoff"])
    episodes = parse(Int, opts["episodes"])
    F_new = parse(Float64, opts["F_new"])
    tau = parse(Float64, opts["tau"])

    if haskey(opts, "local")
        n, node_tables = load_local_policy(policy_json)
        nq = 2n - 2
        fidelities, delivery_times = run_episodes(n, tau, episodes) do registers
            run_episode_local!(registers, node_tables, n, nq, p, p_s, cutoff, F_new)
        end
    else
        n, policy_dict = load_policy(policy_json)
        allow_discard = haskey(opts, "discard")
        nq = 2n - 2
        fidelities, delivery_times = run_episodes(n, tau, episodes) do registers
            run_episode!(registers, policy_dict, n, nq, p, p_s, cutoff, F_new, allow_discard)
        end
    end

    result = Dict(
        "avg_fidelity" => mean(fidelities),
        "episodes" => episodes,
        "avg_delivery_time" => mean(delivery_times),
    )
    open(io -> JSON3.write(io, result), result_json, "w")
    println("Episodes: $episodes, Avg. delivery time: $(mean(delivery_times)), Avg. fidelity: $(mean(fidelities))")
end

main()
