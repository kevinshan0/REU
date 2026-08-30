# Discrete-time linear-repeater-chain simulator with *delayed classical
# information*.
#
# The global-knowledge model (environment.py / policy.py, physically replayed by
# simulate.jl) lets every node see the whole chain instantly. This file drops
# that assumption: a node knows its own slots exactly, and learns about
# everything else only through messages that travel one hop per timestep.
#
# Why this is a hand-rolled tick loop rather than QuantumSavory's ProtocolZoo:
# EntanglerProt/SwapperProt/CutoffProt are asynchronous ConcurrentSim processes
# that poll, lock slots, and retry on their own schedules (see
# ~/.julia/packages/QuantumSavory/*/src/ProtocolZoo/). They cannot be made to
# follow an arbitrary per-node policy table, and they have no notion of the
# strict intra-timestep phase order required here (age -> LLEG -> receive ->
# swap -> cutoff -> transmit). simulate.jl already made the same call for the
# same reason. QuantumSavory is still doing all of the physics: `Register`s with
# `Depolarization` backgrounds, `DepolarizedBellPair` initialization,
# `EntanglementSwap`, and `observable` for the end-to-end fidelity.
#
# The classical-messaging design *is* borrowed from ProtocolZoo, in miniature:
# a swap announces its outcome to whoever it believed was on each far end
# (EntanglementUpdateX/Z), a discard announces a dead link (EntanglementDelete),
# and a node that has already swapped a slot away keeps a re-addressing entry so
# that late messages get forwarded onward (EntanglementHistory). Occupancy
# stamps play the role of ProtocolZoo's `EntanglementID`: they stop a message
# about a long-dead link from being applied to a slot that has since been reused.
#
# Qubit indexing is flat and 1-based, `1:2n-2` (NOTE: simulate.jl uses the same
# layout but 0-based, so its index i is this file's i+1):
#   node 1     owns {1}                    (right-facing)
#   node j     owns {2j-2, 2j-1}           (left-facing, right-facing)
#   node n     owns {2n-2}                 (left-facing)
# Elementary link k joins qubits (2k-1, 2k), i.e. the two slots facing each
# other across edge k. Odd index = right-facing, even index = left-facing.

using QuantumSavory
using QuantumSavory.CircuitZoo: EntanglementSwap
using QuantumSavory.StatesZoo: DepolarizedBellPair
using JSON3
using Random
using Statistics

const PERFECT_PAIR = (Z1 ⊗ Z1 + Z2 ⊗ Z2) / sqrt(2)

const TRIALS = 1000

"""How many timesteps of staleness the tabular observation encoding
distinguishes before saturating. MUST match `S_MAX` in info_delay_solver.py --
the two have to agree on the lookup key or the solved policy silently degrades
into its SWAP-ASAP fallback."""
const S_MAX = 2

# --------------------------------------------------------------------------- #
# --------------------------------  PARAMETERS  ----------------------------- #
# --------------------------------------------------------------------------- #

struct Params
    "number of nodes in the chain, including the two end nodes"
    n::Int
    "probability that one LLEG attempt on an elementary link succeeds"
    p::Float64
    "probability that an entanglement swap succeeds"
    p_s::Float64
    "fidelity of a freshly generated elementary link"
    F_new::Float64
    "end-to-end fidelity the retention time is budgeted against"
    F_min::Float64
    "depolarization time constant of the memories"
    τ::Float64
    "physical duration of one timestep (memories decay in physical time)"
    dt::Float64
    "retention time, in timesteps: a qubit is discarded once its age reaches it"
    t_cut::Int
    "give up on an episode after this many timesteps"
    max_steps::Int
end

"""Retention time (in physical time units) from the state-based cutoff of
arXiv:2207.06533: pick `t_cut` so that even the worst case -- all `n-1`
elementary links sitting at the cutoff age when they are swapped together --
still clears `F_min`. In terms of the depolarization parameter `p = (4F-1)/3`,
which is multiplicative under swapping and decays as `exp(-t/τ)`, that is
`(p_new * exp(-t_cut/τ))^(n-1) = p_min`."""
function retention_time(F_new, F_min, τ, n)
    p_new = (4F_new - 1) / 3
    p_min = (4F_min - 1) / 3
    return -τ * log(p_min^(1 / (n - 1)) / p_new)
end

function Params(; n=5, p=0.9, p_s=0.9, F_new=1.0, F_min=0.98, τ=10.0, dt=1.0,
                  t_cut=nothing, max_steps=10_000)
    steps = if isnothing(t_cut)
        floor(Int, retention_time(F_new, F_min, τ, n) / dt)
    else
        t_cut
    end
    if steps < 1
        @warn "the fidelity budget allows less than one timestep of storage " *
              "(t_cut = $(retention_time(F_new, F_min, τ, n)) time units, dt = $dt); " *
              "clamping to 1 timestep -- raise τ, lower F_min, or shrink dt" maxlog=1
        steps = 1
    end
    return Params(n, p, p_s, F_new, F_min, τ, dt, steps, max_steps)
end

# --------------------------------------------------------------------------- #
# ------------------------  FLAT INDEX <-> REGISTER/SLOT  ------------------- #
# --------------------------------------------------------------------------- #

"""Node (1-based register number) owning flat qubit `q`."""
owner(q) = (q ÷ 2) + 1

"""Slot index within that node's register. Middle nodes keep their left-facing
qubit in slot 1 and their right-facing qubit in slot 2; end nodes have one slot."""
slotof(q, n) = (q == 1 || q == 2n - 2) ? 1 : (iseven(q) ? 1 : 2)

"""Flat indices of the qubits `node` owns, left-facing first."""
ownqubits(node, n) = node == 1 ? (1,) : node == n ? (2n - 2,) : (2node - 2, 2node - 1)

# --------------------------------------------------------------------------- #
# -------------------------------  GROUND TRUTH  ---------------------------- #
# --------------------------------------------------------------------------- #

"""The true state of the chain, which no single node ever gets to see.

`age[q] == -1` means the slot is empty. `partner[q] == 0` means the qubit is not
entangled with anything: combined with `age[q] >= 0` that is a *hanging* qubit --
still stored, still ageing, still occupying the slot, but useless, and its owner
has no local way of telling. `gen[q]` is bumped every time the slot is filled, so
that a message about a link that died long ago cannot be applied to whatever
occupies the slot now."""
mutable struct Chain
    n::Int
    nq::Int
    registers::Vector{Register}
    age::Vector{Int}
    partner::Vector{Int}
    gen::Vector{Int}
    gencount::Int
end

function Chain(par::Params)
    nq = 2par.n - 2
    registers = Register[]
    for node in 1:par.n
        slots = (node == 1 || node == par.n) ? 1 : 2
        traits = [Qubit() for _ in 1:slots]
        reprs = [QuantumOpticsRepr() for _ in 1:slots]
        bg = [Depolarization(par.τ) for _ in 1:slots]
        push!(registers, Register(traits, reprs, bg))
    end
    return Chain(par.n, nq, registers, fill(-1, nq), zeros(Int, nq), zeros(Int, nq), 0)
end

qref(c::Chain, q) = c.registers[owner(q)][slotof(q, c.n)]

occupied(c::Chain, q) = c.age[q] >= 0
hanging(c::Chain, q) = occupied(c, q) && c.partner[q] == 0

"""True when the two end nodes share a virtual link. This is ground truth: the
end nodes themselves do not find out until the swap announcements reach them."""
delivered(c::Chain) = occupied(c, 1) && c.partner[1] == c.nq

"""Empty a slot both physically and in the bookkeeping. `traceout!` is a no-op on
an already-unassigned slot (e.g. one `EntanglementSwap` has just measured out),
so this is safe however the qubit came to be free."""
function free!(c::Chain, q)
    traceout!(qref(c, q))
    c.age[q] = -1
    c.partner[q] = 0
    return nothing
end

# --------------------------------------------------------------------------- #
# ---------------------------  WHAT A NODE KNOWS  --------------------------- #
# --------------------------------------------------------------------------- #

"""A classical message travelling along the chain, one hop per timestep.

`:update` says "the link you think you hold now ends at `new_remote` instead"
(a swap succeeded); `:delete` says "the link you think you hold is gone" (a swap
failed, or the far end was discarded). It is addressed to a *specific occupancy*
`(target, target_gen)` of a slot, so it is dropped if that slot has since been
recycled."""
struct Msg
    target::Int
    target_gen::Int
    kind::Symbol
    new_remote::Int
    new_remote_gen::Int
    author::Int
    t::Int
end

"""Everything one node believes, and nothing it has no way of knowing.

`partner`/`partner_gen`/`partner_asof` are the node's belief about the far end of
each of *its own* qubits -- the belief that can be wrong, and the reason a node
cannot tell a live link from a hanging qubit.

`chain_age`/`chain_asof` are the flooded ledger: every node re-asserts the age of
its own slots every timestep and passes the whole table to both neighbours, so
node `i`'s entry for a qubit at node `j` is `j`'s truth as of `t - |i-j|`.

`history` is the ProtocolZoo `EntanglementHistory` trick: after this node swaps
(or discards) an occupancy away, messages still in flight for it are re-addressed
to wherever the entanglement actually went, or dropped if it went nowhere."""
mutable struct NodeView
    node::Int
    n::Int
    partner::Vector{Int}
    partner_gen::Vector{Int}
    partner_asof::Vector{Int}
    chain_age::Vector{Int}
    chain_asof::Vector{Int}
    history::Dict{Tuple{Int,Int},Tuple{Int,Int}}
end

function NodeView(node, n)
    nq = 2n - 2
    return NodeView(node, n, zeros(Int, nq), zeros(Int, nq), fill(-1, nq),
                    fill(-1, nq), fill(-1, nq), Dict{Tuple{Int,Int},Tuple{Int,Int}}())
end

"""How old the flooded ledger implies qubit `q` is *now*: the age this node last
heard, plus the time since. `-1` when the node has never heard about `q`. This is
a projection, not a fact -- `q` may have been discarded or swapped in the
meantime, which is exactly the uncertainty a policy has to price in."""
function projected_age(v::NodeView, q, t)
    (v.chain_asof[q] < 0 || v.chain_age[q] < 0) && return -1
    return v.chain_age[q] + (t - v.chain_asof[q])
end

"""Handle a message that has arrived at this node. Returns `nothing` if the
message was applied or dropped, or the (possibly re-addressed) message that still
has to keep travelling."""
function receive!(c::Chain, v::NodeView, msg::Msg)
    key = (msg.target, msg.target_gen)
    if haskey(v.history, key)
        newtarget, newgen = v.history[key]
        newtarget == 0 && return nothing # that entanglement is dead, drop the message
        return Msg(newtarget, newgen, msg.kind, msg.new_remote, msg.new_remote_gen,
                   msg.author, msg.t)
    end
    owner(msg.target) == v.node || return msg # not ours, keep it moving
    # the addressed occupancy is gone (slot emptied or refilled since) -- stale
    (occupied(c, msg.target) && c.gen[msg.target] == msg.target_gen) || return nothing
    if msg.t > v.partner_asof[msg.target]
        isupdate = msg.kind === :update
        v.partner[msg.target] = isupdate ? msg.new_remote : 0
        v.partner_gen[msg.target] = isupdate ? msg.new_remote_gen : 0
        v.partner_asof[msg.target] = msg.t
    end
    return nothing
end

# --------------------------------------------------------------------------- #
# ------------------------------  THE ACTIONS  ------------------------------- #
# --------------------------------------------------------------------------- #

"""Entanglement generation on every elementary link whose two slots are free.

LLEG is attempted unconditionally and only shows up in the state if both slots
were available, which is the point of doing it this way: no node has to signal
the BSA that it is ready, so no half-timestep of signalling is charged. Success
is heralded, so both endpoints of the new link learn about it in the same
timestep -- this is the only knowledge in the model that is not delayed."""
function attempt_generation!(c::Chain, views, t, par::Params)
    for k in 1:(c.n - 1)
        qa, qb = 2k - 1, 2k
        (occupied(c, qa) || occupied(c, qb)) && continue
        rand() < par.p || continue

        ga, gb = c.gencount + 1, c.gencount + 2
        c.gencount += 2
        initialize!((qref(c, qa), qref(c, qb)), DepolarizedBellPair(F=par.F_new);
                    time=t * par.dt)
        c.age[qa] = 0; c.age[qb] = 0
        c.partner[qa] = qb; c.partner[qb] = qa
        c.gen[qa] = ga; c.gen[qb] = gb

        for (q, remote, remotegen) in ((qa, qb, gb), (qb, qa, ga))
            v = views[owner(q)]
            v.partner[q] = remote
            v.partner_gen[q] = remotegen
            v.partner_asof[q] = t
        end
    end
    return nothing
end

"""`node` fires a Bell-state measurement on its own two qubits.

The node commits to this from its *belief*, so all three outcomes are possible:

- both slots really do hold links: succeeds with probability `p_s`, joining the
  two far ends into one longer link; on failure both links are destroyed and both
  far ends are left hanging, unaware, for `|i-j|` timesteps.
- a slot is actually hanging (its far end was cut or swapped out from under it):
  the BSM still consumes both local qubits, and the *good* side's far end is left
  hanging too. This is the compounding effect the investigation is about -- a
  problem that was `j` hops from being resolvable is now `j+k` hops away.

Whatever happens, the announcements go to whoever the node *believed* was on each
far end. If that belief was stale, the message is addressed to a slot that has
moved on, and the `history` tables along the way re-address or drop it."""
function apply_swap!(c::Chain, views, outbox, node, t, par::Params)
    (node == 1 || node == c.n) && return nothing # end nodes have nothing to swap
    v = views[node]
    qL, qR = ownqubits(node, c.n)

    believedL, believedLgen = v.partner[qL], v.partner_gen[qL]
    believedR, believedRgen = v.partner[qR], v.partner_gen[qR]
    genL, genR = c.gen[qL], c.gen[qR]
    farL, farR = c.partner[qL], c.partner[qR]

    usable = occupied(c, qL) && occupied(c, qR) && farL != 0 && farR != 0
    success = usable && rand() < par.p_s

    if success
        refs = (qref(c, qL), qref(c, farL), qref(c, qR), qref(c, farR))
        uptotime!(refs, t * par.dt)
        EntanglementSwap()(refs...)
        # per CUTOFF-INFO.md this is a qubit-retention model, so the surviving
        # far ends keep their own ages -- the new link does not get a fresh clock
        c.partner[farL] = farR
        c.partner[farR] = farL
        free!(c, qL); free!(c, qR)
    else
        for (own, far) in ((qL, farL), (qR, farR))
            occupied(c, own) || continue
            far != 0 && (c.partner[far] = 0) # tracing out `own` leaves `far` mixed
            free!(c, own)
        end
    end

    v.partner[qL] = 0; v.partner[qR] = 0
    v.partner_asof[qL] = t; v.partner_asof[qR] = t
    v.history[(qL, genL)] = success ? (believedR, believedRgen) : (0, 0)
    v.history[(qR, genR)] = success ? (believedL, believedLgen) : (0, 0)

    kind = success ? :update : :delete
    if believedL != 0
        push!(outbox[node], Msg(believedL, believedLgen, kind,
                                success ? believedR : 0, success ? believedRgen : 0,
                                node, t))
    end
    if believedR != 0
        push!(outbox[node], Msg(believedR, believedRgen, kind,
                                success ? believedL : 0, success ? believedLgen : 0,
                                node, t))
    end
    return nothing
end

"""Drop qubit `q`, whether because its owner chose to or because it hit the
retention limit. The far end is left hanging and will not find out until the
announcement gets there."""
function apply_discard!(c::Chain, views, outbox, q, t)
    node = owner(q)
    v = views[node]
    believed, believedgen = v.partner[q], v.partner_gen[q]
    gen = c.gen[q]
    far = c.partner[q]

    far != 0 && (c.partner[far] = 0)
    free!(c, q)

    v.partner[q] = 0
    v.partner_asof[q] = t
    v.history[(q, gen)] = (0, 0)
    believed != 0 && push!(outbox[node], Msg(believed, believedgen, :delete, 0, 0, node, t))
    return nothing
end

"""Retention-time cutoff. A node only ever needs its own clock for this, so no
information delay is involved -- but the *consequences* are delayed, and the two
ends of a swapped link generally have different ages, so cutting one of them
manufactures a hanging qubit at the other end."""
function apply_cutoff!(c::Chain, views, outbox, t, par::Params)
    for q in 1:c.nq
        c.age[q] == par.t_cut && apply_discard!(c, views, outbox, q, t)
    end
    return nothing
end

"""Phase 3: drain each node's inbox. Anything not consumed here is queued for one
more hop in the transmit phase."""
function receive_messages!(c::Chain, views, inbox, forwards)
    for i in 1:c.n
        msgs = inbox[i]
        inbox[i] = Msg[]
        for msg in msgs
            onward = receive!(c, views[i], msg)
            isnothing(onward) || push!(forwards[i], onward)
        end
    end
    return nothing
end

"""Phase 6: everything a node says this timestep lands on its neighbours next
timestep, and nowhere further. Information about node `j` therefore reaches node
`i` after exactly `|i-j|` timesteps."""
function transmit!(c::Chain, views, inbox, outbox, forwards, t)
    # the flooded ledger: refresh what each node knows first-hand, then exchange
    # whole tables with both neighbours. Snapshots are taken before any merging so
    # that nothing travels more than one hop per timestep.
    for i in 1:c.n, q in ownqubits(i, c.n)
        views[i].chain_age[q] = c.age[q]
        views[i].chain_asof[q] = t
    end
    snapshots = [(copy(v.chain_age), copy(v.chain_asof)) for v in views]
    for i in 1:c.n, j in (i - 1, i + 1)
        1 <= j <= c.n || continue
        theirage, theirasof = snapshots[j]
        v = views[i]
        for q in 1:c.nq
            if theirasof[q] > v.chain_asof[q]
                v.chain_age[q] = theirage[q]
                v.chain_asof[q] = theirasof[q]
            end
        end
    end

    # targeted messages move exactly one hop towards the node that owns the slot
    # they are addressed to
    for i in 1:c.n
        for msg in Iterators.flatten((outbox[i], forwards[i]))
            dest = owner(msg.target)
            dest == i && continue
            push!(inbox[dest > i ? i + 1 : i - 1], msg)
        end
        empty!(outbox[i])
        empty!(forwards[i])
    end
    return nothing
end

# --------------------------------------------------------------------------- #
# --------------------------------  POLICIES  -------------------------------- #
# --------------------------------------------------------------------------- #

abstract type AbstractPolicy end

"""What one node does in one timestep: fire a swap on its own two qubits, and/or
discard some of its own qubits (given as flat indices). Same `(swap, discard)`
shape the distilled local policies in distill.py / simulate.jl already use."""
struct NodeAction
    swap::Bool
    discard::Vector{Int}
end

const NOOP = NodeAction(false, Int[])

"""What a node gets to see when it decides. Everything here is either local and
exact (`age`, and the protocol parameters every node is configured with) or
explicitly stamped with how stale it is. This is the observation an RL agent
would be handed -- keep it hashable-friendly and free of ground-truth leakage.

Per own qubit: `q` flat index, `age` (exact, `-1` if the slot is empty), `far_q`
the believed far-end qubit (`0` = believed unentangled), `far_node` its node,
`stale` how many timesteps ago that belief was last confirmed, and `far_age` the
age the flooded ledger projects for the far end right now (`-1` if unknown)."""
function observe(c::Chain, v::NodeView, node, t, par::Params)
    own = map(ownqubits(node, c.n)) do q
        farq = v.partner[q]
        (; q,
           age = c.age[q],
           far_q = farq,
           far_node = farq == 0 ? 0 : owner(farq),
           stale = v.partner_asof[q] < 0 ? -1 : t - v.partner_asof[q],
           far_age = farq == 0 ? -1 : projected_age(v, farq, t))
    end
    return (; node, n = c.n, t, t_cut = par.t_cut, own,
              chain_age = v.chain_age, chain_asof = v.chain_asof)
end

"""Swap the moment both slots are believed to hold a link. The baseline to beat:
it cannot tell a live link from a hanging qubit, so it happily burns a good link
against a dead one and pushes the damage further down the chain."""
struct SwapASAP <: AbstractPolicy end

function decide(::SwapASAP, obs)
    length(obs.own) == 2 || return NOOP
    ready = all(o -> o.age >= 0 && o.far_q != 0, obs.own)
    return ready ? NodeAction(true, Int[]) : NOOP
end

"""Swap-asap plus the two inferences the delayed ledger actually supports:

1. if the flooded ledger projects that the far end has already reached its own
   retention limit, this slot is (almost certainly) hanging -- free it now so
   LLEG can restart, instead of feeding it to a swap that is bound to fail and
   bound to hang somebody else's qubit;
2. optionally, do not swap links that are within `margin` of being cut anyway,
   on the theory that the resulting long link would die almost immediately
   having consumed two links.

**`margin` defaults to 0, i.e. rule 2 off, because measurement says rule 2 is
harmful.** A link at age exactly `t_cut` is on its final timestep, so swapping it
now is the only value it has left; any `margin > 0` refuses that last-chance swap.
At n=5, p_s=0.9 that costs (swap-asap / margin=1 / margin=0 expected delivery):

    p=0.4, t_cut=2:  13.085 / 21.193 / 13.085
    p=0.4, t_cut=4:   9.950 / 11.497 /  9.729
    p=0.9, t_cut=6:   4.678 /  4.604 /  4.538

-- ruinous at low `p`, where a discarded link takes ~1/p timesteps to replace and
throwing away a free option is expensive. At `margin=0` rule 1 stands on its own
and is a small win. The parameter is kept so the effect stays reproducible.

A deliberately simple hand-written alternative to SWAP-ASAP -- the point is to
have something to measure the solved policies against, not to be optimal."""
struct CutoffAwareSwap <: AbstractPolicy
    margin::Int
end

CutoffAwareSwap() = CutoffAwareSwap(0)

function decide(pol::CutoffAwareSwap, obs)
    discard = Int[]
    for o in obs.own
        o.age >= 0 || continue
        # `> t_cut`, not `>=`: the cutoff sweep runs in phase 5, so a far end
        # whose projected age is exactly `t_cut` is still alive while this
        # decision is being made -- it dies at the end of this timestep
        if o.far_q == 0 || (o.far_age > obs.t_cut)
            push!(discard, o.q)
        end
    end
    # a qubit is still usable at age `t_cut` (it is only swept away in phase 5),
    # so `margin == 0` means "swap anything alive" and `margin == k` means "leave
    # at least k more timesteps of life". The margin is clamped because demanding
    # more headroom than the retention time can ever offer would turn this into a
    # policy that refuses to swap any link that has been stored at all.
    margin = min(pol.margin, obs.t_cut - 1)
    swappable = length(obs.own) == 2 && isempty(discard) &&
        all(obs.own) do o
            o.age >= 0 && o.far_q != 0 &&
                o.age <= obs.t_cut - margin &&
                (o.far_age < 0 || o.far_age <= obs.t_cut - margin)
        end
    return NodeAction(swappable, discard)
end

"""Wraps any `obs -> NodeAction` callable. This is the hook the RL work plugs
into: train against `observe`'s output, then hand the greedy/sampled action
function to `run_trials` to get delivery times and fidelities out of the same
simulator the baselines were measured in."""
struct FunctionPolicy{F} <: AbstractPolicy
    f::F
end

decide(pol::FunctionPolicy, obs) = pol.f(obs)

"""The compact local observation a solved tabular policy is keyed on. Per own
qubit, left-facing first: exact `age` clipped at `t_cut` (`-1` = empty), hops to
the believed far end (`0` = believed unentangled), staleness of that belief
saturating at [`S_MAX`](@ref), and what the flooded ledger projects for the far
end (`0` never heard, `1` alive, `2` past its cutoff).

MUST stay in lockstep with `encode_observation` in info_delay_solver.py: this is
the shared key format, and the two implementations have no way of noticing if
they drift apart."""
function encode_observation(obs)
    parts = String[]
    for o in obs.own
        age = o.age < 0 ? -1 : min(o.age, obs.t_cut)
        if o.far_q == 0
            append!(parts, string.((age, 0, 0, 0)))
        else
            # floored at 1 so that dist == 0 means "believed unentangled" and
            # nothing else -- see the matching note in info_delay_solver.py
            dist = min(max(abs(o.far_node - obs.node), 1), obs.n - 1)
            stale = o.stale < 0 ? S_MAX : min(o.stale, S_MAX)
            status = o.far_age < 0 ? 0 : (o.far_age > obs.t_cut ? 2 : 1)
            append!(parts, string.((age, dist, stale, status)))
        end
    end
    return join(parts, ",")
end

"""A per-node lookup table solved by info_delay_solver.py's decentralized policy
iteration. One table per node, keyed on [`encode_observation`](@ref).

Tables only store the observations where the solved policy *disagrees* with the
policy JESP was seeded from, so anything missing falls back to that seed and an
empty table reproduces it exactly. The seed travels in the JSON's `fallback`
field; files written before seeding existed have no such field and are read as
`"swap-asap"`, which is what they were.

Getting this wrong is silent: a cutoff-aware-seeded table applied with a
swap-asap fallback would quietly behave as neither."""
struct TabularPolicy <: AbstractPolicy
    tables::Vector{Dict{String,NodeAction}}
    fallback::AbstractPolicy
end

const TABULAR_FALLBACKS = Dict{String,AbstractPolicy}(
    "swap-asap" => SwapASAP(),
    "cutoff-aware" => CutoffAwareSwap(0),
)

function TabularPolicy(path::AbstractString)
    data = JSON3.read(read(path, String))
    fallback_name = String(get(data, :fallback, "swap-asap"))
    haskey(TABULAR_FALLBACKS, fallback_name) || error(
        "policy at $path declares fallback $(fallback_name), which this simulator " *
        "does not know; expected one of $(collect(keys(TABULAR_FALLBACKS)))")
    data.s_max == S_MAX || error(
        "policy at $path was solved with S_MAX=$(data.s_max) but this simulator " *
        "uses S_MAX=$S_MAX; the observation keys would not line up")
    length(data.nodes) == data.n || error(
        "policy at $path has $(length(data.nodes)) node tables for an $(data.n)-node chain")
    tables = Dict{String,NodeAction}[]
    for (node, entries) in enumerate(data.nodes)
        own = ownqubits(node, data.n)
        table = Dict{String,NodeAction}()
        for (key, action) in pairs(entries)
            # the solver stores discards as positions into `ownqubits`, which is
            # node-agnostic; they become flat qubit indices here
            table[String(key)] = NodeAction(action.swap, [own[i + 1] for i in action.discard])
        end
        push!(tables, table)
    end
    return TabularPolicy(tables, TABULAR_FALLBACKS[fallback_name])
end

function decide(pol::TabularPolicy, obs)
    hit = get(pol.tables[obs.node], encode_observation(obs), nothing)
    return isnothing(hit) ? decide(pol.fallback, obs) : hit
end

# --------------------------------------------------------------------------- #
# ---------------------------  EPISODE / TRIALS  ----------------------------- #
# --------------------------------------------------------------------------- #

"""Fidelity of the delivered end-to-end pair, relative to the |Φ⁺⟩ that
`DepolarizedBellPair` targets. The Pauli corrections from the swaps are applied
by `EntanglementSwap` immediately rather than being carried as a classical frame
to the end nodes; that is the usual bookkeeping simplification and does not
change the fidelity."""
function e2e_fidelity(c::Chain, t, par::Params)
    obs = observable([c.registers[1], c.registers[c.n]], [1, 1],
                     SProjector(PERFECT_PAIR); time=t * par.dt)
    return isnothing(obs) ? NaN : real(obs)
end

"""One episode. Returns `(delivery_time, fidelity)`, or `nothing` if the chain
failed to deliver within `par.max_steps`.

The phases inside a timestep are exactly the order in the model spec:

1. every stored qubit ages by one
2. LLEG on every elementary link that has both slots free
3. classical information from the neighbours lands
4. each node picks an action from its own delayed view; swaps fire, then the
   voluntary discards those actions asked for
5. anything that has reached the retention limit is discarded
6. every node tells its neighbours what it now knows

Delivery is checked after generation (an elementary link can be the end-to-end
link when `n == 2`) and after swapping, before the cutoff sweep gets a chance to
tear down a link that has just been completed."""
function run_trial(policy::AbstractPolicy, par::Params)
    c = Chain(par)
    views = [NodeView(i, par.n) for i in 1:par.n]
    inbox = [Msg[] for _ in 1:par.n]
    outbox = [Msg[] for _ in 1:par.n]
    forwards = [Msg[] for _ in 1:par.n]

    for t in 1:par.max_steps
        for q in 1:c.nq
            c.age[q] >= 0 && (c.age[q] += 1)
        end

        attempt_generation!(c, views, t, par)
        delivered(c) && return t, e2e_fidelity(c, t, par)

        receive_messages!(c, views, inbox, forwards)

        actions = [decide(policy, observe(c, views[i], i, t, par)) for i in 1:par.n]
        # applied in node order: a swap reads the *current* far ends, so two
        # adjacent nodes swapping in the same timestep compose correctly, and a
        # failure at one of them correctly poisons the other's qubit
        for i in 1:par.n
            actions[i].swap && apply_swap!(c, views, outbox, i, t, par)
        end
        delivered(c) && return t, e2e_fidelity(c, t, par)
        for i in 1:par.n, q in actions[i].discard
            occupied(c, q) && apply_discard!(c, views, outbox, q, t)
        end

        apply_cutoff!(c, views, outbox, t, par)
        transmit!(c, views, inbox, outbox, forwards, t)
    end
    return nothing
end

"""Run `trials` independent episodes and report the averages.

`avg_delivery_time` and `avg_fidelity` are taken over the episodes that actually
delivered; `timeouts` counts the rest, and should be zero for any sane parameter
set (a nonzero count means the averages are conditioned on success and are
optimistic)."""
function run_trials(policy::AbstractPolicy=SwapASAP(); trials=TRIALS,
                    par::Params=Params(), seed=nothing)
    isnothing(seed) || Random.seed!(seed)
    times = Int[]
    fidelities = Float64[]
    timeouts = 0
    for _ in 1:trials
        result = run_trial(policy, par)
        if isnothing(result)
            timeouts += 1
        else
            push!(times, result[1])
            push!(fidelities, result[2])
        end
    end
    return (; trials, delivered=length(times), timeouts,
              avg_delivery_time = isempty(times) ? NaN : mean(times),
              avg_fidelity = isempty(fidelities) ? NaN : mean(fidelities),
              delivery_times = times, fidelities)
end

function main()
    par = Params()
    physical = retention_time(par.F_new, par.F_min, par.τ, par.n)
    println("n = $(par.n), p = $(par.p), p_s = $(par.p_s), " *
            "F_new = $(par.F_new), F_min = $(par.F_min), τ = $(par.τ)")
    println("retention time = $(round(physical, digits=4)) time units = " *
            "$(par.t_cut) timesteps at dt = $(par.dt), trials = $TRIALS")
    for (name, policy) in (("swap-asap", SwapASAP()),
                           ("cutoff-aware", CutoffAwareSwap()))
        r = run_trials(policy; trials=TRIALS, par, seed=1234)
        println("$(rpad(name, 14)) delivered $(r.delivered)/$(r.trials)  " *
                "avg. delivery time $(round(r.avg_delivery_time, digits=3))  " *
                "avg. fidelity $(round(r.avg_fidelity, digits=5))")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
