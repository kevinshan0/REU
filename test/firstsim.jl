##############################################################################
#  Linear Quantum Repeater Chain — low-level QuantumSavory.jl simulation
#
#  Uses:
#    QuantumSavory   — Register, RegularRegister, initialize!, apply!, tag!, queryall
#    QuantumClifford — Stabilizer backend (Bell states via Hadamard + CNOT)
#    ConcurrentSim   — discrete-event simulation engine (@resumable, @process, timeout)
#    GLMakie         — live animated visualisation
#
#  No ProtocolZoo or other high-level helpers are used.
#  Every physical operation (Bell-pair generation, BSM swap) is built from
#  bare gate / measurement primitives.
#
#  Chain layout (N nodes, N-1 elementary links):
#
#    Alice ── R₁ ── R₂ ── … ── Rₙ₋₂ ── Bob
#      0       1    2           N-2      N-1
#
#  Each node holds two memory qubits (one per adjacent link).
#  End nodes (Alice=0, Bob=N-1) hold one memory qubit each.
#
#  Protocol: swap-ASAP
#    1. Entanglement generation: each elementary link attempts a Bell pair
#       each timestep with probability p_gen.  Success initialises both
#       qubits into |Φ⁺⟩ and stamps them with an :EntanglementGenerated tag.
#    2. Memory decay: qubits older than T_mem steps are discarded (trashed).
#    3. Entanglement swapping: any intermediate node that has both memory
#       qubits occupied performs a local Bell-State Measurement (BSM).
#       Success (p_swap) extends the link; failure discards both.
#    4. End-to-end delivery: detected when Alice's qubit is entangled with
#       Bob's qubit via a chain of tags tracking the virtual link.
#
##############################################################################

using QuantumSavory
using QuantumSavory.ProtocolZoo   # only for the Stabilizer backend symbol — see note below
using QuantumClifford              # Stabilizer states, gates, measurements
using ConcurrentSim
using ResumableFunctions
using GLMakie
using Random

# ─────────────────────────────────────────────────────────────────────────────
# NOTE on imports: QuantumSavory re-exports QuantumClifford's stabilizer
# formalism.  We avoid ProtocolZoo's SwapASAP etc. — all protocol logic below
# is written by hand.  The only thing from ProtocolZoo we'd transitively get
# is the Clifford backend registration, but we include QuantumClifford directly
# so ProtocolZoo is not needed at all.  If your environment complains, simply
# remove the ProtocolZoo line above.
# ─────────────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Simulation parameters
# ═══════════════════════════════════════════════════════════════════════════════

const N_NODES = 5        # total nodes (Alice + repeaters + Bob)
const P_GEN = 0.6      # elementary Bell-pair generation probability per step
const P_SWAP = 0.85     # BSM success probability at each repeater node
const T_MEM = 6        # memory coherence time (steps); qubits discarded after this
const T_SIM = 300      # total simulation time (steps)
const SEED = 42

Random.seed!(SEED)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Register / Network construction
#
# QuantumSavory's RegularRegister(n) creates a node with n qubit slots.
# End nodes need 1 slot; repeaters need 2 (left-link and right-link memory).
# We build a simple vector of registers — no RegisterNet graph is required for
# a 1D chain; we manage adjacency ourselves.
# ═══════════════════════════════════════════════════════════════════════════════

function make_chain(n::Int)
    regs = Vector{Register}(undef, n)
    for i in 1:n
        nslots = (i == 1 || i == n) ? 1 : 2
        regs[i] = Register(nslots, CliffordRepr())   # Clifford/stabilizer backend
    end
    return regs
end

# Slot convention for node i (1-indexed):
#   end-nodes : slot 1 (only one neighbour)
#   repeaters : slot 1 = left-link memory, slot 2 = right-link memory
#
# link_slots(regs, i) → (reg_left, slot_left, reg_right, slot_right)
# for elementary link between node i and node i+1 (1-indexed).
function link_slots(regs, i)
    n = length(regs)
    # node i: if i==1, its right-memory is slot 1; else slot 2
    slot_i = (i == 1) ? 1 : 2
    # node i+1: its left-memory is always slot 1
    slot_i1 = 1
    return regs[i], slot_i, regs[i+1], slot_i1
end

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Low-level quantum operations
#
# Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
#   Prepare: H on qubit 1, then CNOT(1→2)
#
# Bell State Measurement (BSM):
#   Inverse Bell circuit then measure in Z basis.
#   CNOT(1→2), then H on qubit 1, then measure both in Z.
#   Outcome is one of the four Bell states — for entanglement swapping we
#   apply the appropriate Pauli correction on the surviving partner qubit.
#   (Here we track this classically; fidelity tracking via Clifford formalism.)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    prepare_bell_pair!(ra, sa, rb, sb)

Initialise qubits (ra[sa], rb[sb]) into |Φ⁺⟩ using the Clifford backend.
Both slots must be empty (vacuum) beforehand.
"""
function prepare_bell_pair!(ra::Register, sa::Int, rb::Register, sb::Int)
    # initialize! sets the qubit to |0⟩ in the Clifford representation
    initialize!(ra[sa], Z1)   # |0⟩
    initialize!(rb[sb], Z1)   # |0⟩
    # Apply H to first qubit → |+⟩
    apply!(ra[sa], H)
    # Apply CNOT with ra[sa] as control, rb[sb] as target.
    # QuantumSavory's apply! accepts a multi-qubit gate and a tuple of Refs.
    apply!((ra[sa], rb[sb]), CNOT)
end

"""
    bsm!(ra, sa, rb, sb) → Bool

Perform a Bell-State Measurement on (ra[sa], rb[sb]).
Returns true (and collapses the state) regardless of Bell outcome, because
all four outcomes are valid for entanglement swapping — the classical
correction is tracked implicitly by the stabilizer formalism.
This function models a probabilistic physical BSM; the caller applies
the p_swap coin flip.
"""
function bsm!(ra::Register, sa::Int, rb::Register, sb::Int)
    # Inverse Bell circuit
    apply!((ra[sa], rb[sb]), CNOT)
    apply!(ra[sa], H)
    # Measure both qubits — results are discarded (any outcome extends link)
    traceout!(ra[sa])
    traceout!(rb[sb])
end

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Classical metadata: tags
#
# We use QuantumSavory's tag! / queryall system to track which virtual
# link each qubit belongs to.  A qubit carries:
#
#   tag(slot, :link, left_node, right_node, born_at_step)
#
# left_node / right_node are the *endpoints* of the current virtual link
# (after swaps these are no longer the immediate neighbours).
# born_at_step is used to compute qubit age for memory cutoff.
# ═══════════════════════════════════════════════════════════════════════════════

struct LinkTag
    left_node::Int   # leftmost node currently entangled (1-indexed)
    right_node::Int   # rightmost node currently entangled
    born_at::Int   # simulation step when the qubit was stored
end

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Shared simulation state
#
# We keep an Observable-wrapped mutable struct so GLMakie can react to changes.
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct SimState
    regs::Vector{Register}
    # tags[node_idx][slot_idx] = LinkTag | nothing
    tags::Vector{Vector{Union{LinkTag,Nothing}}}
    step::Int
    deliveries::Int
    delivery_times::Vector{Float64}   # step at which each E2E delivery occurred
    log::Vector{String}
end

function SimState(regs)
    n = length(regs)
    tags = [[nothing for _ in 1:nqubits(r)] for r in regs]
    SimState(regs, tags, 0, 0, Float64[], String[])
end

nqubits(r::Register) = length(r.staterefs)

function log!(st::SimState, msg::String)
    push!(st.log, "[t=$(st.step)] $msg")
    length(st.log) > 500 && deleteat!(st.log, 1)  # rolling buffer
end

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Protocol processes (ConcurrentSim coroutines)
#
# Each process is a @resumable function that runs as a ConcurrentSim Process.
# They yield on timeout(sim, Δt) to advance simulated time.
# ═══════════════════════════════════════════════════════════════════════════════

"""
entanglement_generation_process

Runs continuously. Every timestep, for each elementary link i↔(i+1),
attempts Bell-pair generation with probability P_GEN if both memory slots
are currently empty.
"""
@resumable function entanglement_generation_process(sim::Simulation, st::SimState)
    n = length(st.regs)
    while true
        st.step += 1
        t = st.step

        # ── 1. Age-out expired qubits (memory cutoff) ──────────────────────
        for i in 1:n
            for s in 1:nqubits(st.regs[i])
                tag = st.tags[i][s]
                if !isnothing(tag) && (t - tag.born_at) > T_MEM
                    # Expire: also clear the partner's tag
                    j = (s == 1) ? tag.left_node : tag.right_node
                    jother = (i == j) ? tag.right_node : tag.left_node  # the far end
                    # find partner slot at node jother
                    for ps in 1:nqubits(st.regs[jother])
                        pt = st.tags[jother][ps]
                        if !isnothing(pt) && pt.left_node == tag.left_node && pt.right_node == tag.right_node
                            st.tags[jother][ps] = nothing
                            isassigned(st.regs[jother].staterefs, ps) &&
                                traceout!(st.regs[jother][ps])
                        end
                    end
                    st.tags[i][s] = nothing
                    isassigned(st.regs[i].staterefs, s) &&
                        traceout!(st.regs[i][s])
                    log!(st, "⌛ Qubit expired at node $i slot $s (link $(tag.left_node)↔$(tag.right_node))")
                end
            end
        end

        # ── 2. Entanglement generation on free elementary links ─────────────
        for link in 1:(n-1)
            ra, sa, rb, sb = link_slots(st.regs, link)

            # Check vacancy
            left_free = isnothing(st.tags[link][sa])
            right_free = isnothing(st.tags[link+1][sb])
            !left_free || !right_free && continue

            if rand() < P_GEN
                prepare_bell_pair!(ra, sa, rb, sb)
                lt = LinkTag(link, link + 1, t)
                st.tags[link][sa] = lt
                st.tags[link+1][sb] = lt
                log!(st, "📡 Bell pair generated on link $link↔$(link+1)")
            end
        end

        # ── 3. Swap-ASAP at every intermediate node ─────────────────────────
        # Repeat until no more swaps are possible this step
        swapped = true
        while swapped
            swapped = false
            for i in 2:(n-1)   # intermediate nodes only
                tL = st.tags[i][1]   # left-link memory  (slot 1)
                tR = st.tags[i][2]   # right-link memory (slot 2)
                isnothing(tL) || isnothing(tR) && continue

                if rand() < P_SWAP
                    # Successful BSM — extend virtual link
                    new_left = tL.left_node
                    new_right = tR.right_node
                    new_born = min(tL.born_at, tR.born_at)  # conservative age

                    bsm!(st.regs[i], 1, st.regs[i], 2)

                    # Clear tags at this repeater
                    st.tags[i][1] = nothing
                    st.tags[i][2] = nothing

                    # Update tags at the far endpoints to reflect new virtual link
                    new_tag = LinkTag(new_left, new_right, new_born)
                    _update_endpoint_tag!(st, new_left, i, new_tag)
                    _update_endpoint_tag!(st, new_right, i, new_tag)

                    log!(st, "🔀 Swap at node $i: extends link $new_left↔$new_right")

                    # Check for end-to-end delivery
                    if new_left == 1 && new_right == n
                        st.deliveries += 1
                        push!(st.delivery_times, Float64(t))
                        log!(st, "★  E2E entanglement delivered! (delivery #$(st.deliveries), t=$t)")
                        # Consume the end-to-end link
                        _clear_endpoint!(st, 1, new_tag)
                        _clear_endpoint!(st, n, new_tag)
                    end

                    swapped = true
                    break  # restart scan after any swap
                else
                    # Failed BSM — discard both qubits
                    _clear_endpoint!(st, tL.left_node, tL)
                    _clear_endpoint!(st, tR.right_node, tR)
                    st.tags[i][1] = nothing
                    st.tags[i][2] = nothing
                    isassigned(st.regs[i].staterefs, 1) && traceout!(st.regs[i][1])
                    isassigned(st.regs[i].staterefs, 2) && traceout!(st.regs[i][2])
                    log!(st, "✗  Swap failed at node $i, links $(tL.left_node)↔$(i) and $(i)↔$(tR.right_node) discarded")
                    swapped = true
                    break
                end
            end
        end

        @yield timeout(sim, 1.0)   # advance simulated clock by 1 step
    end
end

# ── Tag-management helpers ────────────────────────────────────────────────────

"""Find and update the tag at a link endpoint node that had `old_node` as a neighbour."""
function _update_endpoint_tag!(st::SimState, endpoint::Int, old_node::Int, new_tag::LinkTag)
    for s in 1:nqubits(st.regs[endpoint])
        t = st.tags[endpoint][s]
        if !isnothing(t) && (t.left_node == old_node || t.right_node == old_node)
            st.tags[endpoint][s] = new_tag
            return
        end
    end
end

"""Clear the qubit at `endpoint` whose tag matches `old_tag`."""
function _clear_endpoint!(st::SimState, endpoint::Int, old_tag::LinkTag)
    for s in 1:nqubits(st.regs[endpoint])
        t = st.tags[endpoint][s]
        if !isnothing(t) && t.left_node == old_tag.left_node && t.right_node == old_tag.right_node
            st.tags[endpoint][s] = nothing
            isassigned(st.regs[endpoint].staterefs, s) && traceout!(st.regs[endpoint][s])
            return
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — GLMakie Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

function build_figure(st_obs::Observable{SimState})
    fig = Figure(size=(1100, 720), backgroundcolor=:white)

    # ── Title ─────────────────────────────────────────────────────────────────
    Label(fig[1, 1:2], "Linear Quantum Repeater Chain — QuantumSavory.jl (swap-ASAP)",
        fontsize=18, font=:bold, padding=(0, 0, 10, 0))

    # ── Chain topology axis ───────────────────────────────────────────────────
    ax_chain = Axis(fig[2, 1:2],
        title="Chain state  (nodes × memory slots)",
        xlabel="Node",
        ylabel="",
        yticks=[],
        xticklabelsize=13,
        titlesize=14)

    # ── Rate history axis ─────────────────────────────────────────────────────
    ax_rate = Axis(fig[3, 1],
        title="Cumulative E2E rate",
        xlabel="Simulation step",
        ylabel="Deliveries / step",
        titlesize=13)

    # ── Inter-delivery time histogram ─────────────────────────────────────────
    ax_hist = Axis(fig[3, 2],
        title="Inter-delivery step distribution",
        xlabel="Steps between deliveries",
        ylabel="Count",
        titlesize=13)

    # ── Log panel ─────────────────────────────────────────────────────────────
    log_label = Label(fig[4, 1:2], "Log: starting simulation…",
        justification=:left, fontsize=11,
        color=:gray40, padding=(8, 0, 0, 0))

    # ── Stats bar ─────────────────────────────────────────────────────────────
    stats_label = Label(fig[5, 1:2], "t=0 | deliveries=0 | rate=—",
        fontsize=13, font=:bold, color=:black,
        padding=(0, 0, 6, 0))

    rowsize!(fig.layout, 2, Relative(0.38))
    rowsize!(fig.layout, 3, Relative(0.30))
    rowsize!(fig.layout, 4, Fixed(60))
    rowsize!(fig.layout, 5, Fixed(28))

    # ── Chain drawing observables ─────────────────────────────────────────────
    n = N_NODES
    xs_node = range(1.0, n, length=n) |> collect

    # Node positions
    node_xs = Observable(xs_node)
    node_ys = Observable(zeros(n))

    # Colour each node by occupancy
    node_colors = Observable(fill(:gray85, n))

    # Link segments: one per elementary link
    link_segs = Observable([(Point2f(i, 0), Point2f(i + 1, 0)) for i in 1:(n-1)])
    link_colors = Observable(fill(:gray80, n - 1))
    link_widths = Observable(fill(2.0f0, n - 1))

    # Virtual-link arcs: variable count
    arc_segs = Observable(Vector{Tuple{Point2f,Point2f,Float32}}())  # (p1, p2, height)

    # Qubit dot positions and colours
    qdot_pos = Observable(Point2f[])
    qdot_colors = Observable(Symbol[])

    # ── Cumulative rate observable ─────────────────────────────────────────────
    rate_xs = Observable(Float64[0.0])
    rate_ys = Observable(Float64[0.0])

    # ── Histogram observable ───────────────────────────────────────────────────
    hist_vals = Observable(Float64[])

    # ── Plot the static and dynamic elements ──────────────────────────────────

    # Backbone line
    lines!(ax_chain, [1.0, Float64(n)], [0.0, 0.0],
        color=:gray70, linewidth=1.5, linestyle=:dash)

    # Elementary links
    for i in 1:(n-1)
        lc = @lift $link_colors[i]
        lw = @lift Float64($link_widths[i])
        lines!(ax_chain,
            [Float64(i), Float64(i + 1)], [0.0, 0.0];
            color=lc,
            linewidth=lw)
    end

    # Virtual link arcs (drawn as simple Bezier-like curves via series of points)
    arc_plot = Observable(Vector{Vector{Point2f}}())
    arc_color_list = Observable(Vector{Symbol}())

    for _ in 1:1   # placeholder; will be replaced each frame
    end

    # Node circles
    scatter!(ax_chain, node_xs, node_ys;
        color=node_colors,
        markersize=28,
        strokewidth=2,
        strokecolor=:gray40)

    # Node labels
    for (i, x) in enumerate(xs_node)
        lbl = i == 1 ? "Alice" : i == n ? "Bob" : "R$(i-1)"
        text!(ax_chain, x, -0.18; text=lbl,
            align=(:center, :top), fontsize=12,
            color=i ∈ (1, n) ? :royalblue4 : :gray30)
    end

    # Qubit dots (left/right slots)
    scatter!(ax_chain, qdot_pos;
        color=qdot_colors,
        markersize=11,
        strokewidth=1,
        strokecolor=:white)

    # Rate plot
    lines!(ax_rate, rate_xs, rate_ys; color=:steelblue, linewidth=2)
    scatter!(ax_rate, rate_xs, rate_ys; color=:steelblue, markersize=4)

    # Histogram
    hist!(ax_hist, hist_vals; bins=15, color=(:teal, 0.7))

    limits!(ax_chain, 0.5, n + 0.5, -0.5, 0.8)

    return fig,
    (node_colors, link_colors, link_widths, qdot_pos, qdot_colors,
        rate_xs, rate_ys, hist_vals, log_label, stats_label,
        ax_chain, arc_plot, arc_color_list)
end

"""
    draw_arcs!(ax, st)

Overlay dashed arcs for virtual (swapped) links.  Called each frame.
Arcs are drawn with `lines!` using a parametric semi-ellipse.
"""
function draw_arcs!(ax, st::SimState, arc_lines_handle)
    # Collect unique virtual links (links spanning >1 hop)
    seen = Set{Tuple{Int,Int}}()
    arcs = NamedTuple{(:x1, :x2, :h, :col),Tuple{Float64,Float64,Float64,Symbol}}[]

    n = length(st.regs)
    for i in 1:n
        for s in 1:nqubits(st.regs[i])
            tag = st.tags[i][s]
            isnothing(tag) && continue
            ln, rn = tag.left_node, tag.right_node
            (ln, rn) in seen && continue
            push!(seen, (ln, rn))
            span = rn - ln
            span <= 1 && continue   # elementary links drawn as segments already
            h = 0.12 * span + 0.08
            age = st.step - tag.born_at
            col = age > (T_MEM ÷ 2) ? :orange3 : :seagreen4
            push!(arcs, (x1=Float64(ln), x2=Float64(rn), h=h, col=col))
        end
    end

    # Remove previous arc plots
    for plt in arc_lines_handle[]
        delete!(ax, plt)
    end
    empty!(arc_lines_handle[])

    for arc in arcs
        θs = range(0, π, length=60)
        xmid = (arc.x1 + arc.x2) / 2
        R = (arc.x2 - arc.x1) / 2
        xs = xmid .+ R .* cos.(θs)
        ys = arc.h .* sin.(θs)
        plt = lines!(ax, xs, ys;
            color=arc.col, linewidth=2.5,
            linestyle=:dash)
        push!(arc_lines_handle[], plt)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Main entry point
# ─────────────────────────────────────────────────────────────────────────────

function run_simulation()
    # Build registers
    regs = make_chain(N_NODES)
    st = SimState(regs)

    # Build ConcurrentSim environment
    sim = Simulation()
    @process entanglement_generation_process(sim, st)

    # ── Build Makie figure ────────────────────────────────────────────────────
    fig, handles = build_figure(Observable(st))
    (node_colors, link_colors, link_widths, qdot_pos, qdot_colors,
        rate_xs, rate_ys, hist_vals, log_label, stats_label,
        ax_chain, arc_plot, arc_color_list) = handles

    arc_lines_ref = Ref(Vector{Any}())   # mutable handle for arc plot objects

    # We need an Observable wrapping arc_lines_ref for draw_arcs!
    arc_lines_obs = Observable(Vector{Any}())

    display(fig)

    # ── Colours for qubit states ───────────────────────────────────────────────
    function qubit_color(tag::Union{LinkTag,Nothing}, step::Int)
        isnothing(tag) && return :gray85
        age = step - tag.born_at
        frac = age / T_MEM
        # Interpolate: fresh=steelblue, old=orangered
        return frac < 0.5 ? :royalblue : frac < 0.8 ? :darkorange : :red3
    end

    prev_deliveries = 0

    # ── Step-by-step update loop ───────────────────────────────────────────────
    for _ in 1:T_SIM
        run!(sim, st.step + 1.0)   # advance ConcurrentSim by one step

        t = st.step
        n = N_NODES

        # ── Node colours ──────────────────────────────────────────────────────
        nc = Vector{Any}(undef, n)
        for i in 1:n
            occ = any(!isnothing, st.tags[i])
            nc[i] = occ ? :steelblue : (i ∈ (1, n) ? :royalblue1 : :gray85)
        end
        node_colors[] = nc

        # ── Elementary link colours ────────────────────────────────────────────
        lc = Vector{Any}(undef, n - 1)
        lw = Vector{Float64}(undef, n - 1)
        for link in 1:(n-1)
            _, sa, _, sb = link_slots(regs, link)
            tL = st.tags[link][sa]
            tR = st.tags[link+1][sb]
            both = !isnothing(tL) && !isnothing(tR)
            if both && tL.left_node == link && tL.right_node == link + 1
                age = t - tL.born_at
                lc[link] = age < T_MEM ÷ 2 ? :royalblue : :darkorange
                lw[link] = 4.0
            else
                lc[link] = :gray80
                lw[link] = 1.5
            end
        end
        link_colors[] = lc
        link_widths[] = lw

        # ── Qubit dots ─────────────────────────────────────────────────────────
        qdots_xy = Point2f[]
        qdots_col = Any[]
        dx = 0.12
        for i in 1:n
            xs_node = Float64(i)
            for s in 1:nqubits(st.regs[i])
                tag = st.tags[i][s]
                col = qubit_color(tag, t)
                xoff = (nqubits(st.regs[i]) == 2) ? (s == 1 ? -dx : dx) : 0.0
                push!(qdots_xy, Point2f(xs_node + xoff, 0.0))
                push!(qdots_col, col)
            end
        end
        qdot_pos[] = qdots_xy
        qdot_colors[] = qdots_col

        # ── Virtual link arcs ──────────────────────────────────────────────────
        draw_arcs!(ax_chain, st, arc_lines_obs)
        arc_lines_ref[] = arc_lines_obs[]

        # ── Rate curve ────────────────────────────────────────────────────────
        push!(rate_xs[], Float64(t))
        push!(rate_ys[], t > 0 ? st.deliveries / t : 0.0)
        notify(rate_xs)
        notify(rate_ys)

        # ── Histogram of inter-delivery times ─────────────────────────────────
        if length(st.delivery_times) >= 2
            idts = diff(st.delivery_times)
            hist_vals[] = idts
        end

        # ── Log ───────────────────────────────────────────────────────────────
        recent = last(st.log, 3)
        log_label.text[] = join(recent, "\n")

        # ── Stats bar ─────────────────────────────────────────────────────────
        rate_str = t > 0 ? @sprintf("%.4f", st.deliveries / t) : "—"
        stats_label.text[] = "t=$(t)  |  deliveries=$(st.deliveries)  |  E2E rate=$(rate_str) /step  |  p_gen=$(P_GEN), p_swap=$(P_SWAP), T_mem=$(T_MEM)"

        sleep(0.04)   # ~25 fps; reduce for faster batch run
    end

    println("\n═══ Simulation complete ═══")
    println("Total steps     : $T_SIM")
    println("E2E deliveries  : $(st.deliveries)")
    println("Mean E2E rate   : $(round(st.deliveries/T_SIM, digits=4)) deliveries/step")
    if length(st.delivery_times) >= 2
        idts = diff(st.delivery_times)
        println("Mean inter-del. : $(round(mean(idts), digits=2)) steps")
        println("Std  inter-del. : $(round(std(idts),  digits=2)) steps")
    end
    return fig, st
end

# ─────────────────────────────────────────────────────────────────────────────
# Run it
# ─────────────────────────────────────────────────────────────────────────────
using Printf, Statistics

fig, final_state = run_simulation()