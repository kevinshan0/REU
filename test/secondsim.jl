##############################################################################
#  Linear Quantum Repeater Chain — ProtocolZoo version
#
#  This achieves the same swap-ASAP simulation as repeater_chain.jl, but
#  using QuantumSavory's prebuilt protocol building blocks instead of
#  handwritten gate-level code:
#
#    EntanglerProt       — drives Bell-pair generation on each elementary link
#    SwapperProt         — drives entanglement swapping (BSM) at each repeater
#    EntanglementTracker — propagates classical metadata after each swap so
#                          every node always knows its current virtual partner
#
#  These three protocols together implement swap-ASAP automatically.
#  The user only needs to:
#    1. Build a RegisterNet with the right topology
#    2. Instantiate one protocol object per role per node
#    3. Launch them as ConcurrentSim processes
#    4. Call run! and visualise
#
#  Visualisation uses GLMakie via QuantumSavory's built-in
#  registernetplot_axis recipe, which draws registers, qubit slots,
#  and live entanglement links.  Additional custom panels show the
#  cumulative E2E delivery rate and the inter-delivery time histogram.
#
#  Packages needed (install once):
#    using Pkg
#    Pkg.add(["QuantumSavory", "ConcurrentSim", "ResumableFunctions", "GLMakie"])
#
##############################################################################

using QuantumSavory
using QuantumSavory.ProtocolZoo          # EntanglerProt, SwapperProt, EntanglementTracker
using ConcurrentSim                      # discrete-event simulation engine
using ResumableFunctions                 # @resumable, @process macros
using GLMakie
using Printf, Statistics, Random

Random.seed!(42)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Parameters
# ═══════════════════════════════════════════════════════════════════════════════

const N_NODES = 5       # total nodes: Alice, repeaters, Bob
const P_GEN = 0.6     # probability of Bell-pair generation per attempt
const P_SWAP = 0.85    # probability of successful BSM at a repeater
const T_MEM = 6.0     # memory coherence time (simulation time units)
const T_SIM = 300.0   # total simulation time
const GEN_PERIOD = 1.0    # how often each EntanglerProt retries (time units)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Build the RegisterNet
#
# RegisterNet wraps a vector of Registers and provides the message-buffer
# infrastructure that EntanglementTracker uses to propagate classical updates.
#
# Slot layout (same as the low-level version):
#   Alice (node 1) : 1 slot  (right-link memory only)
#   Repeaters      : 2 slots (slot 1 = left, slot 2 = right)
#   Bob   (node N) : 1 slot  (left-link memory only)
#
# T1Decay(T_MEM) is the memory noise model: exponential amplitude decay with
# time constant T_MEM.  QuantumSavory applies it lazily — the density matrix
# is advanced only when a qubit is read or operated on.
# ═══════════════════════════════════════════════════════════════════════════════

function build_network(n::Int, t_mem::Float64)
    regs = map(1:n) do i
        # Alice abd Bob nodes only have 1 slot, each repeater node has a 
        # left and a right slot
        nslots = (i == 1 || i == n) ? 1 : 2
        # T1Decay channel describes energy loss of a qubit to its environment
        # in physical systems, the
        Register(nslots, T1Decay(t_mem))
    end
    return RegisterNet(regs)
end

net = build_network(N_NODES, T_MEM)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Instantiate prebuilt protocols
#
# EntanglerProt(sim, net, nodeA, slotA, nodeB, slotB; ...)
#   Repeatedly attempts to generate a Bell pair between net[nodeA][slotA] and
#   net[nodeB][slotB].  On success it initialises both qubits into |Φ⁺⟩ and
#   stamps them with EntanglementCounterpart tags.  On failure it waits
#   `retry_lock_time` before trying again.
#
# SwapperProt(sim, net, node; ...)
#   Monitors the two slots of an intermediate node.  When both are occupied
#   (both carry EntanglementCounterpart tags) it performs a BSM, succeeding
#   with probability `swapper_success_prob`.  On success it emits
#   EntanglementUpdate messages to both remote partners and clears its own
#   slots.  On failure it discards both qubits.
#
# EntanglementTracker(sim, net, node)
#   Listens for EntanglementUpdate messages addressed to `node` and rewrites
#   the local EntanglementCounterpart tag to reflect the new virtual partner.
#   Without this, end nodes would never learn they hold a long-range link.
# ═══════════════════════════════════════════════════════════════════════════════

sim = get_time_tracker(net)

# ── EntanglerProt: one per elementary link ────────────────────────────────────
# Slot convention matches build_network:
#   link i↔(i+1): right slot of node i, left slot of node i+1
#   end-nodes have only one slot so their "right/left" slot index is 1.
for i in 1:(N_NODES-1)
    slotA = (i == 1) ? 1 : 2    # right slot of node i
    slotB = 1                   # left slot of node i+1 (always slot 1)

    @process EntanglerProt(
        sim,
        net,
        i,
        i + 1;
        chooseslotA=slotA,
        chooseslotB=slotB,
        success_prob=P_GEN,
        attempts=-1,          # -1 = keep retrying indefinitely
        retry_lock_time=GEN_PERIOD
    )()
end

# ── SwapperProt: one per intermediate repeater node ────────────────────────────
for i in 2:(N_NODES-1)
    @process SwapperProt(
        sim,
        net,
        i;
        nodeL = <(i),
        nodeH = >(i),
        chooseL = argmin,
        chooseH = argmax,
        rounds = -1,      # -1 = run forever
    )()
end

# ── EntanglementTracker: one per node (needed at every node for tag updates) ──
for i in 1:N_NODES
    @process EntanglementTracker(sim, net, i)()
end

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Delivery detection
#
# QuantumSavory does not have a built-in "notify me when Alice and Bob share
# an E2E link" hook, so we write a lightweight @resumable watcher process.
# It polls the EntanglementCounterpart tag on Alice's slot and checks whether
# its recorded partner is Bob.  When detected it increments the counter and
# clears the link so the chain can start over.
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct DeliveryCounter
    count::Int
    times::Vector{Float64}
    log::Vector{String}
end
DeliveryCounter() = DeliveryCounter(0, Float64[], String[])

dc = DeliveryCounter()

@resumable function e2e_watcher(sim::Simulation, net::RegisterNet, dc::DeliveryCounter)
    alice_slot = net[1][1]
    bob_node = N_NODES

    while true
        @yield timeout(sim, 0.5)   # check twice per time unit

        # Query Alice's slot for an entanglement tag whose partner is Bob
        entry = query(net[1], EntanglementCounterpart, bob_node, ❓)
        if !isnothing(entry)
            dc.count += 1
            t = now(sim)
            push!(dc.times, t)
            msg = @sprintf("★ E2E delivery #%d at t=%.1f", dc.count, t)
            push!(dc.log, msg)
            length(dc.log) > 200 && popfirst!(dc.log)

            # Consume the link: traceout both endpoints so the chain restarts
            traceout!(alice_slot)
            querydelete!(net[1], EntanglementCounterpart, bob_node, ❓)
            bob_slot = net[bob_node][1]
            traceout!(bob_slot)
            querydelete!(net[bob_node], EntanglementCounterpart, 1, ❓)
        end
    end
end

@process e2e_watcher(sim, net, dc)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GLMakie visualisation
#
# QuantumSavory ships a Makie extension that defines registernetplot_axis,
# which renders the RegisterNet as a graph:
#   • Nodes = registers (circles)
#   • Edges = established entanglement links (coloured by qubit age / fidelity)
#   • Each node shows its qubit slots as smaller inner dots
#
# We add three custom panels below it:
#   • Cumulative E2E rate curve
#   • Inter-delivery time histogram
#   • Rolling log of recent events
# ═══════════════════════════════════════════════════════════════════════════════

fig = Figure(size=(1200, 780), backgroundcolor=:white)

# ── Title ─────────────────────────────────────────────────────────────────────
Label(fig[1, 1:3],
    "Linear Quantum Repeater Chain — QuantumSavory ProtocolZoo (swap-ASAP)",
    fontsize=18, font=:bold, padding=(0, 0, 10, 0))

# ── ProtocolZoo's built-in network graph ──────────────────────────────────────
# registernetplot_axis draws the RegisterNet live; it returns an Observable
# that we must notify to trigger redraws.
_, ax_net, plt_net, obs_net = registernetplot_axis(
    fig[2, 1:3], net;
    register_color_fn=(reg, slot) -> begin
        # Colour by qubit occupancy: blue=entangled, grey=empty
        e = query(reg, EntanglementCounterpart, ❓, ❓)
        isnothing(e) ? :gray80 : :royalblue
    end
)
ax_net.title = "Network state (entanglement links highlighted in blue)"

# ── Rate history panel ─────────────────────────────────────────────────────────
ax_rate = Axis(fig[3, 1],
    title="Cumulative E2E delivery rate",
    xlabel="Simulation time",
    ylabel="Deliveries / unit time",
    titlesize=13)

rate_ts = Observable(Float64[0.0])
rate_rs = Observable(Float64[0.0])
lines!(ax_rate, rate_ts, rate_rs; color=:steelblue, linewidth=2)

# ── Inter-delivery time histogram ──────────────────────────────────────────────
ax_hist = Axis(fig[3, 2],
    title="Inter-delivery time distribution",
    xlabel="Time between deliveries",
    ylabel="Count",
    titlesize=13)

hist_vals = Observable(Float64[])
hist!(ax_hist, hist_vals; bins=20, color=(:teal, 0.7))

# ── Stats and log ─────────────────────────────────────────────────────────────
ax_log = Axis(fig[3, 3],
    title="Event log",
    titlesize=13)
hidedecorations!(ax_log)
hidespines!(ax_log)

log_text = Observable("Simulation starting…")
text!(ax_log, 0.0, 1.0;
    text=log_text,
    align=(:left, :top),
    fontsize=10,
    font=:regular,
    color=:gray30,
    space=:relative)

stats_label = Label(fig[4, 1:3],
    "t=0.0 | deliveries=0 | rate=—",
    fontsize=13, font=:bold, padding=(0, 0, 6, 0))

rowsize!(fig.layout, 2, Relative(0.40))
rowsize!(fig.layout, 3, Relative(0.34))
rowsize!(fig.layout, 4, Fixed(28))

display(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Simulation loop with live updates
# ═══════════════════════════════════════════════════════════════════════════════

const STEP = 1.0    # advance ConcurrentSim by this much per visual frame
const SLEEP_DT = 0.03   # seconds between frames (~33 fps target)

t = 0.0
while t < T_SIM
    global t
    t += STEP
    run(sim, t)

    # ── Update built-in network plot ─────────────────────────────────────────
    notify(obs_net)

    # ── Rate curve ───────────────────────────────────────────────────────────
    push!(rate_ts[], t)
    push!(rate_rs[], t > 0 ? dc.count / t : 0.0)
    notify(rate_ts)
    notify(rate_rs)

    # ── Histogram ────────────────────────────────────────────────────────────
    if length(dc.times) >= 2
        hist_vals[] = diff(dc.times)
    end

    # ── Log panel ────────────────────────────────────────────────────────────
    recent = last(dc.log, 6)
    log_text[] = isempty(recent) ? "No deliveries yet…" : join(recent, "\n")

    # ── Stats bar ────────────────────────────────────────────────────────────
    rate_str = t > 0 ? @sprintf("%.4f", dc.count / t) : "—"
    stats_label.text[] = @sprintf(
        "t=%.1f  |  deliveries=%d  |  E2E rate=%s /unit time  |  p_gen=%.2f, p_swap=%.2f, T_mem=%.1f",
        t, dc.count, rate_str, P_GEN, P_SWAP, T_MEM)

    sleep(SLEEP_DT)
end

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Final summary
# ═══════════════════════════════════════════════════════════════════════════════

println("\n═══ Simulation complete ═══")
@printf("Total sim time    : %.1f\n", T_SIM)
@printf("E2E deliveries    : %d\n", dc.count)
@printf("Mean E2E rate     : %.4f deliveries/unit time\n", dc.count / T_SIM)
if length(dc.times) >= 2
    idts = diff(dc.times)
    @printf("Mean inter-del.   : %.2f time units\n", mean(idts))
    @printf("Std  inter-del.   : %.2f time units\n", std(idts))
end