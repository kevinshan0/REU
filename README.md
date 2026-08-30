# Optimal policies for quantum repeater chains

REU work on **when a repeater node should swap, wait, or discard** in a linear
entanglement-distribution chain — first assuming a central controller that sees
the whole chain, then dropping that assumption and giving each node only what
classical messages can carry to it.

Two things account for most of the work here, and they are the two the rest of
this README is about:

1. **The global-knowledge, qubit-age-cutoff MDP (no discard).** An exact
   discrete MDP over the full chain state, solved by policy iteration, giving
   the optimal expected delivery time and the policy that achieves it — plus a
   swap-asap baseline to measure it against and a QuantumSavory simulator that
   replays the solved policy against real depolarizing registers to get
   end-to-end fidelity.
2. **The delayed-information (Dec-POMDP) setup.** The same chain with the global
   view removed: every node sees only its own slots, everything else arrives one
   hop per timestep, and swapping against a link that has already died is now
   possible — and common.

Everything else in the repo (discard-enabled variants, BRTDP, policy
distillation, belief-space POMDP, the `rl/` and Julia tutorial folders) is
exploratory or side investigation, and is summarized briefly at the end.

---

## Repository map

```
linear-repeater-chain/     ← the actual research code
  src/
    environment.py           qubit-age cutoff MDP: states, actions, transitions
    policy.py                policy iteration, swap-asap baseline, BRTDP, I/O
    simulate.py              hands a solved policy to Julia for fidelity measurement
    simulate.jl              QuantumSavory replay of that policy on noisy registers
    info_delay_setup.jl      delayed-information chain + physics (the simulator)
    info_delay_solver.py     same dynamics without physics + JESP Dec-POMDP solver
    distill.py               side: global policy → per-node local tables
    local_policy.py          side: directly optimized memoryless local policy
    po_environment.py        side: belief-state primitives
    po_policy.py             side: belief-space decentralized POMDP policy iteration
    *.ipynb                  parameter-sweep heatmaps for each of the above
    data_*/                  solver output caches (gitignored)
  visualizations/          figures produced by the notebooks/scripts
  CUTOFF-INFO.md           why qubit-age cutoffs instead of the paper's link-age cutoffs
  SPARSE-POLICY-EVAL.md    the sparse direct policy-evaluation solve, and its speedups
  INFO-DELAY.md            full description of the delayed-information model
  INFO-DELAY-TRACE.md      timestep-by-timestep transcript of two real episodes

rl/                        summer-school RL exercises (bandits, coin toss, blackjack)
discrete-event-simulation/ ConcurrentSim/ResumableFunctions exercises
qsexamples/                QuantumSavory example scripts, run and annotated
tutorials/                 Makie/Graphs/QuantumSavory tutorial scratch work
test/                      first QuantumSavory simulations
```

## Setup

Python (uv, Python ≥ 3.12) and Julia both live in `linear-repeater-chain/`:

```bash
cd linear-repeater-chain
uv sync                                   # numpy, scipy, matplotlib, ipykernel
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # QuantumSavory, JSON3, ...
```

The Python solvers are pure numpy/scipy and need no Julia. Julia is needed only
for the fidelity-aware simulators (`simulate.jl`, `info_delay_setup.jl`).

---

# Part 1 — The global-knowledge qubit-cutoff model

## The physical setting

A chain of `n` nodes. The two end nodes hold one qubit each; every middle node
holds two (left-facing and right-facing), so there are `2n-2` qubits, flat-indexed
`0 … 2n-3`. Elementary link `k` joins qubits `(2k, 2k+1)` — the two qubits facing
each other across an edge.

Each time slot:

- every *elementary* link with both slots free attempts generation, succeeding
  with probability `p`;
- the controller picks a set of middle nodes to swap; each swap succeeds with
  probability `p_s`, fusing its two input links into one longer link and freeing
  the middle node's qubits;
- any qubit whose age reaches the cutoff `t_cut` is discarded, taking its
  partner with it.

The episode ends when the two end nodes share a link. The objective is to
**minimize the expected number of time slots to that end-to-end link**.

Four parameters: `n`, `p`, `p_s`, `t_cut`.

## What is different from the paper

The model follows Iñesta et al., *"On the Effect of Quantum Memory Cutoffs"*
([arXiv:2207.06533](https://arxiv.org/pdf/2207.06533)) and its reference
implementation ([AlvaroGI/optimal-homogeneous-chain](https://github.com/AlvaroGI/optimal-homogeneous-chain)),
with one deliberate substitution, documented in [CUTOFF-INFO.md](linear-repeater-chain/CUTOFF-INFO.md):

The paper uses a **link-age** cutoff — a swapped link inherits the age of the
oldest link that produced it, and the whole link is cut when that age hits
`t_cut`. That is only meaningful if someone knows the whole link's age, which
requires the global knowledge the paper assumes. This repo uses a **qubit-age**
(retention-time) cutoff instead: each qubit is discarded when *its own* age
reaches `t_cut`, which is something a node can actually check locally. The two
conventions are solved under identical topology and probability dynamics, so
they can be compared directly.

Consequence worth knowing: under qubit ages, the two qubits of one virtual link
can carry *different* ages, so the exported state matrix is not symmetric the way
the paper's is. Everything the paper's analysis code checks structurally (link
existence, end-to-end connectivity, valid swap nodes) only looks at which cells
are finite, so its plotting helpers still load this data unchanged.

## The MDP

**State** — the age of every qubit (`-1` = empty), plus a `hanging` flag used only
by the discard variants. Links are *reconstructed* from the occupancy pattern
rather than stored: qubits pair up non-crossing, even-is-left / odd-is-right
(`State.links_of`, `environment.py:100`).

**Action** — any subset of the middle nodes that currently hold two live qubits,
meaning "attempt a swap here this slot". The empty set is "wait"; the full set is
swap-asap. Actions are enumerated by ascending combination size, so **swap-asap
is always the last entry of the action space** — which is how the baseline
evaluator picks it out without a search.

**Transition** (`step`, `policy.py:179`) composes one full time slot in this
order: apply the action → apply cutoffs → age every qubit → attempt generation.
If the action already produced an end-to-end link, evolution stops there.

Two exponentials sit inside a transition — which swaps succeed, and which
generation attempts succeed. They are independent of each other, so the step is
split at exactly that point (`_cached_swap_resolution` / `_cached_post_swap_step`)
and every `(state, action)` pair passing through the same intermediate
configuration shares the work. On top of that, `_cached_step` memoizes whole
transitions: the transition *structure* never changes across sweeps, only the
values flowing through it.

**Reward** — `-1` per time slot, undiscounted (`γ = 1`), terminal states
absorbing at `V = 0`. So `V(s) = -E[time slots from s]`, and the headline number
is the expected delivery time from the empty chain:

```python
expected_delivery_time = -(state_info[0]["value"] + 1)
```

The `+1` matches the reference implementation's 0-indexed convention, where
delivering on the very first attempt counts as time slot 0.

**State space is discovered lazily.** `Agent.observe` registers a state the first
time a walk reaches it, so nothing is ever enumerated that the dynamics cannot
actually produce.

## Solving it

```bash
cd linear-repeater-chain

# optimal policy via policy iteration
python src/policy.py --n 5 --p 0.9 --p_s 0.5 --cutoff 2

# same thing with the exact sparse solve for the evaluation step
python src/policy.py --n 5 --p 0.9 --p_s 0.5 --cutoff 2 --method direct

# swap-asap baseline, for comparison
python src/policy.py --n 5 --p 0.9 --p_s 0.5 --cutoff 2 --policy swap-asap
```

Results are pickled under `src/data_policyiter/` and `src/data_swapasap/`, in the
same on-disk shape as the paper's own data files, and are reused automatically on
a re-run with identical parameters.

### `--method direct`: the evaluation step as a linear solve

Policy iteration alternates evaluation and improvement; improvement is one pass,
so evaluation is the whole cost. The default `iterative` evaluation does what the
paper does — repeat Bellman backups until they settle — and each sweep propagates
information exactly **one time step of the horizon**, so the sweep count scales
with how slowly the chain mixes. Low `p` and low `p_s` are precisely the physically
interesting regime *and* the regime that hurts most: at `p=0.25, p_s=0.35`,
expected delivery is ~74 slots and evaluation needs thousands of sweeps.

But for a *fixed* policy the Bellman equation is not a fixed point to iterate
toward — it is a linear system:

```
V = -1 + P_π V     ⟺     (I − P_π) V = −1
```

`_evaluate_policy_direct` assembles that system sparsely (COO for assembly, CSC
for the solve, one `spsolve`) and solves it outright. Cost then depends on the
sparsity of the transition graph, not on the mixing time — the horizon factor
disappears. Measured speedups run from 1.6× on easy cases to **224×** on
`n=4, p=0.25, p_s=0.35` with discard enabled, and the direct answer is also the
*more* accurate one, since iterative evaluation stops one tolerance short of
`V^π` by construction. The full derivation, sparsity measurements, and timing
table are in [SPARSE-POLICY-EVAL.md](linear-repeater-chain/SPARSE-POLICY-EVAL.md).

One subtlety this creates: exact evaluation makes Q-values at some states
numerically tied, so improvement can flip between equally-good actions forever
and never register as "stable". The outer loop therefore stops on **either** an
exactly stable policy **or** a value change under tolerance.

The same `(I − P)` assembly is reused everywhere a fixed policy needs an exact
score — BRTDP's envelope, the relaxed lower bound, the local/POMDP/info-delay
solvers, and (transposed, `(I − P)ᵀx = e₀`) expected visitation counts in
`distill.py`. Value functions and visitation counts are the same matrix read in
opposite directions.

## Measuring fidelity: `simulate.py` → `simulate.jl`

The MDP has no notion of fidelity — swaps are binary success/failure and the only
objective is time. To find out what the optimal policy does to the *quantum state*,
`simulate.py` regenerates the raw qubit-index action list for every state (a pure
function of the state and configuration, so the ordering matches whatever `policy`
was recorded against), dumps the state→action table as JSON, and hands it to
`simulate.jl`.

`simulate.jl` drives real QuantumSavory `Register`s with `Depolarization(τ)`
backgrounds tick by tick under *exactly* that table — deliberately not
ProtocolZoo's `SwapperProt`/`CutoffProt`, which are generic age-limit heuristics
and cannot follow an arbitrary lookup table. It reports empirical delivery times
and average end-to-end Bell-pair fidelity.

```bash
python src/simulate.py --n 4 --p 0.8 --p_s 0.8 --cutoff 5 --episodes 1000 --tau 10
```

## Figures

`visualizations/` holds the output of the notebooks and scripts: optimal vs.
swap-asap delivery time across `p` for several chain lengths, delivery time vs.
cutoff, sample trajectories, and heatmaps of the improvement over swap-asap.

---

# Part 2 — The delayed-information setup

Full write-up: **[INFO-DELAY.md](linear-repeater-chain/INFO-DELAY.md)**.
Worked transcript of two real episodes: **[INFO-DELAY-TRACE.md](linear-repeater-chain/INFO-DELAY-TRACE.md)**.

## Why

Part 1 assumes a controller that sees every qubit's age instantly and issues a
joint action. That assumption is physically wrong. Entanglement swapping is a
local Bell-state measurement whose outcome has to be *announced*, and the
announcement travels at the speed of light — one hop per timestep here. The far
end of a link cannot know its link was consumed, redirected, or destroyed until
that message arrives.

Dropping the assumption changes the problem's character completely:

- it becomes a **Dec-POMDP** — each node acts on a local observation that is not
  Markov, because what happens next depends on parts of the chain it cannot see
  *and* on what other nodes are simultaneously deciding;
- a node **cannot distinguish a live link from a dead one**: its slot is occupied
  and ageing, but whether the qubit at the far end still exists is a *belief*,
  and the belief can be stale or simply wrong;
- which produces the pathology the whole investigation is about — **hanging
  qubits**: stored, ageing, slot-blocking qubits whose partner is already gone,
  and whose owner has no local way to find out.

## Ground truth vs. beliefs

Two data structures, and the split between them is the entire point.

**`Chain`** is ground truth, which no node ever reads: each qubit's true `age`
(`-1` empty), its true `partner` (`0` = *nothing*), and a `gen` occupancy stamp
bumped each time the slot is refilled. `age[q] >= 0 && partner[q] == 0` **is** a
hanging qubit — it still occupies the slot, still ages, and still blocks
generation on its elementary link.

**`NodeView`** is what one node believes: the same shape, but every far-end field
is a belief carrying a timestamp, plus a separate ledger of what it has heard
about the rest of the chain.

## The timestep

Six phases, in this order:

1. every stored qubit ages by one;
2. **LLEG** on every elementary link with both slots free;
3. classical information from the neighbours lands;
4. every node picks an action from its own delayed view — swaps fire, then
   voluntary discards;
5. the retention-time cutoff sweeps anything at `age == t_cut`;
6. every node tells its neighbours what it now knows.

Delivery is checked after phase 2 and after the swaps in phase 4, before the
cutoff sweep can tear down a link that was just completed.

## What actually travels

**Channel A — the flooded age ledger (broadcast).** Every node, every timestep,
writes the true ages of its own slots into its ledger stamped with the current
time, then hands its *entire* table to both neighbours; on receipt, newest stamp
wins. This reaches the whole chain, degrading by exactly one timestep of freshness
per hop, so node *i*'s entry about a qubit at node *j* is *j*'s truth as of
`t − |i−j|`. Using it requires extrapolating forward — a **projection, not a
fact**, since the qubit may have been discarded or swapped away since.

**Channel B — targeted announcements (routed, point-to-point).** A message is
emitted only when something *happens* to a link: a swap sends `:update` (your link
now ends elsewhere) on success or `:delete` on failure; a discard — voluntary or
forced by the cutoff sweep — sends `:delete`. It is addressed to a specific
**occupancy** of a slot, not just the slot, and travels `|author − owner(target)|`
hops. Addressing occupancies is what makes late messages safe: if the slot has
since been emptied and refilled, `gen` no longer matches and the message is
dropped instead of corrupting a fresh link. Applying one runs three guards —
re-address through this node's own swap/discard history, forward if it is not
mine, drop if the occupancy is dead — and then applies only if the message's
*send* time beats the current belief's timestamp, so a message that took a detour
cannot clobber a newer belief. (These are ProtocolZoo's `EntanglementID` and
`EntanglementHistory` ideas in miniature.)

**The one undelayed channel — LLEG heralding.** Entanglement generation success
is heralded, so both endpoints of a new elementary link learn about it in the same
timestep it is created. This is the only knowledge in the model that is not
delayed, and it is what physically justifies the whole rest of the model being
delayed.

Solved (tabular) policies are keyed on a deliberately coarse encoding — four
integers per own qubit: its own exact `age` (a node can always see its own slots),
the `dist` in hops to the believed far end (a belief), `stale` = timesteps since
that belief was last confirmed (the node's only handle on *how much* it is
risking), and `status`, what the ledger projects for the far end.

## Two implementations, deliberately

`info_delay_setup.jl` is the simulator: the dynamics above **plus** the physics —
QuantumSavory registers with depolarization, `DepolarizedBellPair`,
`EntanglementSwap`, end-to-end fidelity via `observable`. `info_delay_solver.py`
is a port of the same classical dynamics with the physics stripped out, because
the solver optimizes delivery time only. They must agree, and `--verify` checks
it: swap-asap through both, two-sample z-test on the difference of mean delivery
times.

## The solver

Optimal Dec-POMDP solutions are NEXP-complete, so `info_delay_solver.py` uses
**JESP** (Nair et al. 2003): hold every other node's policy fixed, compute one
node's best response, move on, repeat. That converges to a Nash equilibrium over
joint policies, not a global optimum, so the seed matters — both handwritten
policies are available as starting points. Since a local observation is not
Markov, there is no transition function to write down; each best-response step
estimates `P(o'|o,a)` from ε-greedy Monte-Carlo rollouts and then solves *that*
empirical MDP exactly, with the same sparse `(I − P_π)V = −1` machinery from
Part 1. Two guards exist to stop the solver fooling itself: `--min-visits`
refuses to let improvement pick an action with too few samples, and the reported
final number is a **re-evaluation on a fresh batch**, because taking the argmin
over noisy sweeps biases the winner optimistically.

Solved tables land in `src/data_infodelay/` as JSON, alongside a `_history.json`
per sweep. They store only the observations where they *disagree* with their seed
policy, so an empty table reproduces the baseline exactly — which is why the seed
name travels in the JSON and is validated on load.

## What was learned

**Swapping against a hanging qubit destroys the other input link too.** If either
input is hanging, the BSM fails deterministically (`p_s` is not even rolled), both
local qubits are consumed, and the *good* side's far end becomes a new hanging
qubit — a problem that was `j` hops from resolvable is now `j + k` hops away, and
the new victim will not find out for `|i − j|` timesteps while its slot keeps
ageing and keeps blocking generation. Over 3,000 swap-asap episodes at `n=5`,
**10–13% of all swaps had at least one hanging input**, and at `p=0.9, t_cut=6`
roughly **23% of all occupied slot-time** was being spent on qubits that were
already useless.

**A cutoff-aware policy helps, but only when `t_cut` is large enough for the news
to outrun the local sweep.** `CutoffAwareSwap` adds the two inferences the delayed
information actually supports: discard immediately when an arrived `:delete` has
zeroed the belief, and discard when the ledger projects the far end past its own
retention limit. Measured:

| setting | E[T] swap-asap | E[T] cutoff-aware | discards that beat the sweep |
|---|---|---|---|
| p=0.9, t_cut=6 | 4.657 | **4.519** (−3.0%) | 325 of 351 (93%) |
| p=0.5, t_cut=3 | 7.563 | **7.435** (−1.7%) | 420 of 935 (45%) |
| p=0.4, t_cut=2 | 13.371 | 13.371 (0.0%) | **0 of 773 (0%)** |

At `t_cut = 2` every discard fires at age exactly 2 — the same timestep the cutoff
sweep would have removed the qubit anyway, so the trajectories are bit-identical
to swap-asap's. **The information arrives, is correctly incorporated, and is still
worthless, because a message needs at least one hop and the qubit only lives for
two.**

**The ledger channel never fired at all.** Across every setting instrumented,
100% of the cutoff-aware rule's deviations from swap-asap came from channel B;
the ledger-projection rule triggered **zero** times. The reason is structural: a
qubit is swept the instant its age reaches `t_cut`, and the resulting `:delete`
and the ledger update both travel one hop per timestep, so the targeted message
can never arrive *later* than the inference the ledger would support.

**The `stale` feature is in the observation and neither handwritten policy reads
it** — that is the obvious place to aim a better one.

## Running it

```bash
cd linear-repeater-chain

# baselines in the physics simulator
julia --project=. src/info_delay_setup.jl

# check the Python dynamics against the Julia simulator (two-sample z on E[T])
python src/info_delay_solver.py --verify --n 5 --p 0.9 --p_s 0.9 --t-cut 6

# solve a decentralized policy with JESP, then measure it with fidelity
python src/info_delay_solver.py --n 5 --p 0.9 --p_s 0.9 --t-cut 6 --simulate
```

With a handwritten policy of your own:

```julia
include("src/info_delay_setup.jl")
par = Params(; n=5, p=0.9, p_s=0.9, τ=1000.0, t_cut=6)
run_trials(SwapASAP();                trials=2000, par, seed=1234)
run_trials(CutoffAwareSwap(0);        trials=2000, par, seed=1234)
run_trials(FunctionPolicy(my_policy); trials=2000, par, seed=1234)
```

`Params` can derive `t_cut` from a fidelity budget (`retention_time`) instead of
taking it directly: pick the retention time so that even the worst case — all
`n-1` elementary links sitting at the cutoff age when swapped together — still
clears `F_min`. Note `τ` moves fidelity only; the classical layer never reads it,
so it can be chosen freely without disturbing delivery times.

`src/info_delay_heatmap.ipynb` sweeps `(p, t_cut)` grids per `p_s`, comparing the
cutoff-aware policy against swap-asap.

---

# Side investigations

These took comparatively little of the time and are included for completeness.

### Voluntary discard (`--discard`)

An extension of the Part 1 model in which the controller may also *choose* to
discard occupied qubits each slot, rather than only losing them to the cutoff.
Actions become `(swap_nodes, discard_qubits)` pairs. Discarding one qubit leaves
its partner **hanging** — occupied and ageing but part of no link — which is where
the `hanging` flag in `environment.py` comes from. `--discard k` caps discards at
`k` per slot, keeping that dimension of the action space linear rather than
exponential in the number of occupied qubits.

```bash
python src/policy.py --n 4 --p 0.25 --p_s 0.35 --cutoff 3 --discard 1 --method direct
```

### BRTDP (`--policy brtdp`)

Bounded real-time dynamic programming, built for the discard-enabled setting where
the inflated action space makes full sweeps expensive. It maintains a lower bound
(from a relaxed, no-decoherence chain solved exactly once per `(n, p, p_s)` and
reused as a pattern database) and an upper bound (from swap-asap, reachable from
any state via discard), and runs trials that chase whichever successors carry the
most bound-gap uncertainty, stopping when the root's own gap is inside tolerance.

The state-space saving is consistent (~2–5× fewer states at `n=3..5` with discard)
but does not automatically buy the same factor of wall clock, since BRTDP backs up
each visited state many times where `--method direct` makes one pass and hands the
rest to a sparse solve. It is the memory saving, not the time, that is dependable;
for small chains `policy_iteration` is both exact and comparably quick. Its one
genuine advantage over policy iteration is that it returns an **error bar** rather
than a labeled solved-or-not.

### Policy distillation (`distill.py`)

Takes the *already-solved* global policy and projects it onto per-node tables keyed
only on that node's own qubits. Two methods: a **greedy** majority vote (one vote
per consistent global state), and a **soft** version that weights each state's vote
by its expected visitation count under the optimal policy and temperature-scales
the result. The greedy method has a systematic bias worth recording: swap-asap is
exactly optimal in the large majority of enumerated global states, so it dominates
the vote almost everywhere and drowns out the rare states where the global policy
is actually doing something non-trivial — the distilled policy ends up close to
swap-asap itself. Visitation weighting is the fix, since it reflects how often a
local state is actually *encountered* rather than how many combinatorially
distinct microstates happen to share it.

### Directly optimized local policy (`local_policy.py`)

Instead of distilling after the fact, optimize the memoryless per-node tables
directly by self-consistent coordinate-ascent policy iteration: each node's
decision is a function of only its own current qubits, and the same table entry is
shared by every global state giving that node the same local view — a
self-consistency constraint an unconstrained global policy does not have.

### Belief-space POMDP (`po_environment.py`, `po_policy.py`)

Incomplete, and the least developed of these. Each node maintains a Bayesian
belief over the full global state and policy iteration runs over the augmented
hyper-state `(true state, belief_0, …, belief_{n-1})`. Because planning is offline
with full visibility into every node's *actual* belief, the usual infinite regress
(node i reasoning about node j's belief about node k's…) is sidestepped. The
belief space still has to be rounded and pruned to stay finite, and every extra bit
of precision multiplies the hyper-state space. Left here as a starting point — the
delayed-information setup in Part 2 turned out to be the more productive way to
attack the same question.

### Learning and tooling folders

- `rl/` — RL exercises worked through at the start: n-armed bandit, coin toss
  (policy evaluation / improvement / iteration), blackjack (on-policy first-visit
  Monte Carlo control).
- `discrete-event-simulation/` — ConcurrentSim / ResumableFunctions exercises,
  including a repeater exercise, from the summer school's discrete-event day.
- `qsexamples/`, `tutorials/`, `test/` — QuantumSavory example scripts, Makie and
  Graphs plotting tutorials, and the first standalone simulations, kept as
  reference.

---

## References

- Iñesta, Vardoyan, Scavuzzo, Wehner, *On the Effect of Quantum Memory Cutoffs
  in Entanglement Distribution* — [arXiv:2207.06533](https://arxiv.org/pdf/2207.06533),
  reference implementation [AlvaroGI/optimal-homogeneous-chain](https://github.com/AlvaroGI/optimal-homogeneous-chain).
- Nair, Tambe, Yokoo, Pynadath, Marsella, *Taming Decentralized POMDPs: Towards
  Efficient Policy Computation for Multiagent Settings* (JESP), IJCAI 2003.
- McMahan, Likhachev, Gordon, *Bounded Real-Time Dynamic Programming*, ICML 2005.
- [QuantumSavory.jl](https://github.com/QuantumSavory/QuantumSavory.jl).
