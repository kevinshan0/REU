# The delayed-information setup

How `src/info_delay_setup.jl` and `src/info_delay_solver.py` model a repeater
chain in which **no node can see the whole chain**, and each node learns about
the rest of it only through classical messages that travel one hop per timestep.

## Why this setup exists

`environment.py` / `policy.py` solve the *global-knowledge* MDP: a single agent
sees every qubit's age instantly and picks a joint action, so the state is Markov
and policy iteration is exact. That assumption is physically wrong. Entanglement
swapping is a local Bell-state measurement whose outcome has to be *announced* —
the far end of a link cannot know its link was consumed, redirected, or destroyed
until a classical message arrives, and that message travels at the speed of
light, one hop per timestep here.

Dropping the assumption changes the problem's character completely:

- The chain is now a **Dec-POMDP**. Each node acts on a local observation that is
  not Markov, because what happens next also depends on parts of the chain it
  cannot see and on what the other nodes are simultaneously deciding.
- A node **cannot tell a live link from a dead one**. Its own slot is occupied
  and ageing; whether the qubit at the far end still exists is a *belief*, and
  the belief can be stale or simply wrong.
- That produces the pathology the whole investigation is about: **hanging
  qubits** — a stored, ageing, slot-occupying qubit whose partner is already
  gone, and whose owner has no local way of knowing.

## Model structure: ground truth vs. beliefs

Two data structures, and the split between them is the entire point.

**`Chain`** (`info_delay_setup.jl:132`, `info_delay_solver.py:104`) is ground
truth, which no node ever reads:

| field | meaning |
|---|---|
| `age[q]` | true age of the qubit in slot `q`; `-1` = empty |
| `partner[q]` | the flat index of the qubit `q` is really entangled with; `0` = **nothing** |
| `gen[q]` | occupancy stamp, bumped every time the slot is refilled |

`age[q] >= 0 && partner[q] == 0` **is** a hanging qubit. It still occupies the
slot, still ages, and still blocks entanglement generation on its elementary
link.

**`NodeView`** (`info_delay_setup.jl:208`, `info_delay_solver.py:133`) is what
one node believes. Same shape, but every far-end field is a belief carrying a
timestamp, and there is a second, separate ledger of what it has heard about the
rest of the chain.

### Qubit layout

Flat, 1-based, `1:2n-2`. Node 1 owns `{1}`, node `j` owns `{2j-2, 2j-1}`
(left-facing, right-facing), node `n` owns `{2n-2}`. Elementary link `k` joins
qubits `(2k-1, 2k)`.

### The timestep

Six phases, in this exact order (`run_trial`, `info_delay_setup.jl:665`;
`run_episode`, `info_delay_solver.py:449`):

1. every stored qubit ages by one
2. **LLEG** on every elementary link with *both* slots free
3. classical information from the neighbours lands
4. every node picks an action from its own delayed view; **swaps fire, then
   voluntary discards**
5. the retention-time cutoff sweeps anything at `age == t_cut`
6. every node tells its neighbours what it now knows

Delivery is checked after phase 2 and after the swaps in phase 4 — before the
cutoff sweep can tear down a link that was just completed.

### Two implementations, deliberately

`info_delay_setup.jl` is the simulator: same dynamics **plus** the physics
(QuantumSavory `Register`s with `Depolarization` backgrounds,
`DepolarizedBellPair`, `EntanglementSwap`, end-to-end fidelity via `observable`).
`info_delay_solver.py` is a port of the same classical dynamics with the physics
stripped out, because the solver optimizes delivery time only. They must agree;
`python info_delay_solver.py --verify` runs swap-asap through both and does a
two-sample z-test on the difference of mean delivery times.

---

# Concrete answers

## 1. What does each node send each timestep, and how far does it travel?

**Two channels, plus one that is not delayed at all.**

### Channel A — the flooded age ledger (broadcast, whole chain)

Every node, every timestep, in phase 6 (`transmit!`,
`info_delay_setup.jl:404`):

1. writes the **true ages of its own slots** into its ledger, stamped with the
   current time: `chain_age[q] = age[q]`, `chain_asof[q] = t`;
2. hands its **entire ledger table** — all `2n-2` `(age, asof)` entries, not just
   its own — to both neighbours.

Snapshots of every node's table are taken *before* any merging, so nothing
travels more than one hop per timestep no matter how the loop is ordered.

This channel reaches the whole chain, degrading by exactly one timestep of
freshness per hop: **node `i`'s entry about a qubit at node `j` is `j`'s truth as
of `t - |i-j|`.**

### Channel B — targeted announcements (point-to-point, routed)

A `Msg` (`info_delay_setup.jl:185`) is emitted only when something *happens* to a
link, and carries:

| field | meaning |
|---|---|
| `target`, `target_gen` | the specific **occupancy** of the slot being told, not just the slot |
| `kind` | `:update` (a swap succeeded — your link now ends elsewhere) or `:delete` (your link is gone) |
| `new_remote`, `new_remote_gen` | for `:update`, the new far end |
| `author`, `t` | who sent it and when |

Emitted by:

- **a swap** (`apply_swap!`, `info_delay_setup.jl:307`) — one message to whoever
  the node *believed* was on each far end. `:update` on success, `:delete` on
  failure.
- **a discard** (`apply_discard!`, `:359`), whether voluntary or forced by the
  cutoff sweep — a `:delete` to the believed far end.

These are **routed**, one hop per timestep, toward `owner(target)`. So a message
travels exactly `|author - owner(target)|` hops and then stops — unlike the
ledger, it does not flood the chain.

Addressing an *occupancy* rather than a slot is what makes late messages safe: if
the slot has since been emptied and refilled, `gen` no longer matches and the
message is dropped instead of corrupting a fresh link. (This is ProtocolZoo's
`EntanglementID` idea in miniature.)

### The one undelayed channel — LLEG heralding

Entanglement generation success is **heralded**, so both endpoints of a new
elementary link learn about it in the *same* timestep it is created
(`attempt_generation!`, `info_delay_setup.jl:268`). This is the only knowledge in
the model that is not delayed, and it is physically justified: the heralding
signal is what tells both sides the attempt worked.

## 2. How is received information incorporated into a node's state?

Two merges, then a compression.

### Merging the ledger — newest stamp wins

In phase 6, for every qubit `q` in the chain, a node overwrites its entry from a
neighbour's only if the neighbour's is fresher:

```julia
if theirasof[q] > v.chain_asof[q]
    v.chain_age[q] = theirage[q]; v.chain_asof[q] = theirasof[q]
end
```

The ledger stores the age *as of* a past time, so it must be extrapolated
forward to be useful:

```
projected_age(v, q, t) = chain_age[q] + (t - chain_asof[q])
```

This is a **projection, not a fact** — the qubit may have been discarded or
swapped away in the interim. That gap is exactly the uncertainty a policy has to
price in.

### Applying a message — three guards, then last-writer-wins

`receive!` (`info_delay_setup.jl:237`, `info_delay_solver.py:259`) runs in
phase 3, in this order:

1. **History re-addressing.** If this node already swapped or discarded the
   addressed occupancy away, `history[(target, gen)]` says where that
   entanglement actually went. The message is re-addressed and keeps travelling —
   or dropped if it went nowhere. (ProtocolZoo's `EntanglementHistory`.)
2. **Not mine → forward.** `owner(target) != v.node` means keep it moving.
3. **Occupancy check.** If the slot is empty, or `gen[target] != target_gen`, the
   link this message is about is long dead: drop it silently.
4. **Apply**, but only if `msg.t > partner_asof[target]` — ordering is by the
   message's *send* time, not its arrival, so an older message that took a
   detour cannot clobber a newer belief. `:update` sets
   `partner`/`partner_gen` to the new far end; `:delete` zeroes them. Either way
   `partner_asof[target] = msg.t`.

### Compressing to an observation

Handwritten policies receive the rich observation from `observe`
(`info_delay_setup.jl:464`): per own qubit `(q, age, far_q, far_node, stale,
far_age)`, plus the whole `chain_age`/`chain_asof` ledger.

Tabular (solved) policies are keyed on a deliberately coarse encoding —
`encode_observation` (`info_delay_setup.jl:565`, `info_delay_solver.py:326`) —
**four integers per own qubit, left-facing first**:

| feature | meaning | trustworthiness |
|---|---|---|
| `age` | own slot's exact age, clipped at `t_cut`; `-1` = empty | **exact** — a node can always see its own slots |
| `dist` | hops to the believed far end; `0` = believed unentangled, floored at 1 otherwise | a belief; can be stale or wrong |
| `stale` | timesteps since that belief was last confirmed, saturating at `S_MAX` | the node's only handle on *how much* it is risking |
| `status` | what the ledger projects for the far end: `0` never heard, `1` alive, `2` past its cutoff | a projection |

`S_MAX = 2` and must match between the two files — the Julia loader hard-errors
if the JSON declares a different one, since mismatched keys would silently
degrade a solved policy into its fallback.

## 3. Are the handwritten policies written, and do they use the arriving information?

**Yes — two of them, and one of them is the interesting case.**

Both live in `info_delay_setup.jl` and are mirrored exactly in the Python solver
so they can serve as JESP seeds:

| policy | Julia | Python |
|---|---|---|
| `SwapASAP` | `:481` | `swapasap_action`, `:363` |
| `CutoffAwareSwap(margin)` | `:514` | `cutoffaware_action`, `:377` |

`FunctionPolicy` (`:550`) wraps any `obs -> NodeAction` callable, which is the
hook for anything new you write.

### What `CutoffAwareSwap` does with incoming information

It is swap-asap plus the two inferences the delayed information actually
supports:

1. **`far_q == 0` → discard now.** The belief was zeroed by an arrived `:delete`
   message, so the node *knows* this slot is hanging. Free it so LLEG can restart
   rather than feeding a doomed swap. (Channel B.)
2. **`far_age > t_cut` → discard now.** The flooded ledger projects the far end
   past its own retention limit, so this slot is almost certainly hanging.
   (Channel A.)

Plus an optional `margin`: refuse to swap links within `margin` of being cut
anyway. **It defaults to 0 (off) because measurement says it is harmful** — a
link at age exactly `t_cut` is on its final timestep, so swapping it now is the
only value it has left. The docstring records the cost at n=5, p_s=0.9:
`p=0.4, t_cut=2` goes 13.085 → 21.193 at `margin=1`.

### Measured: rule 2 never fires, and rule 1 only pays off when `t_cut` is large

Instrumenting both policies over 2,000–3,000 episodes (n=5, p_s=0.9):

| setting | decisions | differ from swap-asap | triggered by rule 1 (`dist==0`) | by rule 2 (`status==2`) |
|---|---|---|---|---|
| p=0.9, t_cut=6 | 44,580 | 850 (1.91%) | 850 | **0** |
| p=0.5, t_cut=3 | 71,950 | 815 (1.13%) | 815 | **0** |
| p=0.4, t_cut=2 | 125,670 | 468 (0.37%) | 468 | **0** |

**Rule 2 never fires at all.** The reason is structural: a qubit is swept the
instant its age reaches `t_cut`, and the resulting `:delete` message and the
ledger update both travel at one hop per timestep. The targeted message therefore
never arrives *later* than the inference the ledger would support — the ledger
can never scoop it. Everything `CutoffAwareSwap` actually achieves, it achieves
from channel B.

And the discard is only worth anything when it beats the local cutoff sweep:

| setting | E[T] swap-asap | E[T] cutoff-aware | discards issued | fired *before* the sweep would have |
|---|---|---|---|---|
| p=0.9, t_cut=6 | 4.657 | **4.519** (−3.0%) | 351 | 325 (93%) |
| p=0.5, t_cut=3 | 7.563 | **7.435** (−1.7%) | 935 | 420 (45%) |
| p=0.4, t_cut=2 | 13.371 | **13.371** (0.0%) | 773 | **0 (0%)** |

At `t_cut = 2` every single discard fires at age exactly 2 — the same timestep
phase 5 would have removed the qubit anyway. The trajectories are bit-identical
to swap-asap's. **The information arrives, is correctly incorporated, and is
still worthless, because a message needs at least one hop and the qubit only
lives for two.** Useful information requires `t_cut` large enough that the news
outruns the local sweep.

This is the natural place to aim a better handwritten policy: the `stale` feature
is in the observation and neither handwritten policy reads it.

## 4. How is swap-asap implemented here?

```julia
function decide(::SwapASAP, obs)
    length(obs.own) == 2 || return NOOP          # end nodes never swap
    ready = all(o -> o.age >= 0 && o.far_q != 0, obs.own)
    return ready ? NodeAction(true, Int[]) : NOOP
end
```

Three things matter:

- **`far_q != 0` is a belief, not truth.** This is the whole point of the
  baseline. Swap-asap cannot distinguish a live link from a hanging qubit, so it
  happily burns a good link against a dead one and pushes the damage further down
  the chain.
- **It never discards voluntarily** — `NodeAction(true, Int[])`. Slots are only
  ever freed by a swap or by the phase-5 cutoff sweep.
- **End nodes always no-op.** They own one qubit and have nothing to swap.

The Python mirror (`swapasap_action`, `:363`) reads the same two conditions off
the encoded tuple: `obs[0] >= 0 and obs[1] != 0` for the left qubit,
`obs[4] >= 0 and obs[5] != 0` for the right.

Swap-asap also **doubles as the default fallback** for `TabularPolicy`: solved
tables store only the observations where they *disagree* with the seed policy, so
an empty table reproduces the baseline exactly. The seed's name travels in the
JSON's `fallback` field, and getting it wrong would be silent — a
cutoff-aware-seeded table applied with a swap-asap fallback behaves as neither —
so the loader validates it.

## 5. Hanging qubit used in a swap: yes, it destroys the other input link

Correct, and it is deliberate. From `apply_swap!`
(`info_delay_setup.jl:307`, `info_delay_solver.py:185`):

```julia
usable = occupied(c, qL) && occupied(c, qR) && farL != 0 && farR != 0
success = usable && rand() < par.p_s
```

If either input is hanging (`far == 0`), `usable` is false, so **the BSM fails
deterministically — `p_s` is not even rolled.** The failure branch then runs:

```julia
for (own, far) in ((qL, farL), (qR, farR))
    occupied(c, own) || continue
    far != 0 && (c.partner[far] = 0)   # the good side's far end is now hanging
    free!(c, own)
end
```

So all three consequences land:

1. **Both local qubits are consumed.** The measurement is physical; it happens
   whether or not the links were real.
2. **The good side's far end becomes a new hanging qubit** — `partner[far] = 0`.
   A problem that was `j` hops from being resolvable is now `j + k` hops away.
   This is the compounding effect the setup exists to study.
3. **The new victim does not find out for `|i - j|` timesteps.** A `:delete` is
   addressed to whoever the swapping node *believed* was there; meanwhile that
   slot keeps ageing and keeps blocking LLEG on its elementary link, since
   generation requires *both* slots free.

Note the swapping node's belief was not necessarily wrong through carelessness —
the far end may have been cut by its own cutoff sweep a single timestep ago, with
the announcement still in flight.

### How often this actually happens

Instrumented over 3,000 swap-asap episodes:

| setting | swaps fired | with ≥1 hanging input | new victims created | occupied slot-timesteps that are hanging |
|---|---|---|---|---|
| n=5, p=0.9, p_s=0.9, t_cut=6 | 13,148 | **10.5%** | 1,371 | **22.8%** |
| n=5, p=0.5, p_s=0.9, t_cut=3 | 16,857 | 11.0% | 1,837 | 12.1% |
| n=5, p=0.4, p_s=0.9, t_cut=2 | 25,062 | 13.4% | 3,326 | 10.1% |

"New victims" counts swaps where exactly one side was hanging — the case that
converts a good link into another hanging qubit. Roughly a fifth of all occupied
slot-time at `t_cut = 6` is being spent on qubits that are already useless.

---

## Running it

```bash
# baselines only, in the physics simulator
julia --project=. src/info_delay_setup.jl

# check the Python dynamics against the Julia simulator (two-sample z on E[T])
python src/info_delay_solver.py --verify --n 5 --p 0.9 --p_s 0.9 --t-cut 6

# solve a decentralized policy (JESP), then measure it with fidelity
python src/info_delay_solver.py --n 5 --p 0.9 --p_s 0.9 --t-cut 6 --simulate
```

Julia side, with a handwritten policy of your own:

```julia
include("src/info_delay_setup.jl")
par = Params(; n=5, p=0.9, p_s=0.9, τ=1000.0, t_cut=6)
run_trials(SwapASAP();            trials=2000, par, seed=1234)
run_trials(CutoffAwareSwap(0);    trials=2000, par, seed=1234)
run_trials(FunctionPolicy(my_policy); trials=2000, par, seed=1234)
```

`Params` can derive `t_cut` from a fidelity budget (`retention_time`, `:84`)
instead of taking it directly. Note `τ` moves fidelity only — the classical layer
never reads it — so it can be chosen freely without disturbing delivery times.

## The solver, in one paragraph

Optimal Dec-POMDP solutions are NEXP-complete, so `info_delay_solver.py` uses
**JESP** (Nair et al. 2003): hold every other node's policy fixed, compute one
node's best response, move on, repeat. That converges to a Nash equilibrium over
joint policies, not a global optimum — the seed matters, which is why
`--fallback`/`FALLBACKS` offers both handwritten policies as starting points. A
node's local observation is not Markov, so there is no transition function to
write down; each best-response step instead estimates `P(o'|o,a)` from
epsilon-greedy Monte-Carlo rollouts and then solves *that* empirical MDP exactly
with the same sparse `(I - P_pi) V = -1` machinery described in
[SPARSE-POLICY-EVAL.md](SPARSE-POLICY-EVAL.md). Solved tables land in
`src/data_infodelay/` as JSON, next to a `_history.json` recording each sweep.

Two guards worth knowing about, since both exist to stop the solver fooling
itself: `--min-visits` refuses to let the improvement step pick an action with
too few samples, and the reported final number is a **re-evaluation on a fresh
batch**, because taking the argmin over noisy sweeps biases the winner
optimistically.
