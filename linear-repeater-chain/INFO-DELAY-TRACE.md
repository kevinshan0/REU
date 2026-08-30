# Worked example: ledgers, messages, and merging

A complete timestep-by-timestep trace of what every node records, what it sends,
and how arriving information is merged. Companion to
[INFO-DELAY.md](INFO-DELAY.md), which describes the mechanisms in the abstract.

**Every table on this page is a transcript of a real run**, not an illustration.
Both episodes are `n=4, p=0.6, p_s=0.5, t_cut=3`, swap-asap, under
`info_delay_solver.py`'s Python dynamics with `random.Random(seed)`. The script
that produced them is in the [appendix](#appendix-reproducing-this).

## Reading key

### Chain layout (n = 4)

```
        node 1        node 2         node 3        node 4
          [1]  ~~~  [2] [3]  ~~~  [4] [5]  ~~~  [6]
           └── link 1 ──┘ └── link 2 ──┘ └── link 3 ──┘
```

Node 1 owns qubit 1; node 2 owns {2, 3}; node 3 owns {4, 5}; node 4 owns qubit 6.
Elementary link `k` joins qubits `(2k-1, 2k)`. Arrays below are printed `q1..q6`.

### What each structure holds

| structure | who has it | contents |
|---|---|---|
| **truth** `age`, `partner`, `gen` | nobody — the simulator | real age (`-1` empty), real far end (`0` = **nothing**), occupancy stamp |
| **ledger** `chain_age`, `chain_asof` | every node, one entry per qubit **in the whole chain** | the age that qubit was last *reported* to have, and when that report was stamped |
| **belief** `partner`, `partner_gen`, `partner_asof` | every node, only for **its own** qubits | who this node thinks its far end is, and when it last confirmed that |
| **message** `(target, target_gen, kind, new_remote, new_remote_gen, author, t)` | in flight | addressed to one *occupancy* of one slot |

`age[q] >= 0` with `partner[q] == 0` is a **hanging qubit**: occupied, ageing,
blocking its link, and useless. Its owner has no local way to see this.

### The ledger carries ages, and nothing else

`chain_age` and `chain_asof` are the *only* things flooded. A node learns that a
distant slot is occupied and how old it is — never what it is entangled with.
Partner information lives in `partner`/`partner_gen`/`partner_asof`, which a node
maintains **only for its own qubits**, and which changes only through heralding
or an arriving `UPDATE`/`DELETE`. So "node 3's ledger says q2 has age 0" means
node 3 knows node 2's left slot is full and fresh — not that there is a usable
link there. Knowing a slot is occupied is not knowing it is *alive*, and that gap
is exactly what makes a hanging qubit invisible.

### When these snapshots are taken, and when a node can act on them

**Every ledger table below is the state at the *end* of its timestep, after
phase 6.** It is not what the node knew while deciding. The first chance to act
on anything in it is **phase 4 of the following timestep**.

The two channels reach a neighbour by different routes, with the same effective
latency:

| | route | lands | first usable |
|---|---|---|---|
| ledger | written straight into the neighbour's arrays in **phase 6** — no queue; the one-hop limit comes from snapshotting every node's table *before* any merging | phase 6 of `t` | phase 4 of `t+1` |
| `Msg` | pushed to the neighbour's `inbox` in phase 6, drained in **phase 3** | phase 3 of `t+1` | phase 4 of `t+1` |

### The staleness law

With that in mind, one relation holds exactly and is worth checking against any
row:

> During phase 4 of timestep `t`, node `i`'s ledger entry for a qubit owned by
> node `j` carries `asof = t - |i - j|`.

A node's own entries are always `asof = t` — a node sees its own slots perfectly.
So a table printed at the end of timestep `t` shows entries that node `i` will
read at `t+1` as being `|i - j|` timesteps stale.

---

# Episode A — a link is built, extended, and never acknowledged

`seed=424`. Delivers at **t = 3**. This episode shows the ledger filling in, a
successful swap emitting `UPDATE`s, a belief that is stale but harmless, and a
message that gets re-addressed twice and then dropped.

## t = 1 — generation, and the ledger's first fill

LLEG succeeds on link 1 only. Qubits 1 and 2 are created with `gen` 1 and 2.
Success is **heralded**, so nodes 1 and 2 both learn immediately — this is the
only undelayed information in the model.

```
truth   age [ 0,  0, -1, -1, -1, -1]   partner [2, 1, 0, 0, 0, 0]   gen [1, 2, 0, 0, 0, 0]
```

Nobody swaps (nodes 2 and 3 each hold at most one link). No messages are sent —
messages are only emitted when a link is *destroyed or redirected*, never when
one is created.

**Ledgers at the end of t = 1** (after phase 6):

| node | `chain_age` q1..q6 | `chain_asof` q1..q6 | knows about |
|---|---|---|---|
| 1 | `[ 0, 0,-1,-1,-1,-1]` | `[ 1, 1, 1,-1,-1,-1]` | own q1, plus q2,q3 from node 2 |
| 2 | `[ 0, 0,-1,-1,-1,-1]` | `[ 1, 1, 1, 1, 1,-1]` | own q2,q3, plus q1 and q4,q5 |
| 3 | `[-1, 0,-1,-1,-1,-1]` | `[-1, 1, 1, 1, 1, 1]` | own q4,q5, plus q2,q3 and q6 |
| 4 | `[-1,-1,-1,-1,-1,-1]` | `[-1,-1,-1, 1, 1, 1]` | own q6, plus q4,q5 |

`asof = -1` means *never heard of*. After one timestep the information front has
moved exactly one hop: node 4 still has no entry for q1 (3 hops away), node 3 has
none either (2 hops).

Node 3's `chain_age[2] = 0` is worth pausing on, since it is the row most easily
misread. Nodes 2 and 3 are adjacent, so one hop of flooding is all it takes.
But note what node 3 does and does not have:

- It knows **q2 is occupied and its age is 0**. That is all the ledger carries.
- It does **not** know q2 is entangled with q1, or with anything at all — no
  partner information is ever flooded.
- It cannot **act** on this until **phase 4 of t = 2**. This table is the state
  after phase 6 of t = 1; no decision taken during t = 1 used it.

Nothing here is a decision node 3 made at t = 1 — at t = 1 it held no qubits and
chose `(False, ())`, like everyone else.

## t = 2 — a successful swap, and the first messages

LLEG succeeds on link 2 (qubits 3, 4; `gen` 3, 4). Qubits 1, 2 are now age 1.

**What each node sees when it decides (phase 4):**

| node | observation | decoded |
|---|---|---|
| 1 | `(1, 1, 1, 1)` | q1: age 1, far end 1 hop away, belief 1 step stale, ledger says far end alive |
| 2 | `(1, 1, 1, 1, 0, 1, 0, 0)` | q2: age 1, far 1 hop, stale 1, **alive**; q3: age 0, far 1 hop, stale 0, **status 0** |
| 3 | `(0, 1, 0, 0, -1, 0, 0, 0)` | q4: age 0, far 1 hop, fresh, status 0; q5: empty |
| 4 | `(-1, 0, 0, 0)` | q6: empty |

Node 2's `q3` entry is the first interesting one. Its belief about q3's far end
is **perfectly fresh** (`stale = 0`, set by heralding this same timestep), yet
`status = 0` — "the ledger projects nothing". Node 2's ledger entry for q4 was
written at t = 1, when q4 was still empty. **The heralded belief and the flooded
ledger disagree, and the ledger is the one that is behind.** The two channels
travel at different effective speeds.

Node 2 has both slots believed-linked, so swap-asap fires:

```
node2 SWAP q2,q3    truth far = (1, 4)    belief far = (1, 4)    usable → p_s roll → SUCCESS
truth   age [ 1, -1, -1,  0, -1, -1]   partner [4, 0, 0, 1, 0, 0]
```

Qubits 1 and 4 are now entangled with each other — a two-hop virtual link —
while node 2's own two slots are consumed. **Neither node 1 nor node 3 knows
this yet.**

**Messages emitted** (to whoever node 2 *believed* was on each far end):

| message | meaning |
|---|---|
| `(1, 1, update, 4, 4, author=2, t=2)` | "q1 (occupancy 1): your far end is now q4 (occupancy 4)" |
| `(4, 4, update, 1, 1, author=2, t=2)` | "q4 (occupancy 4): your far end is now q1 (occupancy 1)" |

**History entries recorded at node 2** — the forwarding table for anything still
addressed to the slots it just consumed:

```
history[(2, 2)] = (4, 4)      history[(3, 3)] = (1, 1)
```

Read as: "anything addressed to my q2/occupancy 2 should really go to q4; my
q3/occupancy 3 now lives at q1."

**Ledgers at the end of t = 2:**

| node | `chain_age` | `chain_asof` |
|---|---|---|
| 1 | `[ 1,-1,-1,-1,-1,-1]` | `[ 2, 2, 2, 1, 1,-1]` |
| 2 | `[ 1,-1,-1, 0,-1,-1]` | `[ 2, 2, 2, 2, 2, 1]` |
| 3 | `[ 0,-1,-1, 0,-1,-1]` | `[ 1, 2, 2, 2, 2, 2]` |
| 4 | `[-1, 0,-1, 0,-1,-1]` | `[-1, 1, 1, 2, 2, 2]` |

Node 4's row is the staleness law in action: `asof[q2] = asof[q3] = 1` at t = 2
because `|4 - 2| = 2`, and it still has no entry at all for q1. Note also that
node 4's `chain_age[2] = 0` is now **wrong** — q2 was consumed by the swap this
timestep — but node 4 will not find out for another two timesteps.

## t = 3 — an UPDATE lands, a same-timestamp message does not

LLEG succeeds on link 3 (qubits 5, 6; `gen` 5, 6).

**Phase 3, messages arrive.** Both `UPDATE`s are delivered, but only one is
applied:

```
node1  belief q1: partner 2 → 4 (asof 2)        APPLIED
node3  belief q4: partner stays 3               NOT APPLIED
```

Node 3's rejection is the guard `msg.t > partner_asof[target]`. The message
carries `t = 2`; node 3's belief about q4 was stamped `asof = 2` by heralding
during phase 2 of t = 2. `2 > 2` is false, so the belief stands. **Node 3
therefore still believes q4's far end is q3, when in truth it is q1.**

**Observations:**

| node | observation | decoded |
|---|---|---|
| 1 | `(2, 2, 1, 0)` | q1: age 2, far end now **2 hops** away, stale 1, status 0 |
| 3 | `(1, 1, 1, 0, 0, 1, 0, 0)` | q4: age 1, far believed 1 hop, stale 1, status 0; q5: age 0, fresh |
| 4 | `(0, 1, 0, 0)` | q6: age 0, fresh |

Node 1's `dist` jumped from 1 to 2 — that is the `UPDATE` being visible in the
policy's input. It is the only signal a node ever gets that its link has been
*extended* rather than destroyed.

Node 3 now walks into the trap this model is built to expose. Its belief says
"q4's far end is q3", and its own ledger says `chain_age[3] = -1` — **q3 is
empty**. A naive rule ("my believed far end is reported empty, so I must be
hanging → discard") would fire here and would be **wrong**: q4 holds a perfectly
good link, to q1, and only the *label* is out of date. Distinguishing "my link
was destroyed" from "my link was extended" is exactly what the `UPDATE`/`DELETE`
distinction exists for, and the age ledger alone cannot do it.

Swap-asap does not read any of that; it sees two believed-live slots and fires:

```
node3 SWAP q4,q5    truth far = (1, 6)    belief far = (3, 6)    usable → p_s roll → SUCCESS
truth   age [ 2, -1, -1, -1, -1,  0]   partner [6, 0, 0, 0, 0, 1]
```

`partner[1] == 6` — **the end nodes now share an end-to-end link. The episode
ends here, at t = 3.** Note the swap succeeded *despite* node 3 acting on a wrong
belief: beliefs gate only *whether* a node swaps, while ground truth decides the
outcome.

The two messages node 3 sends are both misaddressed, because they are addressed
from its stale belief:

| message | intended for | problem |
|---|---|---|
| `(3, 3, update, 6, 6, 3, 3)` | q3 at node 2 | q3 was consumed at t = 2 |
| `(6, 6, update, 3, 3, 3, 3)` | q6 at node 4 | tells node 4 its far end is q3 — but it is really q1 |

## t = 4–6 — where those messages actually go

*The real episode stopped at t = 3. The following is the same run with the
delivery check suppressed, so the routing can be followed to its end.*

**t = 4.** Node 2 receives the message addressed to `(3, 3)`, finds it in its
history table, and **re-addresses** it:

```
history[(3, 3)] = (1, 1)   →   (3, 3, update, 6, 6, 3, 3)  becomes  (1, 1, update, 6, 6, 3, 3)
                               forwarded onward to node 1
```

Node 4 receives its `UPDATE` — and rejects it, `3 > 3` being false, for the same
same-timestamp reason node 3 rejected one at t = 3. Node 4 goes on believing its
far end is q5.

Node 1's observation this timestep is `(3, 2, 2, 1)`, and every field of it is a
lie except the first:

- `age = 3` — true, and local.
- `dist = 2` — it believes its far end is q4. Truth: q6, three hops away.
- `stale = 2` — saturated at `S_MAX`; the belief is genuinely 2 steps old.
- `status = 1` — "the ledger projects the far end is alive". Node 1's entry is
  `chain_age[4] = 0, chain_asof[4] = 2`, so it projects `0 + (4 - 2) = 2 ≤ t_cut`.
  **Truth: q4 was consumed at t = 3.** The entry is two hops stale and confidently
  wrong.

Then node 1's own cutoff sweep fires — q1 has reached `age == t_cut == 3` — and
destroys the end-to-end link from the inside:

```
cutoff sweep [1]
truth   age [-1, -1, -1, -1, -1,  1]   partner [0, 0, 0, 0, 0, 0]
hanging [6]
node1 emits (4, 4, delete, 0, 0, author=1, t=4)
```

Node 1 announces the death to `q4` — the qubit it *believed* it was linked to,
which has not existed since t = 3. Meanwhile q6 is now hanging and node 4 has no
idea.

**t = 5.** The re-addressed `UPDATE` finally reaches node 1 — and is dropped by
node 1's **own history table**. The cutoff sweep that killed q1 at t = 4 went
through `apply_discard`, which recorded the occupancy as having gone nowhere:

```
history[(1, 1)] = (0, 0)   →   (1, 1, update, 6, 6, 3, 3) arrives → dropped
```

Note it is the history lookup that fires here, not the "is this slot still
occupied?" check further down — history is consulted first, and every way a slot
can be emptied writes a history entry. See
[the occupancy-stamp note](#a-note-on-the-occupancy-stamp-guard).

**Node 1 never learns it once held an end-to-end link.** The `DELETE` from t = 4
reaches node 2, which does not own q4, so it forwards it toward node 3.

**t = 6.** The `DELETE` reaches node 3, which re-addresses it through *its*
history:

```
history[(4, 4)] = (6, 6)   →   (4, 4, delete, …)  becomes  (6, 6, delete, …)  → node 4
```

So a single announcement has now travelled node 1 → 2 → 3 → 4, being re-addressed
once en route. It will arrive at node 4 at t = 7 — where q6 was swept by the
cutoff at t = 6, which recorded `history[(6, 6)] = (0, 0)`, so node 4 drops it.
The message chased the entanglement across the entire chain and arrived to find
nothing left.

Also at t = 6, LLEG refills q1–q4 with **new occupancy stamps** (`gen` 7, 8, 9,
10). This is what the stamps are for: any of the older messages still in flight,
all addressed to `gen ≤ 6`, can no longer be misapplied to these fresh links.

---

# Episode B — one bad swap, two hanging qubits

`seed=43`. Delivers at t = 5. This is the compounding failure described in
[INFO-DELAY.md §5](INFO-DELAY.md), caught in a single timestep.

## t = 1–2 — setup

t = 1 generates links 1 and 3 (qubits 1,2 and 5,6). t = 2 generates link 2
(qubits 3,4), so all three elementary links exist:

```
truth   age [ 1,  1,  0,  0,  1,  1]   partner [2, 1, 4, 3, 6, 5]
```

Both middle nodes see two believed-live slots, so **both fire in the same
timestep**. They are applied in node order, which matters.

**Node 2 swaps first.** Its inputs are genuinely good, so the `p_s` coin is
tossed — and lost:

```
node2 SWAP q2,q3    truth far = (1, 4)    usable = True → p_s roll → FAILURE
```

The failure branch frees both of node 2's slots and zeroes the far side of each
destroyed link, so **q1 and q4 are now hanging**.

**Node 3 swaps second — into the wreckage.** Its left qubit is q4, which became
hanging microseconds ago in the same phase:

```
node3 SWAP q4,q5    truth far = (0, 6)    usable = False → deterministic FAILURE
```

`farL == 0`, so `usable` is false and **`p_s` is never even rolled**. The BSM is
still physical: both of node 3's qubits are consumed, and the good side's far end
is zeroed too.

```
truth   age [ 1, -1, -1, -1, -1,  1]   partner [0, 0, 0, 0, 0, 0]
hanging [1, 6]
```

Q6 was a healthy half of a healthy link at the start of this phase. It is now
hanging, because it was fed into a swap whose other input was already dead. That
is the compounding effect: node 3 could not distinguish a live q4 from a dead
one, and turned one victim into two.

**Four `DELETE` messages are emitted:**

| author | message | fate |
|---|---|---|
| node 2 | `(1, 1, delete, 0, 0, 2, 2)` | applied at node 1, t = 3 |
| node 2 | `(4, 6, delete, 0, 0, 2, 2)` | **dropped** at node 3, t = 3 |
| node 3 | `(3, 5, delete, 0, 0, 3, 2)` | **dropped** at node 2, t = 3 |
| node 3 | `(6, 4, delete, 0, 0, 3, 2)` | applied at node 4, t = 3 |

The two middle messages are addressed to slots the *recipient itself* consumed in
this very timestep, so each is caught by the recipient's history table, finds
`(0, 0)` — "that entanglement went nowhere" — and is discarded rather than
forwarded. Without that table they would circulate forever.

## t = 3 — the victims find out, and can do nothing

```
node1  belief q1: partner 2 → 0 (asof 2)
node4  belief q6: partner 5 → 0 (asof 2)
```

Their observations become `(2, 0, 0, 0)`: occupied, age 2, and `dist = 0`, which
in this encoding means **"I know I am hanging."** No ledger projection was
needed; the targeted `DELETE` said so outright.

Swap-asap does nothing with this. Both qubits sit occupied and useless, blocking
LLEG on links 1 and 3 — generation requires *both* slots of a link to be free —
until the cutoff sweeps them at t = 4, when they reach `age == t_cut == 3`.

**This is precisely the gap `CutoffAwareSwap` closes.** Its rule 1 fires on
`dist == 0` and discards at t = 3, one timestep before the sweep, freeing links 1
and 3 for regeneration a timestep earlier. Whether that is worth anything depends
entirely on `t_cut`: here it buys one timestep, but at `t_cut = 2` the news would
have arrived exactly as the sweep ran, which is why the measured gain there is
exactly zero.

## t = 4–5 — recovery

t = 4: the cutoff sweeps q1 and q6. Neither emits a message — both nodes already
believe `partner == 0`, and a `DELETE` is only sent to a *believed* far end. LLEG
refills link 2 (`gen` 7, 8).

t = 5: LLEG refills links 1 and 3. All three links live again; nodes 2 and 3 both
swap, both succeed, and the composition in node order joins q1 to q6. **Delivered
at t = 5.**

---

# The ledger merge, in full

The merge itself is three lines (`transmit!`, `info_delay_setup.jl:404`):

```julia
for i in 1:c.n, q in ownqubits(i, c.n)          # 1. re-assert own truth
    views[i].chain_age[q] = c.age[q]; views[i].chain_asof[q] = t
end
snapshots = [(copy(v.chain_age), copy(v.chain_asof)) for v in views]   # 2. freeze
for i in 1:c.n, j in (i - 1, i + 1)             # 3. merge from both neighbours
    theirage, theirasof = snapshots[j]
    for q in 1:c.nq
        if theirasof[q] > v.chain_asof[q]       #    newer stamp wins
            v.chain_age[q] = theirage[q]; v.chain_asof[q] = theirasof[q]
        end
    end
end
```

Three properties are worth spelling out.

**Snapshots are frozen before any merging.** Without this, node 2 could merge
from node 1 and then node 3 could merge node 1's entry out of node 2 in the same
loop, and information would move two hops in one timestep. The snapshot is what
enforces the one-hop-per-timestep speed limit.

**The comparison is on `asof`, not `age`.** A node keeps whichever report was
*stamped* later, regardless of which neighbour it came from or when it arrived.
On a linear chain each entry has only one shortest path, but the rule is what
makes the merge idempotent and order-independent.

**Emptiness propagates like any other value.** When a qubit is swept, its owner
writes `chain_age[q] = -1` with a fresh stamp, and that `-1` floods outward
exactly like an age would. `projected_age` returns `-1` for it — the *same* value
it returns for "never heard of". The encoded `status` feature therefore maps both
to `0`, so **"I have no information" and "I know that slot is empty" are
indistinguishable in the tabular observation.** The raw observation passed to
handwritten policies does carry enough to separate them (`chain_asof[q] >= 0 &&
chain_age[q] < 0` is "known empty"), and no policy currently checks it.

### Worked propagation: node 4's entry for q1, Episode A

Node 1 and node 4 are 3 hops apart, the maximum for `n = 4`.

| end of | node 2 `asof[1]` | node 3 `asof[1]` | node 4 `asof[1]` | node 4's entry |
|---|---|---|---|---|
| t = 1 | 1 | — | — | never heard |
| t = 2 | 2 | 1 | — | never heard |
| t = 3 | 3 | 2 | **1** | `age = 0`, stamped t = 1 |
| t = 4 | 4 | 3 | 2 | `age = 1`, stamped t = 2 |

Node 1 stamped `age[q1] = 0` at t = 1. That entry reaches node 2 at the end of
t = 1, node 3 at the end of t = 2, and node 4 at the end of t = 3 — one hop per
timestep. So when node 4 reads it during phase 4 of t = 4, it sees a report
stamped `t = 1`: **exactly `|4 - 1| = 3` timesteps stale**, and it must project
forward by 3 to use it:

```
projected_age(q1) = chain_age[1] + (t - chain_asof[1]) = 0 + (4 - 1) = 3
```

That projection is only correct if nothing happened to q1 in the intervening
three timesteps. In this episode q1 was swept by node 1's own cutoff at t = 4, so
node 4's projection of "age 3, still alive" was about to be wrong — and would
stay wrong until the `-1` propagated three hops back.

---

## A note on the occupancy-stamp guard

`receive!` checks four things in order: the history table, ownership, then
`occupied(target) && gen[target] == target_gen`, then `msg.t > partner_asof`.
Classifying every arriving message over 4,000 episodes:

| outcome | n=4, p=0.6, t_cut=3 | n=5, p=0.9, t_cut=6 |
|---|---|---|
| history: re-addressed or dropped | 50.6% | 48.4% |
| dropped, not newer than current belief | 33.8% | 41.5% |
| **applied** | **12.3%** | **5.9%** |
| forwarded (not mine) | 3.3% | 4.2% |
| dropped, slot empty | **0** | **0** |
| dropped, slot refilled (`gen` mismatch) | **0** | **0** |

The occupancy check never fires, in 130,000 arrivals. The reason is structural:
every way a node's slot can be emptied — `apply_swap` on either branch, and
`apply_discard`, whether voluntary or from the cutoff sweep — writes a `history`
entry for that occupancy first. Since `history` is consulted first and is never
pruned, any message addressed to a dead occupancy is caught there.

So the `gen` stamps are currently **defensive redundancy** rather than a
load-bearing guard. They would become load-bearing the moment `history` is
bounded or pruned — which a real implementation would need, since it grows
without limit over an episode. Worth knowing before anyone "optimizes" either
one away.

Note also that only ~6–12% of arriving messages change anything. The largest
single category after history is "not newer than what I already believe", which
includes the same-timestamp rejections discussed above.

## What each mechanism looks like, and where to see it

| mechanism | where in the trace |
|---|---|
| Heralding (undelayed) | A t=1: nodes 1 and 2 both know instantly |
| Ledger propagation front | A t=1 `asof` rows: `-1` beyond one hop |
| Staleness law `asof = t - \|i-j\|` | A t=4: node 4's q1 entry stamped t=1 |
| `UPDATE` applied | A t=3: node 1's `dist` 1 → 2 |
| Same-timestamp rejection | A t=3 node 3, A t=4 node 4 (`msg.t > asof` fails) |
| Belief stale but swap still works | A t=3: node 3 believes q3, truth is q1, swap succeeds |
| History re-addressing | A t=4 (`(3,3)`→`(1,1)`), A t=6 (`(4,4)`→`(6,6)`) |
| History drop (`(0,0)`) | B t=3: two of four `DELETE`s discarded |
| History drop of a swept slot | A t=5: cutoff wrote `history[(1,1)] = (0,0)`, `UPDATE` dropped |
| Ledger confidently wrong | A t=4: node 1 reads `status = 1` for a qubit gone since t=3 |
| Hanging qubit created by `p_s` failure | B t=2: node 2's roll fails → q1, q4 hang |
| Hanging input consumed by a swap | B t=2: node 3, `usable = False`, no `p_s` roll, q6 hangs |
| `DELETE` telling a victim | B t=3: nodes 1 and 4 see `dist = 0` |
| Cutoff emitting nothing | B t=4: belief already `0`, so no message |
| Occupancy stamps advancing | A t=6: `gen` 7–10 issued, older messages now unmatchable |

---

## Appendix: reproducing this

Save as `trace.py` beside `src/`, then `python trace.py`. Set `max_t` past the
delivery timestep to follow message routing after the episode would have ended,
as in Episode A t = 4–6.

```python
import sys, random, copy
sys.path.insert(0, "src")
import info_delay_solver as S

def run(n, p, p_s, t_cut, seed, max_t):
    rng = random.Random(seed)
    chain = S.Chain(n)
    views = [None] + [S.NodeView(i, n) for i in range(1, n + 1)]
    inbox = [None] + [[] for _ in range(n)]
    outbox = [None] + [[] for _ in range(n)]
    forwards = [None] + [[] for _ in range(n)]
    for t in range(1, max_t + 1):
        print("=" * 72); print("t =", t)
        for q in range(1, chain.nq + 1):
            if chain.age[q] >= 0: chain.age[q] += 1
        S.attempt_generation(chain, views, t, p, rng)
        print("  truth after LLEG   age", chain.age[1:],
              "partner", chain.partner[1:], "gen", chain.gen[1:])
        if chain.delivered(): print("  >>> delivered() is TRUE here")
        print("  inbox arriving     ", {i: inbox[i] for i in range(1, n+1) if inbox[i]})
        pre = [None] + [copy.deepcopy(v) for v in views[1:]]
        S.receive_messages(chain, views, inbox, forwards)
        print("  re-addressed/fwd   ", {i: forwards[i] for i in range(1, n+1) if forwards[i]})
        for i in range(1, n + 1):
            for q in range(1, chain.nq + 1):
                if (views[i].partner[q], views[i].partner_asof[q]) != \
                   (pre[i].partner[q], pre[i].partner_asof[q]):
                    print("  node%d belief q%d: partner %d -> %d (asof %d)"
                          % (i, q, pre[i].partner[q], views[i].partner[q],
                             views[i].partner_asof[q]))
        acts = {}
        for i in range(1, n + 1):
            o = S.encode_observation(chain, views[i], i, t, t_cut)
            acts[i] = S.swapasap_action(i, n, o)
            print("  node%d obs %-28s -> %s" % (i, o, S.action_space(i, n)[acts[i]]))
        for i in range(1, n + 1):
            if S.action_space(i, n)[acts[i]][0]:
                qL, qR = S.ownqubits(i, n)
                print("  node%d SWAP q%d,q%d  truth far=(%s,%s) belief far=(%s,%s)"
                      % (i, qL, qR, chain.partner[qL], chain.partner[qR],
                         views[i].partner[qL], views[i].partner[qR]))
                k = len(outbox[i])
                S.apply_swap(chain, views, outbox, i, t, p_s, rng)
                print("       emits", outbox[i][k:], " history", dict(views[i].history))
        for i in range(1, n + 1):
            own = S.ownqubits(i, n)
            for slot in S.action_space(i, n)[acts[i]][1]:
                if chain.occupied(own[slot]):
                    S.apply_discard(chain, views, outbox, own[slot], t)
        cut = [q for q in range(1, chain.nq + 1) if chain.age[q] == t_cut]
        S.apply_cutoff(chain, views, outbox, t, t_cut)
        if cut: print("  cutoff sweep       ", cut)
        print("  truth end of step  age", chain.age[1:], "partner", chain.partner[1:])
        print("  hanging            ",
              [q for q in range(1, chain.nq+1) if chain.age[q] >= 0 and chain.partner[q] == 0])
        print("  outbox (to send)   ", {i: outbox[i] for i in range(1, n+1) if outbox[i]})
        S.transmit(chain, views, inbox, outbox, forwards, t)
        print("  LEDGER age  ", {i: views[i].chain_age[1:] for i in range(1, n+1)})
        print("  LEDGER asof ", {i: views[i].chain_asof[1:] for i in range(1, n+1)})
        print("  in flight   ", {i: inbox[i] for i in range(1, n+1) if inbox[i]})

run(4, 0.6, 0.5, 3, 424, 6)   # Episode A
run(4, 0.6, 0.5, 3, 43, 5)    # Episode B
```
