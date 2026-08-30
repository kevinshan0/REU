"""Decentralized policy iteration for the delayed-information repeater chain.

policy.py solves the *global-knowledge* MDP exactly: one agent sees the whole
chain and picks a joint action, so the state is Markov and policy iteration is
exact. info_delay_setup.jl removes that assumption -- each node sees only its own
slots plus classical information that takes one timestep per hop -- which turns
the problem into a Dec-POMDP. Solving those optimally is NEXP-complete, so this
module does the standard tractable thing instead (JESP, Nair et al. 2003):

    hold every other node's policy fixed, compute one node's best response,
    move to the next node, repeat until nobody wants to change.

That converges to a Nash equilibrium over the joint policy space, not a global
optimum -- a different initialization can land somewhere else. It is still a real
policy iteration: each best-response step evaluates the current policy exactly by
solving (I - P_pi) V = -1 with a sparse LU (same convention and same machinery as
policy.py's _evaluate_policy_direct, reward -1 per timestep so V = -E[delivery
time]), then improves greedily against that V.

The one place this departs from policy.py is where the transition model comes
from. A node's local observation is *not* Markov -- what happens next also depends
on the parts of the chain it cannot see, and on what the other nodes are doing --
so there is no transition function to write down in closed form. Each
best-response step therefore estimates P(o'|o,a) empirically, from a batch of
Monte-Carlo rollouts of the joint policy with epsilon-greedy exploration, and then
solves that empirical MDP exactly. Fixing the other agents is what makes the
environment stationary enough for this to be meaningful; it is still an
approximation, and `--episodes` controls how good one.

The dynamics below are a port of info_delay_setup.jl with the quantum physics
left out: the solver optimizes expected delivery time only, exactly as policy.py
does for the global model. Fidelity is not modelled here at all -- it is measured
afterwards by running the exported policy through the QuantumSavory simulator,
which is the same division of labour simulate.jl already uses. `--verify` checks
the port against the Julia implementation by comparing swap-asap statistics.

Output is a JSON table per node, keyed on the encoded local observation, which
info_delay_setup.jl's `TabularPolicy` loads directly.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import os
import random
import subprocess
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
SIMULATOR_JL = SRC_DIR / "info_delay_setup.jl"
DATA_DIR = SRC_DIR / "data_infodelay"

os.chdir(SRC_DIR)

# How many timesteps of staleness the observation encoding distinguishes before
# saturating. Raising it makes the policy finer-grained but splits the rollout
# statistics over more states. MUST match S_MAX in info_delay_setup.jl.
S_MAX = 2


# --------------------------------------------------------------------------- #
# ------------------------  FLAT INDEX / CHAIN LAYOUT  ---------------------- #
# --------------------------------------------------------------------------- #
# Flat qubit indices are 1-based and kept identical to info_delay_setup.jl's, so
# the two implementations can be compared line by line. Arrays are sized nq+1
# with slot 0 unused rather than being rebased to 0.

def owner(q):
    """Node (1-based) owning flat qubit q."""
    return q // 2 + 1


def ownqubits(node, n):
    """Flat indices of the qubits `node` owns, left-facing first."""
    if node == 1:
        return (1,)
    if node == n:
        return (2 * n - 2,)
    return (2 * node - 2, 2 * node - 1)


def action_space(node, n):
    """The actions available to one node, as (swap, discard) with discard given
    as positions into ownqubits(node, n). Swapping and discarding in the same
    timestep is excluded: a swap already consumes both slots."""
    if node == 1 or node == n:
        return [(False, ()), (False, (0,))]
    return [(False, ()), (True, ()), (False, (0,)), (False, (1,)), (False, (0, 1))]


WAIT = 0   # index of the do-nothing action, in every node's action space
SWAP = 1   # index of the swap action, middle nodes only


# --------------------------------------------------------------------------- #
# ---------------------------  DYNAMICS (PORT)  ----------------------------- #
# --------------------------------------------------------------------------- #

class Chain:
    """Ground truth: what actually happened, which no node ever gets to see.

    age[q] == -1 is an empty slot; partner[q] == 0 is a qubit entangled with
    nothing, so age[q] >= 0 with partner[q] == 0 is a *hanging* qubit. gen[q] is
    bumped every time the slot is filled, so a message about a long-dead link
    cannot be applied to whatever occupies the slot now."""

    __slots__ = ("n", "nq", "age", "partner", "gen", "gencount")

    def __init__(self, n):
        self.n = n
        self.nq = 2 * n - 2
        self.age = [-1] * (self.nq + 1)
        self.partner = [0] * (self.nq + 1)
        self.gen = [0] * (self.nq + 1)
        self.gencount = 0

    def occupied(self, q):
        return self.age[q] >= 0

    def delivered(self):
        return self.age[1] >= 0 and self.partner[1] == self.nq

    def free(self, q):
        self.age[q] = -1
        self.partner[q] = 0


class NodeView:
    """Everything one node believes, and nothing it has no way of knowing.

    partner/partner_gen/partner_asof are its belief about the far end of its own
    qubits -- the belief that can be wrong, and the reason it cannot tell a live
    link from a hanging one. chain_age/chain_asof are the flooded ledger, so this
    node's entry for a qubit at node j is j's truth as of t - |i-j|. history
    re-addresses messages that arrive for an occupancy already swapped away."""

    __slots__ = ("node", "n", "partner", "partner_gen", "partner_asof",
                 "chain_age", "chain_asof", "history")

    def __init__(self, node, n):
        nq = 2 * n - 2
        self.node = node
        self.n = n
        self.partner = [0] * (nq + 1)
        self.partner_gen = [0] * (nq + 1)
        self.partner_asof = [-1] * (nq + 1)
        self.chain_age = [-1] * (nq + 1)
        self.chain_asof = [-1] * (nq + 1)
        self.history = {}

    def projected_age(self, q, t):
        """Age the flooded ledger implies q has *now*; -1 if never heard about."""
        if self.chain_asof[q] < 0 or self.chain_age[q] < 0:
            return -1
        return self.chain_age[q] + (t - self.chain_asof[q])


def attempt_generation(chain, views, t, p, rng):
    """LLEG on every elementary link with both slots free. Success is heralded,
    so both endpoints learn about it in the same timestep -- the only knowledge
    in the model that is not delayed."""
    for k in range(1, chain.n):
        qa, qb = 2 * k - 1, 2 * k
        if chain.occupied(qa) or chain.occupied(qb):
            continue
        if rng.random() >= p:
            continue
        ga, gb = chain.gencount + 1, chain.gencount + 2
        chain.gencount += 2
        chain.age[qa] = chain.age[qb] = 0
        chain.partner[qa], chain.partner[qb] = qb, qa
        chain.gen[qa], chain.gen[qb] = ga, gb
        for q, remote, remotegen in ((qa, qb, gb), (qb, qa, ga)):
            v = views[owner(q)]
            v.partner[q] = remote
            v.partner_gen[q] = remotegen
            v.partner_asof[q] = t


def apply_swap(chain, views, outbox, node, t, p_s, rng):
    """Bell-state measurement on `node`'s own two qubits, committed to from its
    belief. If a slot was actually hanging the BSM still consumes both local
    qubits and leaves the good side's far end hanging too -- the compounding
    effect the whole investigation is about."""
    if node == 1 or node == chain.n:
        return
    v = views[node]
    qL, qR = ownqubits(node, chain.n)

    believedL, believedLgen = v.partner[qL], v.partner_gen[qL]
    believedR, believedRgen = v.partner[qR], v.partner_gen[qR]
    genL, genR = chain.gen[qL], chain.gen[qR]
    farL, farR = chain.partner[qL], chain.partner[qR]

    usable = chain.occupied(qL) and chain.occupied(qR) and farL != 0 and farR != 0
    success = usable and rng.random() < p_s

    if success:
        # qubit-retention model: the surviving far ends keep their own ages
        chain.partner[farL] = farR
        chain.partner[farR] = farL
        chain.free(qL)
        chain.free(qR)
    else:
        for own, far in ((qL, farL), (qR, farR)):
            if not chain.occupied(own):
                continue
            if far != 0:
                chain.partner[far] = 0
            chain.free(own)

    v.partner[qL] = v.partner[qR] = 0
    v.partner_asof[qL] = v.partner_asof[qR] = t
    v.history[(qL, genL)] = (believedR, believedRgen) if success else (0, 0)
    v.history[(qR, genR)] = (believedL, believedLgen) if success else (0, 0)

    kind = "update" if success else "delete"
    if believedL != 0:
        outbox[node].append((believedL, believedLgen, kind,
                             believedR if success else 0,
                             believedRgen if success else 0, node, t))
    if believedR != 0:
        outbox[node].append((believedR, believedRgen, kind,
                             believedL if success else 0,
                             believedLgen if success else 0, node, t))


def apply_discard(chain, views, outbox, q, t):
    """Drop qubit q, by choice or because it hit the retention limit. The far end
    is left hanging and will not find out until the announcement gets there."""
    node = owner(q)
    v = views[node]
    believed, believedgen = v.partner[q], v.partner_gen[q]
    gen = chain.gen[q]
    far = chain.partner[q]

    if far != 0:
        chain.partner[far] = 0
    chain.free(q)

    v.partner[q] = 0
    v.partner_asof[q] = t
    v.history[(q, gen)] = (0, 0)
    if believed != 0:
        outbox[node].append((believed, believedgen, "delete", 0, 0, node, t))


def apply_cutoff(chain, views, outbox, t, t_cut):
    for q in range(1, chain.nq + 1):
        if chain.age[q] == t_cut:
            apply_discard(chain, views, outbox, q, t)


def receive(chain, v, msg):
    """Handle one arrived message. Returns None if applied or dropped, or the
    (possibly re-addressed) message that has to keep travelling."""
    target, target_gen, kind, new_remote, new_remote_gen, author, mt = msg
    key = (target, target_gen)
    if key in v.history:
        newtarget, newgen = v.history[key]
        if newtarget == 0:
            return None
        return (newtarget, newgen, kind, new_remote, new_remote_gen, author, mt)
    if owner(target) != v.node:
        return msg
    if not (chain.occupied(target) and chain.gen[target] == target_gen):
        return None
    if mt > v.partner_asof[target]:
        isupdate = kind == "update"
        v.partner[target] = new_remote if isupdate else 0
        v.partner_gen[target] = new_remote_gen if isupdate else 0
        v.partner_asof[target] = mt
    return None


def receive_messages(chain, views, inbox, forwards):
    for i in range(1, chain.n + 1):
        msgs = inbox[i]
        inbox[i] = []
        for msg in msgs:
            onward = receive(chain, views[i], msg)
            if onward is not None:
                forwards[i].append(onward)


def transmit(chain, views, inbox, outbox, forwards, t):
    """Everything said this timestep lands on the neighbours next timestep and
    nowhere further, so information about node j reaches node i after exactly
    |i-j| timesteps."""
    for i in range(1, chain.n + 1):
        for q in ownqubits(i, chain.n):
            views[i].chain_age[q] = chain.age[q]
            views[i].chain_asof[q] = t
    snapshots = [None] + [(list(v.chain_age), list(v.chain_asof))
                          for v in views[1:]]
    for i in range(1, chain.n + 1):
        v = views[i]
        for j in (i - 1, i + 1):
            if not 1 <= j <= chain.n:
                continue
            theirage, theirasof = snapshots[j]
            for q in range(1, chain.nq + 1):
                if theirasof[q] > v.chain_asof[q]:
                    v.chain_age[q] = theirage[q]
                    v.chain_asof[q] = theirasof[q]

    for i in range(1, chain.n + 1):
        for msg in outbox[i] + forwards[i]:
            dest = owner(msg[0])
            if dest == i:
                continue
            inbox[i + 1 if dest > i else i - 1].append(msg)
        outbox[i].clear()
        forwards[i].clear()


# --------------------------------------------------------------------------- #
# --------------------  OBSERVATION ENCODING (SHARED)  ---------------------- #
# --------------------------------------------------------------------------- #

def encode_observation(chain, v, node, t, t_cut):
    """The compact local observation a node's policy is keyed on. Per own qubit,
    left-facing first:

        age     exact age of the slot, clipped at t_cut (-1 = empty). Local and
                always correct -- a node can see its own slots.
        dist    hops to the believed far end (0 = believed unentangled). This is
                a belief and can be stale or simply wrong.
        stale   timesteps since that belief was last confirmed, saturating at
                S_MAX. This is the node's only handle on how much it is risking.
        status  what the flooded ledger projects for the far end right now:
                0 never heard, 1 projected alive, 2 projected past its cutoff.

    Returned as a tuple of ints; MUST stay in lockstep with
    info_delay_setup.jl's `encode_observation`, which produces the same tuple in
    the same order as the lookup key."""
    feats = []
    for q in ownqubits(node, chain.n):
        age = chain.age[q]
        age = -1 if age < 0 else min(age, t_cut)
        farq = v.partner[q]
        if farq == 0:
            feats += [age, 0, 0, 0]
        else:
            # floored at 1 so that dist == 0 means "believed unentangled" and
            # nothing else -- otherwise a far end that resolved to this node's own
            # register would encode as 0 and read back as unentangled, and the two
            # implementations' swap-asap fallbacks would disagree about it
            dist = min(max(abs(owner(farq) - node), 1), chain.n - 1)
            asof = v.partner_asof[q]
            stale = S_MAX if asof < 0 else min(t - asof, S_MAX)
            far_age = v.projected_age(farq, t)
            status = 0 if far_age < 0 else (2 if far_age > t_cut else 1)
            feats += [age, dist, stale, status]
    return tuple(feats)


def swapasap_action(node, n, obs):
    """The action SWAP-ASAP would take from this observation. Doubles as the
    fallback for observations the solved table has no entry for, so a policy only
    has to store where it disagrees with the baseline -- info_delay_setup.jl's
    `TabularPolicy` falls back exactly the same way."""
    if node == 1 or node == n:
        return WAIT
    # obs is (age, dist, stale, status) per qubit; believed-live means the slot is
    # occupied and the node believes it still has a far end
    left_ready = obs[0] >= 0 and obs[1] != 0
    right_ready = obs[4] >= 0 and obs[5] != 0
    return SWAP if (left_ready and right_ready) else WAIT


def cutoffaware_action(node, n, obs):
    """`CutoffAwareSwap(margin=0)` from info_delay_setup.jl, rewritten against the
    encoded observation so it can serve as a JESP starting policy.

    The translation is exact, not an approximation, because every test that
    policy makes survives the encoding:

        o.age >= 0        <-> age >= 0        (age is -1 iff the slot is empty)
        o.far_q == 0      <-> dist == 0       (dist is floored at 1 when linked)
        o.far_age > t_cut <-> status == 2
        o.far_age < 0     <-> status == 0

    At margin=0 the two age guards drop out entirely: the encoded age is already
    clipped at t_cut, and a qubit is swept the moment it reaches t_cut, so
    `age <= t_cut` is vacuously true whenever the slot is occupied."""
    feats = [obs[i:i + 4] for i in range(0, len(obs), 4)]
    discard = [k for k, (age, dist, _stale, status) in enumerate(feats)
               if age >= 0 and (dist == 0 or status == 2)]
    if discard:
        if node == 1 or node == n:
            return 1                       # end node: [(wait), (discard own)]
        return {(0,): 2, (1,): 3, (0, 1): 4}[tuple(discard)]
    if node == 1 or node == n:
        return WAIT
    ready = all(age >= 0 and dist != 0 and status != 2
                for (age, dist, _stale, status) in feats)
    return SWAP if ready else WAIT


# Starting policies JESP can be seeded from. The choice matters: JESP only ever
# asks whether a *single* node gains by deviating, so an improvement that no one
# node can reach alone is invisible from a bad starting point.
FALLBACKS = {"swap-asap": swapasap_action, "cutoff-aware": cutoffaware_action}


def obs_key(obs):
    """JSON object keys have to be strings."""
    return ",".join(str(x) for x in obs)


# --------------------------------------------------------------------------- #
# --------------------------------  ROLLOUTS  -------------------------------- #
# --------------------------------------------------------------------------- #

class JointPolicy:
    """One lookup table per node, mapping encoded observation -> action index.

    Missing entries fall back to `fallback`, so an empty JointPolicy *is* that
    fallback policy. Tables therefore only ever store disagreements with it, and
    the fallback name travels with the exported JSON so the Julia side can apply
    the same one."""

    def __init__(self, n, fallback="swap-asap"):
        if fallback not in FALLBACKS:
            raise ValueError(f"unknown fallback {fallback!r}; expected one of {sorted(FALLBACKS)}")
        self.n = n
        self.fallback = fallback
        self.tables = [None] + [{} for _ in range(n)]

    def fallback_action(self, node, obs):
        return FALLBACKS[self.fallback](node, self.n, obs)

    def act(self, node, obs):
        hit = self.tables[node].get(obs)
        return self.fallback_action(node, obs) if hit is None else hit

    def copy(self):
        clone = JointPolicy(self.n, self.fallback)
        clone.tables = [None] + [dict(t) for t in self.tables[1:]]
        return clone


def run_episode(policy, par, rng, explore_node=None, epsilon=0.0, record=None):
    """One episode of the delayed-information chain, phase for phase as in
    info_delay_setup.jl's run_trial. Returns the delivery time, or None on
    timeout.

    If `record` is a list, every (observation, action) the node `explore_node`
    took is appended to it in order, so the caller can pair step k's observation
    with step k+1's to get that node's empirical transitions. `epsilon` makes
    that node act uniformly at random that often, which is what gives the
    empirical model any coverage of the actions the current policy never picks."""
    n, p, p_s, t_cut = par["n"], par["p"], par["p_s"], par["t_cut"]
    chain = Chain(n)
    views = [None] + [NodeView(i, n) for i in range(1, n + 1)]
    inbox = [None] + [[] for _ in range(n)]
    outbox = [None] + [[] for _ in range(n)]
    forwards = [None] + [[] for _ in range(n)]
    explore_actions = action_space(explore_node, n) if explore_node else None

    for t in range(1, par["max_steps"] + 1):
        # 1. every stored qubit ages by one
        for q in range(1, chain.nq + 1):
            if chain.age[q] >= 0:
                chain.age[q] += 1

        # 2. LLEG wherever both slots are free
        attempt_generation(chain, views, t, p, rng)
        if chain.delivered():
            return t

        # 3. classical information from the neighbours lands
        receive_messages(chain, views, inbox, forwards)

        # 4. every node decides from its own delayed view, then swaps fire
        actions = [None] * (n + 1)
        for i in range(1, n + 1):
            obs = encode_observation(chain, views[i], i, t, t_cut)
            if i == explore_node:
                if epsilon > 0.0 and rng.random() < epsilon:
                    a = rng.randrange(len(explore_actions))
                else:
                    a = policy.act(i, obs)
                if record is not None:
                    record.append((obs, a))
            else:
                a = policy.act(i, obs)
            actions[i] = a

        for i in range(1, n + 1):
            swap, _ = action_space(i, n)[actions[i]]
            if swap:
                apply_swap(chain, views, outbox, i, t, p_s, rng)
        if chain.delivered():
            return t
        for i in range(1, n + 1):
            _, discard = action_space(i, n)[actions[i]]
            own = ownqubits(i, n)
            for slot in discard:
                q = own[slot]
                if chain.occupied(q):
                    apply_discard(chain, views, outbox, q, t)

        # 5. retention-time cutoff
        apply_cutoff(chain, views, outbox, t, t_cut)
        # 6. every node tells its neighbours what it now knows
        transmit(chain, views, inbox, outbox, forwards, t)

    return None


def evaluate(policy, par, episodes, rng):
    """Expected delivery time of the joint policy, by Monte Carlo. Also returns
    the sample standard deviation, which is emphatically *not* close to the mean:
    a chain that hangs a qubit has to wait out the full retention time before the
    slot frees up, so the delivery-time distribution has a long tail of episodes
    at multiples of t_cut. Any comparison of two means here has to be scaled by
    this, not by the mean."""
    times, timeouts = [], 0
    for _ in range(episodes):
        t = run_episode(policy, par, rng)
        if t is None:
            timeouts += 1
        else:
            times.append(t)
    if not times:
        return float("nan"), float("nan"), timeouts
    return float(np.mean(times)), float(np.std(times, ddof=1)), timeouts


def collect_transitions(policy, par, node, episodes, epsilon, rng, max_steps=None):
    """Roll out the joint policy and return `node`'s empirical MDP:

        counts[(o, a)][o'] -> visits, with o' None meaning the episode ended
                              (delivery), which is the absorbing state

    Every timestep costs the same -1 regardless of what anybody did, because the
    objective is shared: the whole chain is trying to minimize the time until an
    end-to-end link exists.

    `max_steps` overrides the episode cap for model estimation only. A greedy
    best response against a thinly-sampled model can produce a joint policy that
    takes hundreds of timesteps to deliver, and the next batch then spends all
    its time replaying that disaster in Python. Truncating costs nothing here --
    over-long episodes are dropped either way, since a truncated episode has no
    well-defined tail value -- but `evaluate` deliberately does not truncate, so
    the reported numbers stay honest."""
    if max_steps is not None:
        par = dict(par, max_steps=max_steps)
    counts = defaultdict(lambda: defaultdict(int))
    timeouts = 0
    for _ in range(episodes):
        record = []
        t = run_episode(policy, par, rng, explore_node=node, epsilon=epsilon,
                        record=record)
        if t is None:
            timeouts += 1
            # a timed-out episode has no well-defined tail value; its transitions
            # would bias every state on the path towards "this is fine"
            continue
        for k, (obs, a) in enumerate(record):
            nxt = record[k + 1][0] if k + 1 < len(record) else None
            counts[(obs, a)][nxt] += 1
    return counts, timeouts


# --------------------------------------------------------------------------- #
# -------------------  PER-AGENT POLICY ITERATION  --------------------------- #
# --------------------------------------------------------------------------- #

def best_response(policy, par, node, counts, min_visits):
    """Exact policy iteration on one node's empirical MDP, holding every other
    node's policy fixed.

    Evaluation is the same direct solve policy.py uses: for a fixed policy,
    V(o) = -1 + sum_o' P(o'|o,pi(o)) V(o') is |O| linear equations in |O|
    unknowns, i.e. (I - P_pi) V = -1, with the absorbing post-delivery state
    contributing nothing (V = 0 there). Improvement is then a greedy argmax over
    the actions that have at least `min_visits` samples -- an action nobody ever
    tried has no estimate, and picking it on the strength of one lucky rollout is
    how this kind of solver talks itself into nonsense.

    Returns the updated table for this node and how many entries changed."""
    actions = action_space(node, par["n"])
    states = sorted({o for (o, _) in counts})
    if not states:
        return dict(policy.tables[node]), 0
    index = {o: i for i, o in enumerate(states)}
    size = len(states)

    # P[o][a] -> (list of (o' index, prob), prob of terminating)
    model = defaultdict(dict)
    for (o, a), succ in counts.items():
        total = sum(succ.values())
        entries = []
        for nxt, c in succ.items():
            if nxt is None or nxt not in index:
                # terminal, or a successor this batch never visited as a source
                # state -- either way it contributes no column to the system
                continue
            entries.append((index[nxt], c / total))
        model[o][a] = (entries, total)

    table = dict(policy.tables[node])
    current = [table.get(o, policy.fallback_action(node, o)) for o in states]

    for _ in range(par.get("inner_iters", 50)):
        rows, cols, data = [], [], []
        for i, o in enumerate(states):
            entry = model[o].get(current[i])
            if entry is not None:
                for j, prob in entry[0]:
                    rows.append(i)
                    cols.append(j)
                    data.append(-prob)
            rows.append(i)
            cols.append(i)
            data.append(1.0)  # the "I" term; coo_matrix sums duplicate entries
        A = coo_matrix((data, (rows, cols)), shape=(size, size)).tocsc()
        b = -np.ones(size)
        try:
            V = spsolve(A, b)
        except Exception:
            break
        if not np.all(np.isfinite(V)):
            # a policy that never reaches delivery from some observation makes
            # (I - P_pi) singular; leave that agent where it is this round
            break

        stable = True
        for i, o in enumerate(states):
            best, best_q = current[i], None
            for a in range(len(actions)):
                entry = model[o].get(a)
                if entry is None or entry[1] < min_visits:
                    continue
                q = -1.0 + sum(prob * V[j] for j, prob in entry[0])
                if best_q is None or q > best_q + 1e-12:
                    best_q, best = q, a
            if best != current[i]:
                current[i] = best
                stable = False
        if stable:
            break

    changed = 0
    for i, o in enumerate(states):
        default = policy.fallback_action(node, o)
        prev = table.get(o, default)
        if current[i] != prev:
            changed += 1
        if current[i] == default:
            table.pop(o, None)  # keep the table to disagreements with the baseline
        else:
            table[o] = current[i]
    return table, changed


def solve(par, episodes=4000, outer_iters=12, epsilon=0.15, min_visits=25,
          eval_episodes=4000, seed=0, progress=True, fallback="swap-asap"):
    """JESP: sweep the nodes, replacing each one's policy with its best response
    to the others, until nobody changes. Each sweep re-rolls the batch after
    every single-node update, because the moment one node's policy changes the
    previous batch is describing a different joint policy.

    Returns (policy, history) where history records the greedy evaluation after
    each sweep. The best sweep is returned, not the last: MC evaluation is noisy
    and best response against a noisy model can step backwards.

    Each sweep's E[T] is printed with its standard error, because picking the
    argmin over sweeps biases that number optimistically -- the winner is partly
    winning on luck. The chosen policy is therefore re-evaluated at the end on a
    fresh batch, and that unbiased figure is what gets reported and compared."""
    rng = random.Random(seed)
    policy = JointPolicy(par["n"], fallback)  # empty table == the fallback policy
    base_time, base_sd, base_timeouts = evaluate(policy, par, eval_episodes, rng)
    base_se = base_sd / (eval_episodes ** 0.5)
    if progress:
        print("%s baseline: E[T] = %.4f +/- %.4f  (timeouts %d)"
              % (fallback, base_time, base_se, base_timeouts))

    # Model-estimation rollouts are truncated relative to how long the baseline
    # actually takes, so one bad sweep cannot make every later sweep crawl.
    rollout_cap = max(100, int(40 * base_time))

    best_policy, best_time = policy.copy(), base_time
    history = [{"iteration": 0, "delivery_time": base_time, "stderr": base_se,
                "changed": 0}]

    for it in range(1, outer_iters + 1):
        started = time.time()
        changed_total = 0
        for node in range(1, par["n"] + 1):
            counts, _ = collect_transitions(policy, par, node, episodes, epsilon,
                                            rng, max_steps=rollout_cap)
            table, changed = best_response(policy, par, node, counts, min_visits)
            policy.tables[node] = table
            changed_total += changed
        mean, sd, timeouts = evaluate(policy, par, eval_episodes, rng)
        se = sd / (eval_episodes ** 0.5)
        history.append({"iteration": it, "delivery_time": mean, "stderr": se,
                        "changed": changed_total, "timeouts": timeouts})
        if progress:
            print("sweep %2d: E[T] = %.4f +/- %.4f  (%d entries changed, %d timeouts, %.1fs)"
                  % (it, mean, se, changed_total, timeouts, time.time() - started))
        if mean < best_time:
            best_policy, best_time = policy.copy(), mean
        if changed_total == 0:
            if progress:
                print("converged: no node wants to deviate")
            break

    # fresh batch, so this number is not the one the selection was made on
    final, final_sd, _ = evaluate(best_policy, par, 4 * eval_episodes, rng)
    final_se = final_sd / ((4 * eval_episodes) ** 0.5)
    history.append({"iteration": "final", "delivery_time": final, "stderr": final_se,
                    "baseline": base_time, "baseline_stderr": base_se})
    if progress:
        gap = base_time - final
        gap_se = (base_se ** 2 + final_se ** 2) ** 0.5
        print("solved:  E[T] = %.4f +/- %.4f   (re-evaluated on a fresh batch)" % (final, final_se))
        print("baseline E[T] = %.4f +/- %.4f" % (base_time, base_se))
        print("improvement %.4f +/- %.4f  (%.1f%%, %.1f sigma)"
              % (gap, gap_se, 100.0 * gap / base_time, gap / gap_se if gap_se else 0.0))
    return best_policy, history


# --------------------------------------------------------------------------- #
# -----------------------------  EXPORT / CLI  ------------------------------- #
# --------------------------------------------------------------------------- #

def policy_filename(par, fallback="swap-asap"):
    """Cache path. The seed is part of the identity of a solved policy -- two
    JESP runs from different starting points are different policies -- but the
    default keeps its historical name so already-solved sweeps stay valid."""
    suffix = "" if fallback == "swap-asap" else "_from-" + fallback
    return DATA_DIR / ("infodelay_n%d_p%.3f_ps%.3f_tcut%d%s.json"
                       % (par["n"], par["p"], par["p_s"], par["t_cut"], suffix))


def export_policy(policy, par, path):
    """Write the per-node tables for info_delay_setup.jl's `TabularPolicy`.

    Only disagreements with SWAP-ASAP are stored; the Julia loader applies the
    same fallback, so an empty table reproduces the baseline exactly."""
    nodes = []
    for node in range(1, par["n"] + 1):
        actions = action_space(node, par["n"])
        entries = {}
        for obs, a in sorted(policy.tables[node].items()):
            swap, discard = actions[a]
            entries[obs_key(obs)] = {"swap": swap, "discard": list(discard)}
        nodes.append(entries)
    payload = {
        "n": par["n"], "p": par["p"], "p_s": par["p_s"], "t_cut": par["t_cut"],
        "s_max": S_MAX, "fallback": policy.fallback, "nodes": nodes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)
    return path


def run_in_simulator(policy_path, par, trials, tau, F_new, margin=0):
    """Hand the exported table to info_delay_setup.jl and get back the delivery
    time and the end-to-end fidelity, which this solver does not model."""
    script = (
        'include("%s");'
        'par = Params(; n=%d, p=%r, p_s=%r, τ=%r, F_new=%r, t_cut=%d);'
        'for (nm, pol) in (("swap-asap", SwapASAP()),'
        '                  ("cutoff-aware", CutoffAwareSwap(%d)),'
        '                  ("solved", TabularPolicy("%s")));'
        '  r = run_trials(pol; trials=%d, par, seed=20240820);'
        '  println(nm, "\\t", r.avg_delivery_time, "\\t", r.avg_fidelity, "\\t", r.timeouts);'
        'end'
    ) % (SIMULATOR_JL, par["n"], par["p"], par["p_s"], tau, F_new, par["t_cut"],
         margin, policy_path, trials)
    out = subprocess.run(["julia", "--project=" + str(REPO_ROOT), "-e", script],
                         check=True, capture_output=True, text=True)
    results = {}
    for line in out.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) == 4:
            results[parts[0]] = {"delivery_time": float(parts[1]),
                                 "fidelity": float(parts[2]),
                                 "timeouts": int(parts[3])}
    return results


def solve_and_export(par, force=False, fallback="swap-asap", **solve_kwargs):
    """Solve `par` and write its policy table, or reuse the one already on disk.

    Mirrors policy.py's check_*_data / load_*_data caching: the filename is a
    pure function of the parameters that actually change the policy (n, p, p_s,
    t_cut -- tau and F_new do not, since the solver never models the physics), so
    a grid sweep that is re-run only pays for the points it has not solved yet.
    Returns (path, history), with history None when the policy was reused."""
    path = policy_filename(par, fallback)
    if path.exists() and not force:
        return path, None
    policy, history = solve(par, fallback=fallback, **solve_kwargs)
    export_policy(policy, par, path)
    return path, history


# Julia driver for a whole grid in one process. QuantumSavory takes ~20s to
# precompile and load, so paying that per grid point would cost more than every
# simulation in the sweep put together.
_GRID_DRIVER = r"""
include(ARGS[1])
using JSON3
spec = JSON3.read(read(ARGS[2], String))
results = []
for pt in spec.points
    par = Params(; n=pt.n, p=pt.p, p_s=pt.p_s, t_cut=pt.t_cut,
                   tau_placeholder..., F_new=spec.F_new)
    policies = Any[("swap-asap", SwapASAP()), ("cutoff-aware", CutoffAwareSwap(spec.margin))]
    for (nm, pth) in pairs(pt.paths)
        push!(policies, (String(nm), TabularPolicy(String(pth))))
    end
    row = Dict{String,Any}("n"=>pt.n, "p"=>pt.p, "p_s"=>pt.p_s, "t_cut"=>pt.t_cut)
    for (nm, pol) in policies
        r = run_trials(pol; trials=spec.trials, par, seed=spec.seed)
        row[nm] = Dict("delivery_time"=>r.avg_delivery_time,
                       "delivery_sd"=>isempty(r.delivery_times) ? NaN : std(r.delivery_times),
                       "fidelity"=>r.avg_fidelity,
                       "delivered"=>r.delivered, "timeouts"=>r.timeouts)
    end
    push!(results, row)
    println("done n=", pt.n, " p=", pt.p, " t_cut=", pt.t_cut)
end
open(io -> JSON3.write(io, results), ARGS[3], "w")
"""


def run_grid_in_simulator(points, trials=4000, tau=50.0, F_new=1.0, seed=20240820,
                          margin=0, progress=True):
    """Measure every grid point in a single Julia process.

    `points` is a list of dicts with n, p, p_s, t_cut and `paths`, a
    {label: policy-table path} mapping (empty to measure only the two built-in
    baselines). Every policy at a point runs under the *same* seed, so they share
    their random draws and the comparison is paired.

    Note tau only moves fidelity, never delivery time -- the classical layer
    never reads it -- so a sweep can pick tau purely to make the fidelity panel
    legible without disturbing the timing panel."""
    spec_path = DATA_DIR / "_grid_spec.json"
    out_path = DATA_DIR / "_grid_results.json"
    driver_path = DATA_DIR / "_grid_driver.jl"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(spec_path, "w") as handle:
        json.dump({"trials": trials, "F_new": F_new, "seed": seed, "margin": margin,
                   "points": [{"n": pt["n"], "p": pt["p"], "p_s": pt["p_s"],
                               "t_cut": pt["t_cut"],
                               "paths": {k: str(v) for k, v in pt.get("paths", {}).items()}}
                              for pt in points]}, handle)
    with open(driver_path, "w") as handle:
        handle.write(_GRID_DRIVER.replace("tau_placeholder...", "τ=%r" % tau))

    proc = subprocess.run(
        ["julia", "--project=" + str(REPO_ROOT), str(driver_path),
         str(SIMULATOR_JL), str(spec_path), str(out_path)],
        check=True, capture_output=True, text=True)
    if progress and proc.stdout.strip():
        print(proc.stdout.strip().splitlines()[-1])
    with open(out_path) as handle:
        return json.load(handle)


def verify_port(par, episodes, seed=0):
    """Cross-check the Python dynamics against info_delay_setup.jl by running
    SWAP-ASAP through both and comparing delivery times. The two are independent
    implementations of the same model, so agreement to within MC noise is the
    evidence that the policy this solver produces was optimized against the
    simulator it will be run on."""
    rng = random.Random(seed)
    mine, sd_mine, timeouts = evaluate(JointPolicy(par["n"]), par, episodes, rng)
    script = (
        'include("%s");'
        'using Statistics;'
        'par = Params(; n=%d, p=%r, p_s=%r, τ=1000.0, t_cut=%d);'
        'r = run_trials(SwapASAP(); trials=%d, par, seed=99);'
        'println(r.avg_delivery_time, "\\t", std(r.delivery_times), "\\t", r.timeouts)'
    ) % (SIMULATOR_JL, par["n"], par["p"], par["p_s"], par["t_cut"], episodes)
    out = subprocess.run(["julia", "--project=" + str(REPO_ROOT), "-e", script],
                         check=True, capture_output=True, text=True)
    theirs, sd_theirs, jl_timeouts = out.stdout.strip().split("\t")
    theirs, sd_theirs = float(theirs), float(sd_theirs)

    # Two-sample z on the difference of means. Scaling by the observed spread
    # rather than by the mean matters a lot here: the tail makes sd several times
    # the mean, so a mean-scaled band flags honest agreement as a mismatch.
    delta = abs(mine - theirs)
    stderr = (sd_mine ** 2 / episodes + sd_theirs ** 2 / episodes) ** 0.5
    z = delta / stderr if stderr > 0 else 0.0
    print("swap-asap E[T]: python %.4f (sd %.3f, %d timeouts) | "
          "julia %.4f (sd %.3f, %s timeouts)"
          % (mine, sd_mine, timeouts, theirs, sd_theirs, jl_timeouts))
    print("difference %.4f, combined s.e. %.4f -> z = %.2f -> %s"
          % (delta, stderr, z, "OK" if z <= 4.0 else "MISMATCH"))
    return z <= 4.0


def main():
    parser = argparse.ArgumentParser(
        description="Decentralized (per-node) policy iteration for the "
                    "delayed-information repeater chain.")
    parser.add_argument("--n", type=int, default=5, help="Number of nodes.")
    parser.add_argument("--p", type=float, default=0.9,
                        help="Elementary link generation success probability.")
    parser.add_argument("--p_s", type=float, default=0.9,
                        help="Entanglement swap success probability.")
    parser.add_argument("--t-cut", type=int, default=6,
                        help="Qubit retention time, in timesteps.")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Give up on an episode after this many timesteps.")
    parser.add_argument("--episodes", type=int, default=4000,
                        help="Rollout episodes per single-node best response.")
    parser.add_argument("--eval-episodes", type=int, default=4000,
                        help="Episodes used for the greedy evaluation after each sweep.")
    parser.add_argument("--iters", type=int, default=12,
                        help="Maximum JESP sweeps over the nodes.")
    parser.add_argument("--inner-iters", type=int, default=50,
                        help="Maximum policy-iteration steps per best response.")
    parser.add_argument("--epsilon", type=float, default=0.15,
                        help="Exploration rate for the node being optimized.")
    parser.add_argument("--min-visits", type=int, default=25,
                        help="Samples an (observation, action) pair needs before "
                             "the improvement step is allowed to choose it.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--simulate", action="store_true",
                        help="Run the solved policy through info_delay_setup.jl "
                             "and report delivery time and fidelity.")
    parser.add_argument("--trials", type=int, default=2000,
                        help="Trials for --simulate.")
    parser.add_argument("--tau", type=float, default=1000.0,
                        help="Memory depolarization time constant, for --simulate.")
    parser.add_argument("--F_new", type=float, default=1.0,
                        help="Fidelity of a fresh elementary link, for --simulate.")
    parser.add_argument("--verify", action="store_true",
                        help="Cross-check the Python dynamics against the Julia "
                             "simulator and exit.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    par = {"n": args.n, "p": args.p, "p_s": args.p_s, "t_cut": args.t_cut,
           "max_steps": args.max_steps, "inner_iters": args.inner_iters}

    if args.verify:
        raise SystemExit(0 if verify_port(par, args.eval_episodes, args.seed) else 1)

    policy, history = solve(par, episodes=args.episodes, outer_iters=args.iters,
                            epsilon=args.epsilon, min_visits=args.min_visits,
                            eval_episodes=args.eval_episodes, seed=args.seed,
                            progress=not args.quiet)
    path = export_policy(policy, par, policy_filename(par))
    entries = sum(len(t) for t in policy.tables[1:])
    print("wrote %s (%d observations where the policy leaves swap-asap)"
          % (path, entries))

    with open(str(path)[:-5] + "_history.json", "w") as handle:
        json.dump(history, handle, indent=1)

    if args.simulate:
        results = run_in_simulator(path, par, args.trials, args.tau, args.F_new)
        print("\n%-14s %12s %12s" % ("policy", "E[T]", "fidelity"))
        for name in ("swap-asap", "cutoff-aware", "solved"):
            if name in results:
                r = results[name]
                print("%-14s %12.4f %12.6f" % (name, r["delivery_time"], r["fidelity"]))


if __name__ == "__main__":
    main()
