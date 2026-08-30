"""Policy iteration for the qubit-age cutoff model defined in environment.py.

The algorithm mirrors the policy iteration used in Iñesta et al.,
"On the Effect of Quantum Memory Cutoffs..." (arXiv:2207.06533) and its
reference implementation (github.com/AlvaroGI/optimal-homogeneous-chain),
but is solved directly against our qubit-centric State/Configuration model
instead of the paper's link-age model. This lets the two cutoff conventions
be compared under identical topology/probability dynamics.

Saved data uses the same on-disk shape as the paper's data_policyiter/...
pickle files (a dict with 'v0_evol', 'state_info', 'exe_time', where each
state_info entry holds 'state', 'action_space', 'policy', 'value'), so the
paper's analysis/plotting helpers (classify_states_policyiter,
gatherdata_optimal_vs_swapasap, ...) can load this data directly. The one
caveat is 'state': under the qubit-age model the two qubits of a virtual
link can carry different ages, so the exported matrix is not symmetric like
the paper's (cell [i][j] holds node i's own qubit age in its link with node
j). Everything the paper's code checks structurally (link existence,
end-to-end connectivity, valid swap nodes) is unaffected by this, since
those checks only look at which cells are finite, never at which specific
age value is stored.
"""
from functools import lru_cache
from pathlib import Path
import argparse
import os
import pickle
import random
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from environment import Configuration, Qubit, State

# save_policyiter_data / save_swapasap_data (and their check_*/load_* companions)
# use paths relative to 'data_policyiter/' and 'data_swapasap/', which they
# resolve against the current process's CWD. Anchoring CWD to this module's own
# directory here means those always land under src/, regardless of where this
# module is run or imported from (matches distill.py / simulate.py).
os.chdir(Path(__file__).resolve().parent)


# --------------------------------------------------------------------------- #
# ------------------------  STATE <-> HASHABLE FORM  ------------------------ #
# --------------------------------------------------------------------------- #

def qubits_to_key(qubits):
    """Hashable representation of a qubit list, suitable for use as a dict
    key. Carries (age, hanging) per qubit -- hanging must be part of the key
    since two qubits with the same age can behave differently (a hanging
    qubit is excluded from links_of), so collapsing it out would silently
    alias distinct states."""
    return tuple((q.age, q.hanging) for q in qubits)


def key_to_qubits(key):
    return [Qubit(age, hanging) for age, hanging in key]


def _discard_key(allow_discard):
    """Collision-safe cache-key stand-in for allow_discard. Python's
    True == 1 and False == 0 (and they hash equal), so passing allow_discard
    straight into an lru_cache-decorated function's argument list would let
    an uncapped-discard entry (True) silently answer a cap-of-1 lookup, and a
    discard-disabled entry (False) silently answer a cap-of-0 lookup, on the
    same (state, ...) key -- even though they mean different action spaces.
    Tagging with the value's own type breaks the alias."""
    return (type(allow_discard).__name__, allow_discard)


# --------------------------------------------------------------------------- #
# ----------------------  PAPER-COMPATIBLE STATE EXPORT  --------------------- #
# --------------------------------------------------------------------------- #

def qubits_to_matrix(qubits):
    """Node-indexed matrix view of a qubit list, for compatibility with the
    paper's state-inspection helpers (Environment.check_e2e_link,
    Environment.generate_action_space, ...). Cell [i][j] holds the age of
    node i's own qubit in the link it shares with node j (np.inf if i and j
    share no link)."""
    n = (len(qubits) + 2) // 2
    matrix = np.full((n, n), np.inf)
    for link in State.links_of(qubits):
        a, b = sorted(link)
        left_node, right_node = a // 2, (b + 1) // 2
        matrix[left_node][right_node] = qubits[a].age
        matrix[right_node][left_node] = qubits[b].age
    return matrix


def action_to_nodes(action):
    """Convert a qubit-index action (as produced by State.get_actions) into
    the paper's node-index convention (the index of the node performing
    the swap)."""
    return [(qubit_index + 1) // 2 for qubit_index in action]


# --------------------------------------------------------------------------- #
# --------------------------  MDP TRANSITION STEP  --------------------------- #
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def _cached_swap_resolution(p_s, qubits_key, swap_nodes):
    """The probabilistic outcomes of attempting swap_nodes from qubits_key --
    success/failure per swap site -- ignoring discard entirely. Depends only
    on (p_s, qubits_key, swap_nodes), never on any discard_qubits paired with
    this swap, so every discard variant of the same swap_nodes (get_actions
    enumerates one action per (swap_nodes, discard_qubits) combination) reuses
    this instead of re-expanding the same 2^|swap_nodes| success/failure
    combinations from scratch -- the single biggest cost discard-enabled
    policy iteration otherwise pays over and over for no new information.
    Returns (next_keys, probs): hashable qubit-age keys and their
    probabilities, summing to 1."""
    parameters = Configuration(p_s=p_s)
    swap_states, swap_probs = State(key_to_qubits(qubits_key), parameters) \
        .get_entanglement_swap_transitions_for_action(list(swap_nodes))
    return tuple(qubits_to_key(s) for s in swap_states), tuple(swap_probs)


def _apply_discard_to_key(qubits_key, discard_qubits):
    """Deterministically apply a discard action to a post-swap qubit-age key,
    producing the resulting key. Unlike the swap phase this has no failure
    mode and costs only O(len(discard_qubits)), so it doesn't need its own
    cache -- it's applied fresh to each of _cached_swap_resolution's (cached,
    shared) branches. Mirrors the discard loop State.get_transitions_for_action
    used to run inline."""
    if not discard_qubits:
        return qubits_key
    qubits = key_to_qubits(qubits_key)
    for q in discard_qubits:
        partner = State.link_partner(qubits, q)
        qubits[q].clear()
        if partner is not None:
            qubits[partner].hanging = True
    return qubits_to_key(qubits)


@lru_cache(maxsize=None)
def _cached_post_swap_step(p, cutoff, discard_key, qubits_key):
    """Resolve the remainder of a time slot from a state immediately after
    this slot's swap+discard has already been applied: terminal check,
    cutoff, aging, entanglement generation. A pure function of (p, cutoff,
    allow_discard) and this post-action configuration alone -- never of which
    original (state, action) pair produced it -- so distinct (state, action)
    pairs that happen to collapse onto the same post-action configuration
    (routine once discard is enabled, since many different discard choices
    can strip a state down to the same remainder) share this instead of each
    separately re-running cutoff/aging/entanglement-generation. discard_key
    is allow_discard run through _discard_key (see there for why the raw
    bool/int can't be used directly as a cache-key argument).
    Returns (next_keys, probs, action_spaces): probs are local to this call
    (sum to 1) and still need scaling by the swap outcome's own probability
    by the caller."""
    parameters = Configuration(p=p, cut=cutoff, allow_discard=discard_key[1])
    post = State(key_to_qubits(qubits_key), parameters)

    no_op_action_space = [((), ())] if parameters.discard_enabled else [[]]

    if post.check_e2e_link():
        return [qubits_key], [1.0], [no_op_action_space]

    post.cutoff()
    post.age_qubits()
    gen_states, gen_probs = post.get_entanglement_generation_transitions()

    next_keys, probs, action_spaces = [], [], []
    for g_qubits, gp in zip(gen_states, gen_probs):
        if gp == 0.0:
            continue
        next_keys.append(qubits_to_key(g_qubits))
        probs.append(gp)
        action_spaces.append(State(g_qubits, parameters).get_actions())

    return next_keys, probs, action_spaces


def step(parameters, qubits, action):
    """Advance one full time slot from a decision state (the state just
    before an action is applied), given `action`. Composition matches the
    paper's Environment.step: apply the action (swap, and discard if
    parameters.discard_enabled), apply cutoffs, age the qubits, then attempt
    entanglement generation -- unless the action already produced an
    end-to-end link, in which case the chain is delivered and evolution
    stops there. Delegates to _cached_swap_resolution and
    _cached_post_swap_step, which split this computation at the one point
    where the two enclosing exponentials (swap success/failure combinations;
    entanglement-generation combinations) are otherwise independent of each
    other and of discard, so that work is shared across every (state,
    action) pair that passes through the same intermediate configuration
    instead of being redone per action.

    Returns (next_states, probabilities, action_spaces): one entry per
    possible outcome, each next_state a qubit list and each action_space the
    valid actions from that next state (a single no-op action for a
    delivered/terminal state, in whichever shape get_actions() uses for
    this Configuration).
    """
    qubits_key = qubits_to_key(qubits)
    if parameters.discard_enabled:
        swap_nodes, discard_qubits = action
    else:
        swap_nodes, discard_qubits = action, ()

    swap_keys, swap_probs = _cached_swap_resolution(parameters.p_s, qubits_key, tuple(swap_nodes))

    next_states, probs, action_spaces = [], [], []
    for s_key, p in zip(swap_keys, swap_probs):
        if p == 0.0:
            continue

        post_key = _apply_discard_to_key(s_key, discard_qubits)
        post_next_keys, post_probs, post_action_spaces = _cached_post_swap_step(
            parameters.p, parameters.cut, _discard_key(parameters.allow_discard), post_key)

        for n_key, gp, a_space in zip(post_next_keys, post_probs, post_action_spaces):
            next_states.append(key_to_qubits(n_key))
            probs.append(p * gp)
            action_spaces.append(a_space)

    return next_states, probs, action_spaces


@lru_cache(maxsize=None)
def _is_terminal(qubits_key):
    """Whether a state (given by its qubit-age key) already has an
    end-to-end link. Depends only on link structure, not on parameters, so
    it's cached independent of (p, p_s, cutoff)."""
    return State(key_to_qubits(qubits_key), None).check_e2e_link()


@lru_cache(maxsize=None)
def _cached_step(p, p_s, cutoff, discard_key, qubits_key, action):
    """Memoized transition lookup, keyed on hashable (state, action,
    parameters). The transition structure out of a given state under a given
    action never changes across policy-evaluation sweeps -- only the value
    estimates propagating through it do -- so recomputing it every sweep
    (deepcopy-ing qubit lists and re-expanding every success/failure
    combination) is pure waste. This turns that into "compute once per
    (state, action) the first time it's visited, then reuse." discard_key
    (allow_discard run through _discard_key) is part of the cache key -- not
    just baked into a shared Configuration -- because it changes both the
    action shape and the transition semantics for the same (p, p_s, cutoff);
    see _discard_key for why the raw bool/int can't be used directly here.
    Returns (next_keys, probs, action_spaces) where next_keys are qubit-age
    keys and action_spaces are tuples of tuples (both hashable, unlike the
    Qubit-list/list-of-lists shapes step() returns)."""
    parameters = Configuration(p=p, p_s=p_s, cut=cutoff, allow_discard=discard_key[1])
    next_states, probs, action_spaces = step(parameters, key_to_qubits(qubits_key), list(action))
    next_keys = tuple(qubits_to_key(s) for s in next_states)
    action_spaces = tuple(tuple(tuple(a) for a in a_space) for a_space in action_spaces)
    return next_keys, tuple(probs), action_spaces


def cached_step(parameters, qubits_key, action):
    """Convenience wrapper around _cached_step taking a Configuration and a
    qubit-age key directly."""
    return _cached_step(parameters.p, parameters.p_s, parameters.cut,
                         _discard_key(parameters.allow_discard), qubits_key, tuple(action))


# --------------------------------------------------------------------------- #
# ------------------  RELAXED (PERFECT-MEMORY) LOWER BOUND  ------------------ #
# --------------------------------------------------------------------------- #
#
# Used by bounded_rtdp_policy (near the bottom of this file) as an admissible
# heuristic: a state's true optimal cost-to-go (expected remaining time
# slots) under the qubit-age cutoff model can never be less than its optimal
# cost-to-go in a relaxed version of the same chain with decoherence switched
# off entirely -- no cutoff, so no reason to ever discard. Every trajectory
# available to the real chain is also available to the relaxed one (dropping
# age/hanging never invalidates a transition, it only removes a way for the
# real chain to get worse), so the relaxed optimum is a valid lower bound on
# the real one. Because the relaxed model never ages a qubit past 0, its
# reachable state space collapses to a small occupied/empty pattern with no
# age dimension at all -- cheap to solve exactly once per (n, p, p_s) and
# reuse as a lookup table (the pattern-database idea) for every cutoff and
# discard-cap setting of the real problem.

def _relaxed_step(p, p_s, qubits_key, action):
    """One time slot of the relaxed (no-decoherence, no-discard) chain: same
    swap/generation dynamics as step(), but never calls cutoff() or
    age_qubits(), so every occupied qubit stays at age 0 forever -- which is
    what keeps the reachable state space small. Reuses _cached_swap_resolution
    for the swap phase unchanged: that computation depends only on p_s, never
    on cutoff/discard/aging, so it's exactly the same function the real model
    uses. `action` is a plain swap-node list (the relaxed model has no
    discard action, so there's no second half to unpack).
    Returns (next_keys, probs, action_spaces) in the same (age, hanging) key
    encoding as the real model -- ages just never exceed 0 here."""
    swap_keys, swap_probs = _cached_swap_resolution(p_s, qubits_key, tuple(action))

    next_keys, probs, action_spaces = [], [], []
    for s_key, prob in zip(swap_keys, swap_probs):
        if prob == 0.0:
            continue
        post_next_keys, post_probs, post_action_spaces = _cached_relaxed_post_swap_step(p, s_key)
        for n_key, gp, a_space in zip(post_next_keys, post_probs, post_action_spaces):
            next_keys.append(n_key)
            probs.append(prob * gp)
            action_spaces.append(a_space)

    return next_keys, probs, action_spaces


@lru_cache(maxsize=None)
def _cached_relaxed_post_swap_step(p, qubits_key):
    """Relaxed analogue of _cached_post_swap_step: terminal check, then
    (skipping cutoff/aging entirely) entanglement generation. Discard is
    never enabled in the relaxed model, so action spaces here are always the
    plain swap-only shape ([[]] for the terminal no-op)."""
    parameters = Configuration(p=p, allow_discard=False)
    post = State(key_to_qubits(qubits_key), parameters)

    if post.check_e2e_link():
        return [qubits_key], [1.0], [[[]]]

    gen_states, gen_probs = post.get_entanglement_generation_transitions()

    next_keys, probs, action_spaces = [], [], []
    for g_qubits, gp in zip(gen_states, gen_probs):
        if gp == 0.0:
            continue
        next_keys.append(qubits_to_key(g_qubits))
        probs.append(gp)
        action_spaces.append(State(g_qubits, parameters).get_actions())

    return next_keys, probs, action_spaces


def _project_to_relaxed_key(qubits_key):
    """Map a real (age, hanging) qubit-age key onto its relaxed counterpart:
    any qubit currently holding a useful entangled link (age != -1, not
    hanging) becomes occupied-at-age-0; every other qubit (empty, or hanging
    -- occupied but excluded from every link) becomes empty. Collapsing a
    hanging qubit to empty rather than occupied is what keeps this an
    admissible relaxation: in the real chain that slot is stuck until cutoff
    or discard frees it, so treating it as already-free in the relaxed model
    can only make the relaxed problem easier, never harder -- exactly what a
    valid lower bound requires."""
    return tuple((0, False) if (age != -1 and not hanging) else (-1, False)
                 for age, hanging in qubits_key)


@lru_cache(maxsize=None)
def _relaxed_optimal_cost_table(n, p, p_s):
    """Exact optimal expected-remaining-time for every state the relaxed
    (no-decoherence, no-discard) n-node chain can reach from empty. Solved
    once via plain exact policy iteration (direct/sparse-LU evaluation,
    since the state space here is small enough that iterative sweeps would
    just be overhead) over this small state space; memoized so repeated
    calls for different cutoffs/discard caps at the same (n, p, p_s) all
    reuse the same solve. Deliberately self-contained rather than reusing
    policy_iteration/_evaluate_policy_direct -- those are wired directly to
    the real model's module-level cached_step, and this model's whole point
    is to be a small, independent, exactly-solved system.
    Returns {qubits_key: expected_remaining_time} (a plain positive cost,
    the convention the rest of bounded_rtdp_policy works in -- NOT this
    module's usual value = -(cost) convention)."""
    parameters = Configuration(p=p, p_s=p_s, allow_discard=False)
    root_key = qubits_to_key(State.initial(n, parameters).qubits)

    state_list = []
    index = {}
    action_space = {}   # key -> list of hashable (tuple) actions
    policy = {}          # key -> probability vector over action_space[key]
    value = {}

    def observe(key):
        idx = index.get(key)
        if idx is None:
            idx = len(state_list)
            index[key] = idx
            state_list.append(key)
            value[key] = 0.0
            if not _is_terminal(key):
                actions = [tuple(a) for a in State(key_to_qubits(key), parameters).get_actions()]
                action_space[key] = actions
                # Uniform-random initial policy, matching Agent.init_policy in
                # the real model: a state whose ONLY currently-known action is
                # a dead self-loop (e.g. "don't swap" at a state where both
                # segments are already established and nothing else can
                # happen) would make (I - P_pi) exactly singular if evaluated
                # as a one-hot pick before the first improvement pass ever
                # runs -- spreading probability over every action guarantees
                # some escape probability out of every non-terminal state.
                policy[key] = [1.0 / len(actions)] * len(actions)
        return idx

    observe(root_key)

    while True:
        ### Exact evaluation of the current (fixed) policy, direct solve ###
        coeffs = {}
        idx = 0
        while idx < len(state_list):
            key = state_list[idx]
            if _is_terminal(key):
                idx += 1
                continue
            for action, action_prob in zip(action_space[key], policy[key]):
                if action_prob == 0.0:
                    continue
                next_keys, probs, _ = _relaxed_step(p, p_s, key, action)
                for s_key, P in zip(next_keys, probs):
                    if P == 0.0:
                        continue
                    s_idx = observe(s_key)
                    if not _is_terminal(s_key):
                        coeff_key = (idx, s_idx)
                        coeffs[coeff_key] = coeffs.get(coeff_key, 0.0) - action_prob * P
            idx += 1

        non_terminal = [i for i in range(len(state_list)) if not _is_terminal(state_list[i])]
        row_of = {a: r for r, a in enumerate(non_terminal)}
        size = len(non_terminal)

        rows, cols, data = [], [], []
        for (r_idx, c_idx), coeff in coeffs.items():
            rows.append(row_of[r_idx])
            cols.append(row_of[c_idx])
            data.append(coeff)
        for a_idx in non_terminal:
            rows.append(row_of[a_idx])
            cols.append(row_of[a_idx])
            data.append(1.0)

        A = coo_matrix((data, (rows, cols)), shape=(size, size)).tocsc()
        b = -np.ones(size)
        V = spsolve(A, b)
        for row, a_idx in enumerate(non_terminal):
            value[state_list[a_idx]] = float(V[row])

        ### Policy improvement ###
        policy_changed = False
        idx = 0
        while idx < len(state_list):
            key = state_list[idx]
            if _is_terminal(key) or len(action_space[key]) == 1:
                idx += 1
                continue
            actions = action_space[key]
            q_values = [0.0] * len(actions)
            for k, action in enumerate(actions):
                next_keys, probs, _ = _relaxed_step(p, p_s, key, action)
                for s_key, P in zip(next_keys, probs):
                    if P == 0.0:
                        continue
                    s_idx = observe(s_key)
                    q_values[k] += P * (-1 + value[state_list[s_idx]])
            best = int(np.argmax(q_values))
            if policy[key][best] < 1.0:
                policy_changed = True
                new_policy = [0.0] * len(actions)
                new_policy[best] = 1.0
                policy[key] = new_policy
            idx += 1

        if not policy_changed:
            break

    # cost := -value (NOT this module's usual -(value+1) "expected delivery
    # time" reporting convention -- that convention shifts by 1 relative to
    # the plain recursion cost(s) = 1 + sum_nonterm P*cost(s'), cost(terminal)
    # = 0 that _brtdp_backup actually runs, since the "+1" per decision in
    # value's own recursion (V(s) = -1 + sum P V(s')) already accounts for
    # every action taken including whichever one reaches a terminal state --
    # value's own boundary condition V(terminal)=0 already prices that in,
    # so translating through -(value+1) instead of -value would silently
    # undercount by exactly one slot everywhere).
    return {key: -value[key] for key in state_list}


def _relaxed_lower_bound(n, p, p_s, qubits_key):
    """Admissible lower bound on the expected remaining time slots to
    delivery from qubits_key under the real (cutoff, possibly discard-
    enabled) model: the relaxed chain's own exact optimal cost from the
    projected state (see _project_to_relaxed_key and this section's module
    docstring for why the projection is safe)."""
    table = _relaxed_optimal_cost_table(n, p, p_s)
    relaxed_key = _project_to_relaxed_key(qubits_key)
    return table[relaxed_key]


# --------------------------------------------------------------------------- #
# ------------------------------  AGENT  ------------------------------------ #
# --------------------------------------------------------------------------- #

class Agent:
    """Bookkeeping for policy iteration: every distinct state discovered so
    far, its valid action space, its current policy (one-hot over
    action_space once converged), and its current value estimate. The
    expected number of time slots to delivery from a state is
    -(value + 1) -- matching the paper's reference implementation
    (AlvaroGI/optimal-homogeneous-chain), whose own Monte Carlo validation
    (main.py's simulate_environment) reports delivery time as a 0-indexed
    count of transitions (T_vec.append(time-1)), i.e. delivering on the very
    first attempt is time slot 0, not 1."""

    def __init__(self, n, parameters):
        s0 = qubits_to_key(State.initial(n, parameters).qubits)
        self._index = {s0: 0}
        self.state_list = [s0]
        no_op_action_space = [((), ())] if parameters.discard_enabled else [[]]
        self.state_info = [{
            "state": s0,
            "action_space": no_op_action_space,
            "policy": [1.0],
            "value": 0.0,
        }]

    def get(self, idx, key):
        return self.state_info[idx][key]

    def update(self, idx, value, key):
        self.state_info[idx][key] = value

    @staticmethod
    def init_policy(action_space):
        """Initial policy is a uniform-random choice among valid actions, so
        that policy evaluation explores every reachable transition."""
        if len(action_space) < 2:
            return [1.0]
        return [1.0 / len(action_space)] * len(action_space)

    def observe(self, key, action_space=None):
        """Look up the index of a state, registering it if it's new."""
        idx = self._index.get(key)
        if idx is None:
            assert action_space is not None
            idx = len(self.state_list)
            self._index[key] = idx
            self.state_list.append(key)
            self.state_info.append({
                "state": key,
                "action_space": action_space,
                "policy": self.init_policy(action_space),
                "value": 0.0,
            })
        return idx


# --------------------------------------------------------------------------- #
# ---------------------------  POLICY EVALUATION  ----------------------------- #
# --------------------------------------------------------------------------- #

def _evaluate_policy_iterative(agent, parameters, tol, progress):
    """Evaluate the current policy stored in `agent` by successive
    approximation: repeatedly apply the Bellman backup to every known state
    until the largest per-state change drops below `tol`. New states
    encountered along the way are registered via agent.observe() and picked
    up on a later pass of the same sweep loop, since it re-reads
    len(agent.state_list) every iteration.
    Returns v0: the evolution of the initial state's value across sweeps."""
    v0 = []
    error = np.inf
    while error > tol:
        error = 0.0
        idx = 0
        while idx < len(agent.state_list):
            key = agent.state_list[idx]
            if _is_terminal(key):
                idx += 1
                continue

            value = agent.get(idx, "value")
            policy = agent.get(idx, "policy")
            action_space = agent.get(idx, "action_space")

            v = 0.0
            for action, action_prob in zip(action_space, policy):
                if action_prob == 0.0:
                    continue
                next_keys, probs, next_action_spaces = cached_step(parameters, key, action)
                for s_key, P, a_space in zip(next_keys, probs, next_action_spaces):
                    if P == 0.0:
                        continue
                    s_idx = agent.observe(s_key, a_space)
                    v += action_prob * P * (-1 + agent.get(s_idx, "value"))

            error = max(error, abs(value - v))
            agent.update(idx, v, "value")
            idx += 1

        v0.append(agent.get(0, "value"))
        if progress:
            print("Policy eval.: error = %.2e > %.2e = tolerance" % (error, tol), end="\r")
    return v0


def _evaluate_policy_direct(agent, parameters):
    """Evaluate the current policy stored in `agent` exactly, by solving the
    linear system it defines rather than iterating a Bellman backup to
    convergence. For a fixed policy, V(s) = -1 + sum_a policy(s,a) sum_s'
    P(s'|s,a) V(s') is just |S| linear equations in |S| unknowns (terminal
    states are absorbing with V=0 by convention, so they contribute no
    column -- only the -1 reward -- and are never part of the unknown
    vector): (I - P_pi) V = -1. Solving it directly sidesteps the case that
    hurts _evaluate_policy_iterative worst -- a slow-mixing chain (e.g. low
    p) that needs a huge number of sweeps before the truncated Bellman sum
    has accounted for enough probability mass to satisfy an absolute
    tolerance. A direct solve's cost depends on the sparsity structure of
    the transition graph, not on how slowly that chain mixes.

    Walks agent.state_list exactly like _evaluate_policy_iterative (and so
    discovers new states via agent.observe() the same way), but instead of
    accumulating value updates it accumulates (row, col) -> coefficient
    entries for the sparse matrix, then solves once.
    Returns v0: [initial state's value] (a single exact value, not a
    per-sweep trace, since there's no sweep-by-sweep convergence here)."""
    coeffs = {}  # (row_agent_idx, col_agent_idx) -> accumulated coefficient

    idx = 0
    while idx < len(agent.state_list):
        key = agent.state_list[idx]
        if _is_terminal(key):
            idx += 1
            continue

        policy = agent.get(idx, "policy")
        action_space = agent.get(idx, "action_space")

        for action, action_prob in zip(action_space, policy):
            if action_prob == 0.0:
                continue
            next_keys, probs, next_action_spaces = cached_step(parameters, key, action)
            for s_key, P, a_space in zip(next_keys, probs, next_action_spaces):
                if P == 0.0:
                    continue
                s_idx = agent.observe(s_key, a_space)
                if not _is_terminal(s_key):
                    # Terminal successors contribute nothing (V=0 there);
                    # only non-terminal successors become matrix columns.
                    coeff_key = (idx, s_idx)
                    coeffs[coeff_key] = coeffs.get(coeff_key, 0.0) - action_prob * P

        idx += 1

    non_terminal_indices = [i for i in range(len(agent.state_list))
                             if not _is_terminal(agent.state_list[i])]
    row_of = {agent_idx: row for row, agent_idx in enumerate(non_terminal_indices)}
    size = len(non_terminal_indices)

    rows, cols, data = [], [], []
    for (r_idx, c_idx), coeff in coeffs.items():
        rows.append(row_of[r_idx])
        cols.append(row_of[c_idx])
        data.append(coeff)
    for agent_idx in non_terminal_indices:
        rows.append(row_of[agent_idx])
        cols.append(row_of[agent_idx])
        data.append(1.0)  # the "I" term; coo_matrix sums duplicate entries on conversion

    A = coo_matrix((data, (rows, cols)), shape=(size, size)).tocsc()
    b = -np.ones(size)
    V = spsolve(A, b)

    for row, agent_idx in enumerate(non_terminal_indices):
        agent.update(agent_idx, float(V[row]), "value")

    return [agent.get(0, "value")]


# --------------------------------------------------------------------------- #
# ---------------------------  POLICY ITERATION  ----------------------------- #
# --------------------------------------------------------------------------- #

def policy_iteration(n, p, p_s, cutoff, tolerance=1e-5, tolerance_stability=1e-1,
                      progress=True, savedata=True, allow_discard=False, method="iterative"):
    """Find the optimal swap policy for an n-node chain under the qubit-age
    cutoff model, minimizing the expected number of time slots to end-to-end
    entanglement.
        ---Inputs---
            · n:    (int) number of nodes in the chain.
            · p:    (float) probability of successful entanglement
                    generation between neighboring nodes.
            · p_s:  (float) probability of successful entanglement swap.
            · cutoff:   (int) qubit-age cutoff - qubits whose age reaches
                        the cutoff are discarded, along with their partner
                        (or on its own, if hanging -- see environment.py).
            · tolerance:    (float) the algorithm stops once EITHER the policy
                            is exactly stable for two consecutive outer
                            iterations, OR every state's value has changed by
                            less than this much since the last outer
                            iteration. The two can diverge when several
                            actions are (numerically) tied in Q-value at some
                            states: exact evaluation (method="direct") can
                            then cycle between equally-good actions forever
                            without ever registering as "exactly stable",
                            even though the value itself stopped moving after
                            the first few iterations -- the value-based check
                            exists to catch exactly that case.
            · tolerance_stability:  (float) looser tolerance used for policy
                                    evaluation sweeps until the policy is
                                    stable; speeds up early iterations.
            · progress: (bool) if True, prints progress.
            · savedata: (bool) if True, saves the outputs in a file.
            · allow_discard:    (bool or int) if False (default), actions are
                                swap-only flat lists, exactly as without this
                                feature. If True, the agent may additionally
                                discard any number of occupied qubits each
                                time slot (State.get_actions' (swap_nodes,
                                discard_qubits) action shape). If an int, discard
                                is enabled but capped at that many qubits per
                                time slot -- e.g. allow_discard=1 restricts to
                                at most one discard per time slot, which keeps
                                the discard dimension of the action space
                                linear rather than exponential in the number
                                of occupied qubits.
            · method:   (str) how to run each policy evaluation step:
                        "iterative" (default) uses successive approximation
                        (Bellman-backup sweeps to a tolerance), matching the
                        paper's approach; "direct" solves the linear system
                        the fixed policy defines exactly via sparse LU
                        (see _evaluate_policy_direct). Direct solving costs
                        don't depend on how slowly the chain mixes, so it is
                        especially advantageous at low p, where iterative
                        evaluation can need very many sweeps to converge.
        ---Outputs---
            · v0_evol:  (list of lists) each list contains the evolution of
                        the value of the empty initial state over one policy
                        evaluation step. The final value is v0_evol[-1][-1].
            · state_info: (list of dicts) each dictionary corresponds to a
                          distinct state of the MDP, with keys 'state'
                          (qubit-age tuple), 'action_space', 'policy'
                          (one-hot once converged), and 'value' (expected
                          delivery time from this state is -(value+1), see
                          the Agent class docstring).
            · exe_time: (float) execution time of the algorithm in seconds."""
    assert isinstance(n, int) and n >= 2, "n must be an integer >= 2"
    assert 0 <= p <= 1, "p must be between zero and one"
    assert 0 <= p_s <= 1, "p_s must be between zero and one"
    assert isinstance(cutoff, int) and cutoff > 0, "cutoff must be a positive integer"
    assert method in ("iterative", "direct"), 'method must be "iterative" or "direct"'

    parameters = Configuration(p=p, p_s=p_s, cut=cutoff, allow_discard=allow_discard)
    agent = Agent(n, parameters)

    start_time = time.time()
    v0_evol = []
    policy_is_stable = False
    policy_was_stable = False
    tol = tolerance_stability

    while True:
        prev_values = [agent.get(i, "value") for i in range(len(agent.state_list))]
        prev_len = len(prev_values)

        ### Policy evaluation step ###
        if method == "direct":
            v0 = _evaluate_policy_direct(agent, parameters)
        else:
            v0 = _evaluate_policy_iterative(agent, parameters, tol, progress)
        v0_evol.append(v0)

        # How much every state's value moved since the last outer iteration
        # (states discovered just now compare against their creation default
        # of 0.0, i.e. as if they'd already existed at that value).
        value_delta = 0.0
        for i in range(len(agent.state_list)):
            before = prev_values[i] if i < prev_len else 0.0
            value_delta = max(value_delta, abs(agent.get(i, "value") - before))

        ### Stop if the policy or its value has converged ###
        if (policy_is_stable and policy_was_stable) or value_delta < tolerance:
            break

        ### Policy improvement step ###
        policy_was_stable = policy_is_stable
        policy_is_stable = True
        tol = tolerance
        idx = 0
        while idx < len(agent.state_list):
            key = agent.state_list[idx]
            action_space = agent.get(idx, "action_space")
            policy = agent.get(idx, "policy")

            if len(action_space) == 1 or _is_terminal(key):
                idx += 1
                continue

            q_values = [0.0] * len(action_space)
            for k, action in enumerate(action_space):
                next_keys, probs, _ = cached_step(parameters, key, action)
                for s_key, P in zip(next_keys, probs):
                    if P == 0.0:
                        continue
                    s_idx = agent.observe(s_key)
                    q_values[k] += P * (-1 + agent.get(s_idx, "value"))

            best_action = int(np.argmax(q_values))
            if policy[best_action] < 1.0:
                policy_is_stable = False
                tol = tolerance_stability
                new_policy = [0.0] * len(action_space)
                new_policy[best_action] = 1.0
                agent.update(idx, new_policy, "policy")

            idx += 1
            if progress:
                print("Policy improvement: state %d/%d" % (idx, len(agent.state_list)) + " " * 40, end="\r")

    if progress:
        print(" " * 80, end="\r")

    exe_time = time.time() - start_time

    if savedata:
        save_policyiter_data(n, p, p_s, cutoff, tolerance, v0_evol, agent.state_info, exe_time,
                              allow_discard=allow_discard)

    return v0_evol, agent.state_info, exe_time


# --------------------------------------------------------------------------- #
# -------------------------  SWAP-ASAP BASELINE  ----------------------------- #
# --------------------------------------------------------------------------- #

def policy_eval_swapasap(n, p, p_s, cutoff, tolerance=1e-5, progress=True, savedata=False,
                          allow_discard=False):
    """Evaluate the swap-asap policy under the qubit-age cutoff model, as a
    baseline against the optimal policy found by policy_iteration. Mirrors
    policy_eval_swapasap in the paper's reference implementation
    (AlvaroGI/optimal-homogeneous-chain/main.py): this is pure policy
    evaluation, with no policy-improvement step, since the policy is fixed
    to "perform every valid swap, every time slot" throughout. Swap-asap is
    always the *last* entry of action_space, since State.get_actions
    generates actions by ascending combination size, so the full-combination
    (swap everything) action is always generated last -- when discard is
    enabled (with or without a cap) this is still the last entry (never
    discarding anything).
        ---Inputs---
            · n, p, p_s, cutoff, allow_discard: see policy_iteration.
            · tolerance:    (float) the algorithm stops when the maximum
                            change in value between sweeps is smaller than
                            this tolerance.
            · progress: (bool) if True, prints progress.
            · savedata: (bool) if True, saves the outputs in a file.
        ---Outputs---
            · v0_evol, state_info, exe_time: see policy_iteration. Here
              'policy' in each state_info entry is fixed to swap-asap rather
              than optimized."""
    assert isinstance(n, int) and n >= 2, "n must be an integer >= 2"
    assert 0 <= p <= 1, "p must be between zero and one"
    assert 0 <= p_s <= 1, "p_s must be between zero and one"
    assert isinstance(cutoff, int) and cutoff > 0, "cutoff must be a positive integer"

    parameters = Configuration(p=p, p_s=p_s, cut=cutoff, allow_discard=allow_discard)
    agent = Agent(n, parameters)

    start_time = time.time()
    v0 = []
    error = np.inf
    while error > tolerance:
        error = 0.0
        idx = 0
        while idx < len(agent.state_list):
            key = agent.state_list[idx]
            if _is_terminal(key):
                idx += 1
                continue

            value = agent.get(idx, "value")
            action_space = agent.get(idx, "action_space")

            # Swap-asap: always perform every valid swap.
            policy_swapasap = [0.0] * len(action_space)
            policy_swapasap[-1] = 1.0
            agent.update(idx, policy_swapasap, "policy")

            v = 0.0
            for action, action_prob in zip(action_space, policy_swapasap):
                if action_prob == 0.0:
                    continue
                next_keys, probs, next_action_spaces = cached_step(parameters, key, action)
                for s_key, P, a_space in zip(next_keys, probs, next_action_spaces):
                    if P == 0.0:
                        continue
                    s_idx = agent.observe(s_key, a_space)
                    v += action_prob * P * (-1 + agent.get(s_idx, "value"))

            error = max(error, abs(value - v))
            agent.update(idx, v, "value")
            idx += 1

        v0.append(agent.get(0, "value"))
        if progress:
            print("Policy eval. (swap-asap): error = %.2e > %.2e = tolerance" % (
                error, tolerance), end="\r")

    if progress:
        print(" " * 80, end="\r")

    exe_time = time.time() - start_time
    v0_evol = [v0]

    if savedata:
        save_swapasap_data(n, p, p_s, cutoff, tolerance, v0_evol, agent.state_info, exe_time,
                            allow_discard=allow_discard)

    return v0_evol, agent.state_info, exe_time


# --------------------------------------------------------------------------- #
# --------------------------  BOUNDED RTDP  ----------------------------------- #
# --------------------------------------------------------------------------- #
#
# bounded_rtdp_policy (near the end of this section) finds a near-optimal
# swap+discard policy for the discard-enabled setting without ever sweeping
# the full reachable state space the way policy_iteration does: it runs many
# short trials from the initial state, at each visited state backing up a
# lower and an upper bound on that state's optimal expected-remaining-time
# using the real model's own cached_step, and steering each trial toward
# whichever successor currently has the most bound-gap uncertainty. Trials
# stop once the root's own [LB, UB] gap is inside `tolerance`. Everything in
# this section works in COST space (expected remaining time slots, a plain
# non-negative number to minimize) rather than this module's usual
# value = -(cost) convention, since RTDP/BRTDP is standardly stated as cost
# minimization and flipping signs throughout would only invite bugs; the
# conversion back to value happens once, at export time.

@lru_cache(maxsize=None)
def _swapasap_cost_table(n, p, p_s, cutoff):
    """Exact expected-remaining-time under the swap-asap policy for every
    state in swap-asap's own reachable closure from the empty state.

    Swap-asap is a valid policy from EVERY state of the discard-enabled
    problem too (it simply never discards -- the action "swap everything,
    discard nothing" exists in every discard-enabled action space, and is
    exactly what _swapasap_action picks), so V^swapasap(s) is a per-state
    upper bound on the optimal cost V*(s) wherever it's known. That makes
    this table a far tighter seed for BRTDP's upper bound than the single
    global constant _global_upper_bound falls back to -- and it costs
    nothing extra, since _global_upper_bound has to run this very evaluation
    anyway to get its own UB_empty.

    Evaluated with allow_discard=False deliberately: swap-asap never
    discards, so its trajectory distribution -- and therefore every state
    value in its closure -- is identical with or without discard enabled,
    while the no-discard action space is the cheaper one to enumerate.

    Uses cost := -value, NOT this module's usual -(value+1) "expected
    delivery time" reporting convention -- see _relaxed_optimal_cost_table's
    docstring for why that shifted convention would silently undercount by
    one slot relative to the plain recursion _brtdp_backup runs. The same
    small safety margin _global_upper_bound adds is applied per entry, for
    the same reason: policy_eval_swapasap's tolerance is tight but finite,
    and RTDP's admissibility guarantee depends on the seed genuinely being
    on the correct side.
    Returns (table, ub_empty): {qubits_key: cost} and the empty initial
    state's own entry."""
    _, state_info, _ = policy_eval_swapasap(n, p, p_s, cutoff, tolerance=1e-9,
                                             progress=False, savedata=False, allow_discard=False)
    table = {entry["state"]: -entry["value"] + 1e-6 for entry in state_info}
    return table, -state_info[0]["value"] + 1e-6


@lru_cache(maxsize=None)
def _global_upper_bound(n, p, p_s, cutoff, discard_key):
    """A cheap, valid upper bound on the optimal expected-remaining-time from
    EVERY state of the real (cutoff, discard-enabled) chain -- not just the
    initial one; the fallback BRTDP seeds any state _swapasap_cost_table
    doesn't cover with. Swap-asap is a valid (if generally suboptimal) policy
    for the real problem, so its own expected delivery time from the empty
    initial state, UB_empty, is by definition an upper bound on the
    *optimal* expected delivery time from the empty state (reuses
    policy_eval_swapasap rather than a hand-derived closed form, so it's
    exact for the real cutoff dynamics, not an approximation that could be
    invalidated by cutoff-driven resets).

    With UNCAPPED discard, UB_empty is valid from every state, because the
    empty state is provably the worst state to be in: at the empty state the
    only available action is the no-op (nothing to swap, nothing to discard),
    so cost*(empty) = 1 + sum_s' P(s'|empty, no-op) cost*(s'); and from ANY
    state s the action "swap nothing, discard everything" leaves exactly the
    empty configuration before cutoff/aging/generation run, so it induces
    that very same successor distribution. Hence cost*(s) <= 1 + sum_s'
    P(s'|empty, no-op) cost*(s') = cost*(empty) <= UB_empty.

    With a CAPPED discard (at most `cap` qubits/slot) that argument no longer
    closes: clearing an arbitrary state takes up to ceil((2n-2)/cap) slots of
    pure discarding (2n-2 being every qubit slot in the chain), and
    entanglement generation keeps refilling free pairs while that happens, so
    the empty state is not reached in one hop. Rather than assert a tighter
    claim than can be justified, this pads UB_empty by those ceil((2n-2)/cap)
    slots as a deliberately conservative cushion. Being loose here costs
    little in practice: the padded constant is only ever the FALLBACK seed,
    used for states outside swap-asap's own closure, while
    _swapasap_cost_table supplies an exact per-state upper bound for
    everything inside it -- which is most of what BRTDP's trials visit.
    discard_key is allow_discard run through _discard_key (see there for why
    the raw bool/int can't be used directly as a cache-key argument).

    Uses cost := -value here, NOT this module's usual -(value+1) "expected
    delivery time" reporting convention -- see _relaxed_optimal_cost_table's
    docstring for why that shifted convention would silently undercount by
    one slot relative to the plain recursion _brtdp_backup runs (this bit
    a first version of this function: seeding every upper bound one slot too
    low meant it could never be corrected back upward by _brtdp_backup's
    min()-only tightening, and every lower bound eventually grew past it).

    The small safety margin _swapasap_cost_table adds carries through here:
    policy_eval_swapasap's own tolerance is tight (1e-9) but still finite,
    and RTDP's admissibility guarantee (bounds only tighten monotonically
    toward the true value, never cross it) depends on the seed genuinely
    being on the correct side -- worth a cushion rather than trusting an
    iterative sweep's last digit."""
    _, ub_empty = _swapasap_cost_table(n, p, p_s, cutoff)

    allow_discard = discard_key[1]
    if allow_discard is True:
        return ub_empty
    cap = max(int(allow_discard), 1)
    clearing_slots = -(-(2 * n - 2) // cap)  # ceil division
    return clearing_slots + ub_empty


def _solve_envelope(parameters, bound, action_space, envelope, max_policy_iters=200):
    """Exact optimal expected-remaining-time over `envelope` -- states whose
    own transitions are fully known because they have all been backed up at
    least once -- with every successor OUTSIDE it pinned at its current
    `bound` value, and terminal successors at 0.

    This is BRTDP's version of the speedup policy_iteration(method="direct")
    gets. Asynchronous Bellman backups propagate information one step of the
    horizon per sweep, so on a slow-mixing chain -- low p, low p_s, where
    expected delivery runs to tens of time slots -- the trials must re-back-up
    every envelope state on the order of the horizon length before the bounds
    settle, no matter how the trial depth is capped (capping it just trades
    depth for more trials at identical total cost; measured). At n=4, p=0.25,
    p_s=0.35, cutoff=3, uncapped discard that came to 160,000 backups over a
    1,529-state envelope -- about 105 per state, ~9s -- to move bounds that a
    handful of sparse solves settle outright. Solving the envelope's own fixed
    point exactly, by policy iteration whose evaluation step is a single
    sparse LU, removes that horizon factor.

    The operator is the same min-over-actions Bellman operator for both
    bounds, so one routine serves both; only the pinned fringe values differ,
    and the caller clamps in its own direction. Pinning the fringe at a lower
    bound makes every policy's value here an underestimate of its true value,
    so the envelope optimum is <= V*; pinning at an upper bound makes it >=
    V*. Both results are therefore admissible, which is what lets the caller
    fold them into lb/ub directly.

    Note the envelope must be exactly the backed-up set: those are the states
    whose successors are all guaranteed to have been discover()ed, hence to
    have a `bound` entry to pin. Widening it to every discovered state would
    mean expanding transitions for every action at states the trials never
    reached -- the very cost _freeze_policy exists to avoid.

    Returns {key: cost}, or None if the linear system came out degenerate (an
    envelope-restricted policy that neither terminates nor leaves the envelope
    has infinite cost, which makes I - P singular); the caller then simply
    keeps whatever its backups already established."""
    keys = [k for k in envelope if not _is_terminal(k)]
    if not keys:
        return {}
    row_of = {k: i for i, k in enumerate(keys)}
    size = len(keys)

    # Per state, per action: the in-envelope successor rows and their
    # probabilities, plus a constant folding in the 1-per-slot cost and every
    # terminal (0) / pinned out-of-envelope successor.
    expanded = []
    for key in keys:
        per_action = []
        for action in action_space[key]:
            next_keys, probs, _ = cached_step(parameters, key, action)
            cols, ps, const = [], [], 1.0
            for s_key, P in zip(next_keys, probs):
                if P == 0.0 or _is_terminal(s_key):
                    continue
                row = row_of.get(s_key)
                if row is None:
                    const += P * bound[s_key]
                else:
                    cols.append(row)
                    ps.append(P)
            per_action.append((np.asarray(cols, dtype=np.intp),
                                np.asarray(ps, dtype=float), const))
        expanded.append(per_action)

    V = np.array([bound[k] for k in keys], dtype=float)
    choice = np.zeros(size, dtype=np.intp)

    def improve():
        """Greedy re-selection against the current V; True if anything moved."""
        changed = False
        for i in range(size):
            best_j, best_q = choice[i], np.inf
            for j, (cols, ps, const) in enumerate(expanded[i]):
                q = const + (float(ps @ V[cols]) if cols.size else 0.0)
                if q < best_q:
                    best_q, best_j = q, j
            if best_j != choice[i]:
                choice[i] = best_j
                changed = True
        return changed

    improve()  # seed the policy from the incoming bound
    for _ in range(max_policy_iters):
        rows, cols_all, data = [], [], []
        b = np.empty(size)
        for i in range(size):
            cols, ps, const = expanded[i][choice[i]]
            b[i] = const
            rows.append(i)
            cols_all.append(i)
            data.append(1.0)  # the "I" term; coo_matrix sums duplicates, so a
            for col, prob in zip(cols, ps):   # self-loop correctly gives 1 - P
                rows.append(i)
                cols_all.append(int(col))
                data.append(-float(prob))

        A = coo_matrix((data, (rows, cols_all)), shape=(size, size)).tocsc()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                V_new = spsolve(A, b)
            except Exception:
                return None
        if not np.all(np.isfinite(V_new)) or np.any(V_new < -1e-9):
            return None  # degenerate: improper policy inside the envelope
        V = V_new
        if not improve():
            break

    return {key: float(V[i]) for i, key in enumerate(keys)}


def _brtdp_backup(key, parameters, lb, ub, action_space, discover, backed_up=None):
    """One full Bellman backup at `key` over every action in its (real)
    action space -- the same per-state cost policy_iteration's own
    improvement step pays, just invoked at a single trial-visited state
    rather than swept over the whole state list. Tightens lb[key]/ub[key] in
    place (never past their previous value: a plain Bellman backup over
    already-admissible successor bounds is itself admissible, so the max/min
    clamps here are a belt-and-suspenders safety net against floating-point
    slack, not something that should ever actually bind). `discover(s_key)`
    lazily seeds lb/ub/action_space for any successor visited for the first
    time (relaxed-model lower bound, global upper bound -- see
    bounded_rtdp_policy).
    Returns [(action, q_lb, q_ub), ...], one triple per action in
    action_space[key], for the caller to pick a next action from without
    recomputing. If `backed_up` is given, records `key` into it -- that set is
    precisely the envelope _solve_envelope is allowed to solve over, since
    backing a state up is what guarantees all of its successors have been
    discover()ed and so carry bounds that can be pinned."""
    if backed_up is not None:
        backed_up.add(key)
    triples = []
    for action in action_space[key]:
        next_keys, probs, _ = cached_step(parameters, key, action)
        q_lb, q_ub = 1.0, 1.0
        for s_key, P in zip(next_keys, probs):
            if P == 0.0 or _is_terminal(s_key):
                continue
            discover(s_key)
            q_lb += P * lb[s_key]
            q_ub += P * ub[s_key]
        triples.append((action, q_lb, q_ub))

    lb[key] = max(lb[key], min(t[1] for t in triples))
    ub[key] = min(ub[key], min(t[2] for t in triples))
    return triples


def _brtdp_trial(root_key, parameters, lb, ub, action_space, discover, rng, max_depth, tau,
                  follow_upper=True):
    """One BRTDP trial: starting at root_key, repeatedly back up the current
    state, greedily pick the action that looks best under the bound being
    followed, then sample the NEXT state weighted by P(s'|s,a)*(ub[s']-lb[s'])
    --
    biasing exploration toward whichever successor still carries the most
    uncertainty, rather than toward whichever is simply most probable (plain
    trial-based sampling) -- which is what gives bounded RTDP its edge over
    plain/labeled RTDP: trials stop wasting themselves on already-resolved
    states and instead chase down the states actually keeping the root's own
    bound gap open. After the forward walk, backs up every visited state once
    more in reverse order, so the trial's newly-tightened leaf bounds
    propagate back up toward the root immediately rather than waiting for a
    future trial to happen to revisit the same states.

    A trial ends when the total gap-weighted mass reachable in one step,
    B = sum_s' P(s'|s,a) * (ub[s'] - lb[s']), falls below
    (ub[root] - lb[root]) / tau -- McMahan/Likhachev/Gordon's own termination
    rule, and the one piece of BRTDP that genuinely bounds trial length. The
    test is RELATIVE to the root's current gap, so it tightens automatically
    as the root converges: early trials (root gap wide) stop shallow, and
    only once the root is nearly solved do trials push deep into the
    remaining uncertainty. An ABSOLUTE per-state gap test cannot do this job
    at all here -- terminal successors are deliberately excluded from the
    sampling candidates (their gap is identically zero, so they carry no
    information), which means the forward walk never actually lands on a
    terminal state, and with nothing else to stop it every trial simply runs
    to max_depth. tau > 1 trades trial length against per-trial progress;
    tau = 10 is the standard choice.

    Also ends on: reaching a terminal state, every outcome of the greedy
    action being terminal, or max_depth (a safety cap for slow-mixing chains,
    e.g. low p).

    `follow_upper` chooses which bound the forward walk is greedy on, and
    which of the two bounds the trial therefore actually makes progress on:

      · True (the textbook BRTDP trial) walks the action minimizing q_ub.
        That converges the UPPER bound and the greedy policy along with it,
        since the states it visits are exactly the ones a near-optimal policy
        actually passes through.
      · False walks the action minimizing q_lb instead. This exists because
        the upper-bound walk alone lets the lower bound STALL outright, at a
        fixed point well below V*: lb[s] is a min over EVERY action, so it
        can be held down by a successor reachable only under some action the
        ub-greedy walk never takes -- that successor is never visited, never
        backed up, and keeps its weak relaxed-model seed forever, so lb[s]
        can never rise past it. (Observed, not hypothetical: at n=3, p=0.2,
        p_s=0.4, cutoff=3 the root's bounds freeze after the first trial and
        do not move again in thousands more, even though ub is by then
        already exactly optimal.) Walking the lb-minimizing action instead
        heads straight for those held-down successors, which is precisely
        what the lower bound needs to climb.

    bounded_rtdp_policy alternates the two, one of each per iteration."""
    # Snapshot the root's gap once, up front, rather than re-reading it each
    # step: this trial's own backups tighten the root as it goes, so a live
    # threshold shrinks out from under the test that's supposed to stop the
    # trial -- and collapses to exactly 0 the moment the root converges,
    # at which point nothing stops the walk short of max_depth. Freezing it
    # keeps each trial's depth tied to what the root gap actually was when
    # the trial was launched, which is what the rule is meant to express.
    stop_mass = (ub[root_key] - lb[root_key]) / tau
    q_index = 2 if follow_upper else 1  # _brtdp_backup yields (action, q_lb, q_ub)

    key = root_key
    trajectory = []
    for _ in range(max_depth):
        if _is_terminal(key):
            break
        trajectory.append(key)
        triples = _brtdp_backup(key, parameters, lb, ub, action_space, discover)

        best_action = min(triples, key=lambda t: t[q_index])[0]
        next_keys, probs, _ = cached_step(parameters, key, best_action)
        candidates = [(s_key, P) for s_key, P in zip(next_keys, probs)
                      if P > 0.0 and not _is_terminal(s_key)]
        weights = [P * (ub[s_key] - lb[s_key]) for s_key, P in candidates]
        total = sum(weights)
        # `total <= 0.0` also covers the already-converged root, where
        # stop_mass is itself 0 and the comparison alone would never fire.
        if total <= 0.0 or total < stop_mass:
            break

        r = rng.random() * total
        acc = 0.0
        chosen = candidates[-1][0]
        for (s_key, _), w in zip(candidates, weights):
            acc += w
            if r <= acc:
                chosen = s_key
                break
        key = chosen

    for visited in reversed(trajectory):
        _brtdp_backup(visited, parameters, lb, ub, action_space, discover)


def _swapasap_action(actions):
    """Swap-asap's own rule at a state: always the last entry of its action
    space (see policy_eval_swapasap's docstring -- get_actions builds swap
    combinations ascending and pairs each with discard combinations
    descending, so the full-swap/no-discard action always sorts last).
    Trivial to evaluate -- no optimization needed -- which is what makes it
    cheap enough to use for closing off the part of the state space BRTDP's
    trials never happened to visit (see _freeze_policy)."""
    return actions[-1]


def _freeze_policy(parameters, root_key, lb, ub, action_space, discover, tolerance):
    """Turn BRTDP's bounds into one fixed, fully-defined policy, and return
    it as {qubits_key: action} covering exactly the states reachable from
    root_key under itself.

    Decides each state's action and expands its successors in a single
    reachability-driven pass, rather than first freezing every state ever
    discovered and only then BFS-closing. Those two orderings give the same
    policy where it matters -- a state the frozen policy can never reach
    cannot affect the root's value -- but the sweep-everything version pays
    a full backup (i.e. a transition expansion for EVERY action) at a large
    number of states the final policy never visits, and, because each such
    backup calls discover() on the successors of every one of those actions,
    it inflates the discovered state set on the way through. That sweep was
    measured as the single most expensive phase of bounded_rtdp_policy
    (n=5, p=p_s=0.8, cutoff=2, cap=1: 1.47s of ~2.5s total, sweeping 1651
    states while growing the set to 3336, to produce a policy over 1439).
    Driving the pass from reachability instead touches only those 1439.

    Each state gets one backup, then its action: BRTDP's own upper-bound-
    greedy choice if that state's [lb, ub] gap is inside `tolerance`, and
    otherwise the cheap swap-asap fallback (_swapasap_action). The gap test
    is why the backup happens here at all -- many states one or two steps
    from delivery are unvisited by trials yet resolve to a tight gap after a
    single backup, so this pass upgrades them from the fallback to a real
    optimized action. Trusting a wide-gap state's own greedy pick instead
    would be the actual risk: unlike a rarely-reached branch, a frozen bad
    action is not weighted down by anything once the policy is executed, and
    an unconverged min() over largely-unresolved Q-estimates can pick a
    genuinely poor action rather than a slightly suboptimal one. Swap-asap
    is always valid and of bounded quality, so it is the safer default
    there.

    Mutates lb/ub/action_space in place (via the backups and discover)."""
    greedy_action = {}
    frontier = [root_key]
    seen = {root_key}
    while frontier:
        key = frontier.pop()
        if _is_terminal(key):
            continue
        discover(key)  # reachable states the trials never turned up

        triples = _brtdp_backup(key, parameters, lb, ub, action_space, discover)
        if ub[key] - lb[key] < tolerance:
            action = min(triples, key=lambda t: t[2])[0]
        else:
            action = _swapasap_action(action_space[key])
        greedy_action[key] = action

        next_keys, probs, _ = cached_step(parameters, key, action)
        for s_key, P in zip(next_keys, probs):
            if P == 0.0 or s_key in seen:
                continue
            seen.add(s_key)
            frontier.append(s_key)

    return greedy_action


def _one_hot(actions, chosen_action):
    policy_vec = [0.0] * len(actions)
    policy_vec[actions.index(chosen_action)] = 1.0
    return policy_vec


def _build_brtdp_agent(n, parameters, root_key, action_space, greedy_action):
    """Assemble an Agent (same bookkeeping policy_iteration itself uses)
    covering the frozen policy from _freeze_policy (BRTDP-greedy and
    swap-asap-fallback states alike), each with its one-hot chosen action as
    'policy' -- ready to hand to _evaluate_policy_direct for one exact
    evaluation pass over this fixed hybrid policy. Iterates greedy_action
    rather than action_space: action_space also holds every state discover()
    picked up as some NON-greedy action's successor along the way (a state
    only reachable via an action a later backup stopped preferring), which
    the frozen policy never actually visits, whereas greedy_action is by
    construction exactly the reachable set with no such orphans."""
    agent = Agent(n, parameters)
    root_idx = 0
    if not _is_terminal(root_key):
        raw_actions = action_space[root_key]
        agent.state_info[root_idx]["action_space"] = raw_actions
        agent.state_info[root_idx]["policy"] = _one_hot(raw_actions, greedy_action[root_key])

    for key, action in greedy_action.items():
        if key == root_key:
            continue
        raw_actions = action_space[key]
        idx = agent.observe(key, raw_actions)
        agent.update(idx, _one_hot(raw_actions, action), "policy")

    return agent


def bounded_rtdp_policy(n, p, p_s, cutoff, tolerance=1e-3, allow_discard=True,
                         max_trials=200000, max_trial_depth=10000, tau=10.0,
                         progress=True, savedata=True, seed=None):
    """Find a near-optimal swap+discard policy for an n-node chain under the
    qubit-age cutoff model via bounded real-time dynamic programming (BRTDP,
    McMahan/Likhachev/Gordon), instead of policy_iteration's exhaustive sweep
    over the full reachable state space. Built specifically for the
    discard-enabled setting, where the action space's discard dimension can
    make policy_iteration's full-state-space sweeps expensive (see this
    module's own _cached_swap_resolution / _cached_post_swap_step docstrings)
    even though most of that state space turns out to be irrelevant to
    reaching the goal quickly.

    Maintains a lower bound (from the relaxed, no-decoherence model --
    _relaxed_lower_bound) and an upper bound (from swap-asap, reachable at
    zero-or-bounded extra cost from any state via discard --
    _global_upper_bound) on every visited state's optimal expected-remaining-
    time, and runs trials (_brtdp_trial) that chase down whichever successors
    still carry the most bound-gap uncertainty, until the root's own gap is
    under `tolerance`. Because trials only ever visit states relevant to that
    gap, most of the real reachable state space -- specifically the part an
    optimal or near-optimal policy essentially never visits -- is never
    examined at all.

    On when this is actually the faster choice: the state-space saving is
    consistent (measured against policy_iteration on the same parameters,
    roughly 2-5x fewer states over n=3..5 with discard enabled), but it does
    not automatically buy the same factor in wall-clock time. BRTDP backs up
    each state it does visit many times over, where policy_iteration's
    "direct" method makes one pass and hands the rest to a sparse solve, so
    the two can come out close -- at n=5, cutoff=2, cap=1 BRTDP visited 2.4x
    fewer states yet ran slightly slower, while at n=4, cutoff=4, cap=1 it
    was 4.5x smaller and 2.7x faster. The saving turns into speed when
    enumerating and evaluating the full state space is what dominates --
    larger n, larger cutoff, uncapped or loosely-capped discard -- and it is
    the memory saving, not the time, that is dependable. For small chains
    policy_iteration is both exact and comparably quick; prefer it there.

        ---Inputs---
            · n, p, p_s, cutoff: see policy_iteration.
            · tolerance:    (float) trials stop once the root state's own
                            [lower bound, upper bound] gap is smaller than
                            this many expected time slots.
            · allow_discard:    (bool or non-negative int) must enable
                                discard -- True (uncapped) or an int cap --
                                since both the admissibility of the upper
                                bound and the point of this algorithm over
                                policy_iteration depend on it. Use
                                policy_iteration for allow_discard=False.
            · max_trials, max_trial_depth: safety caps in case tolerance
                            isn't reached (slow-mixing chain, e.g. low p) --
                            bounded_rtdp_policy still returns its best
                            current bounds/policy rather than looping forever,
                            printing a warning (unless progress=False) if the
                            cap was hit before the gap closed.
            · tau:      (float > 1) how deep each trial runs, as a ratio
                        against the root's own current gap -- see
                        _brtdp_trial. Larger means longer trials that each
                        make more progress; 10 is the standard choice.
            · progress: (bool) if True, prints progress.
            · savedata: (bool) if True, saves the outputs in a file.
            · seed:     (optional int) seeds the trial-sampling RNG, for
                        reproducible runs.
        ---Outputs---
            · v0_evol:  (list of lists) [[lower_bound, upper_bound]] on the
                        root state's expected DELIVERY time (this module's
                        usual -(value+1) reporting convention -- matches
                        -(state_info[0]['value']+1), NOT the raw cost :=
                        -value units _brtdp_backup works in internally), one
                        pair per completed trial -- the bracket
                        bounded_rtdp_policy can report as an error bar on how
                        close state_info[0]['value'] is to optimal, which
                        policy_iteration's labeled solved-or-not convergence
                        can't offer.
            · state_info: (list of dicts) same shape as policy_iteration's
                          own output -- every state reachable under the
                          final policy has an entry ('state', 'action_space',
                          'policy', 'value'), states BRTDP's trials actually
                          backed up alongside swap-asap-padded ones for the
                          rest (see _freeze_policy) -- with 'value'
                          the exact value of this fixed hybrid policy (from
                          one _evaluate_policy_direct pass), not the raw
                          trial bounds.
            · exe_time: (float) execution time of the algorithm in seconds."""
    assert isinstance(n, int) and n >= 2, "n must be an integer >= 2"
    assert 0 <= p <= 1, "p must be between zero and one"
    assert 0 <= p_s <= 1, "p_s must be between zero and one"
    assert isinstance(cutoff, int) and cutoff > 0, "cutoff must be a positive integer"
    assert allow_discard is True or (isinstance(allow_discard, int) and allow_discard is not True
                                      and allow_discard >= 0), (
        "bounded_rtdp_policy requires discard to be enabled (allow_discard=True or a "
        "non-negative int cap) -- both the upper bound and the point of this algorithm "
        "over policy_iteration depend on it; use policy_iteration for allow_discard=False")

    parameters = Configuration(p=p, p_s=p_s, cut=cutoff, allow_discard=allow_discard)
    discard_key = _discard_key(allow_discard)
    swapasap_costs, _ = _swapasap_cost_table(n, p, p_s, cutoff)
    global_ub = _global_upper_bound(n, p, p_s, cutoff, discard_key)
    root_key = qubits_to_key(State.initial(n, parameters).qubits)

    lb, ub, action_space = {}, {}, {}

    def discover(key):
        if key in action_space:
            return
        action_space[key] = State(key_to_qubits(key), parameters).get_actions()
        lb[key] = _relaxed_lower_bound(n, p, p_s, key)
        # Both seeds are valid upper bounds wherever they're defined, so take
        # whichever is tighter: swap-asap's own exact per-state value inside
        # its reachable closure (much the better of the two, and covering
        # most of what trials actually visit), the padded global constant
        # outside it. See _swapasap_cost_table / _global_upper_bound.
        ub[key] = min(swapasap_costs.get(key, global_ub), global_ub)

    discover(root_key)

    rng = random.Random(seed)

    start_time = time.time()
    v0_evol = []
    trials_run = 0
    for trials_run in range(1, max_trials + 1):
        # One trial greedy on each bound: the upper-bound walk converges ub
        # and the greedy policy, the lower-bound walk keeps lb from stalling
        # against successors the upper-bound walk never visits. See
        # _brtdp_trial's `follow_upper` -- without the second walk the root's
        # gap can freeze open permanently.
        for follow_upper in (True, False):
            _brtdp_trial(root_key, parameters, lb, ub, action_space, discover,
                         rng, max_trial_depth, tau, follow_upper=follow_upper)
        # Reported in this module's usual "expected delivery time" units
        # (-(value+1), matching policy_iteration/policy_eval_swapasap's own
        # reporting) rather than the raw cost := -value units _brtdp_backup
        # works in internally -- see _relaxed_optimal_cost_table's docstring
        # for why those differ by exactly 1.
        v0_evol.append([lb[root_key] - 1, ub[root_key] - 1])
        if progress:
            # Printed before the convergence check below, so the gap shown on
            # the final line is often already inside tolerance -- hence a
            # neutral separator rather than a ">" that would then be a lie.
            print("BRTDP trial %d: root gap = %.2e (tolerance %.2e), %d states" % (
                trials_run, ub[root_key] - lb[root_key], tolerance, len(action_space)
            ) + " " * 10, end="\r")
        if ub[root_key] - lb[root_key] < tolerance:
            break

    if progress:
        print(" " * 100, end="\r")
    if ub[root_key] - lb[root_key] >= tolerance and progress:
        print("bounded_rtdp_policy: reached max_trials=%d before tolerance -- "
              "root gap is still %.2e" % (max_trials, ub[root_key] - lb[root_key]))

    ### Freeze one fixed policy over the states it can actually reach,   ###
    ### then get exact values for it.                                    ###
    # The root's gap being tight does NOT imply every visited state's own
    # gap is tight: a state can look "solved" in aggregate purely because
    # its highest-uncertainty successor happens to carry low probability
    # under its current best action (exactly what gap-weighted sampling is
    # supposed to exploit -- correctly leaving genuinely low-relevance
    # branches unexplored). _freeze_policy is what applies the per-state gap
    # test that decides, state by state, whether BRTDP's own greedy pick is
    # trustworthy there or whether swap-asap should stand in.
    greedy_action = _freeze_policy(parameters, root_key, lb, ub, action_space,
                                    discover, tolerance)
    agent = _build_brtdp_agent(n, parameters, root_key, action_space, greedy_action)
    _evaluate_policy_direct(agent, parameters)

    exe_time = time.time() - start_time

    if savedata:
        save_brtdp_data(n, p, p_s, cutoff, tolerance, v0_evol, agent.state_info, exe_time,
                         allow_discard=allow_discard)

    return v0_evol, agent.state_info, exe_time


# --------------------------------------------------------------------------- #
# --------------------------------  I/O  ------------------------------------ #
# --------------------------------------------------------------------------- #

def _discard_enabled(allow_discard):
    """Mirrors Configuration.discard_enabled for the raw allow_discard value
    (bool or int) as passed around the I/O layer, where a full Configuration
    isn't always at hand. False disables discard; True or any int (including
    0, a degenerate always-empty cap) enables it."""
    return allow_discard is not False


def _discard_suffix(allow_discard):
    """Filename suffix distinguishing no-discard / uncapped-discard / capped-
    discard runs, so they never collide on disk."""
    if allow_discard is False:
        return ''
    if allow_discard is True:
        return '_discard'
    return '_discard%d' % allow_discard


def _export_state_info(state_info, allow_discard):
    """Convert internal state_info (qubit-age tuples, qubit-index actions)
    into the paper's node-indexed matrix / node-index-action representation
    (see qubits_to_matrix, action_to_nodes) so that
    AlvaroGI/optimal-homogeneous-chain's analysis and plotting helpers can
    load the saved file directly. The raw qubit-age tuple is kept alongside
    under 'qubits' for full-fidelity reuse within this project.

    When discard is enabled, each action is a (swap_nodes, discard_qubits)
    pair: 'action_space' is exported as just the swap half, in the paper's
    node-index convention, since that's the only half the paper's code
    understands; the discard half is exported alongside under
    'discard_space' (raw qubit indices, one entry per action_space entry, no
    paper equivalent to convert to)."""
    exportable_info = []
    for entry in state_info:
        qubits = key_to_qubits(entry["state"])
        exported = {
            "state": qubits_to_matrix(qubits),
            "qubits": entry["state"],
            "policy": entry["policy"],
            "value": entry["value"],
        }
        if _discard_enabled(allow_discard):
            exported["action_space"] = [action_to_nodes(swap_nodes) for swap_nodes, _ in entry["action_space"]]
            exported["discard_space"] = [list(discard_qubits) for _, discard_qubits in entry["action_space"]]
        else:
            exported["action_space"] = [action_to_nodes(a) for a in entry["action_space"]]
        exportable_info.append(exported)
    return exportable_info


def _policyiter_filename(n, p, p_s, cutoff, tolerance, allow_discard=False):
    return 'data_policyiter/n%s_p%.3f_ps%.3f_tc%s_tol%s%s' % (
        n, p, p_s, cutoff, tolerance, _discard_suffix(allow_discard))


def check_policyiter_data(n, p, p_s, cutoff, tolerance, allow_discard=False):
    """True if policy iteration has already been run and saved for this set
    of parameters."""
    return Path(_policyiter_filename(n, p, p_s, cutoff, tolerance, allow_discard)).exists()


def save_policyiter_data(n, p, p_s, cutoff, tolerance, v0_evol, state_info, exe_time,
                          allow_discard=False):
    """Save policy iteration data for this set of parameters."""
    os.makedirs('data_policyiter', exist_ok=True)
    filename = _policyiter_filename(n, p, p_s, cutoff, tolerance, allow_discard)
    data = {'v0_evol': v0_evol, 'state_info': _export_state_info(state_info, allow_discard),
            'exe_time': exe_time}
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_policyiter_data(n, p, p_s, cutoff, tolerance, allow_discard=False):
    """Load policy iteration data saved by save_policyiter_data / policy_iteration."""
    filename = _policyiter_filename(n, p, p_s, cutoff, tolerance, allow_discard)
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data['v0_evol'], data['state_info'], data['exe_time']


def _swapasap_filename(n, p, p_s, cutoff, tolerance, allow_discard=False):
    return 'data_swapasap/swapasap_n%s_p%.3f_ps%.3f_tc%s_tol%s%s' % (
        n, p, p_s, cutoff, tolerance, _discard_suffix(allow_discard))


def check_swapasap_data(n, p, p_s, cutoff, tolerance, allow_discard=False):
    """True if the swap-asap baseline has already been evaluated and saved
    for this set of parameters."""
    return Path(_swapasap_filename(n, p, p_s, cutoff, tolerance, allow_discard)).exists()


def save_swapasap_data(n, p, p_s, cutoff, tolerance, v0_evol, state_info, exe_time,
                        allow_discard=False):
    """Save swap-asap evaluation data for this set of parameters."""
    os.makedirs('data_swapasap', exist_ok=True)
    filename = _swapasap_filename(n, p, p_s, cutoff, tolerance, allow_discard)
    data = {'v0_evol': v0_evol, 'state_info': _export_state_info(state_info, allow_discard),
            'exe_time': exe_time}
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_swapasap_data(n, p, p_s, cutoff, tolerance, allow_discard=False):
    """Load swap-asap evaluation data saved by save_swapasap_data / policy_eval_swapasap."""
    filename = _swapasap_filename(n, p, p_s, cutoff, tolerance, allow_discard)
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data['v0_evol'], data['state_info'], data['exe_time']


def _brtdp_filename(n, p, p_s, cutoff, tolerance, allow_discard=True):
    # Own directory/prefix, not data_policyiter/ -- bounded_rtdp_policy's
    # result is a converged-to-tolerance approximation, not the exact
    # optimum policy_iteration solves for, so the two shouldn't share a cache
    # namespace under the same (n, p, p_s, cutoff, discard) key.
    return 'data_brtdp/brtdp_n%s_p%.3f_ps%.3f_tc%s_tol%s%s' % (
        n, p, p_s, cutoff, tolerance, _discard_suffix(allow_discard))


def check_brtdp_data(n, p, p_s, cutoff, tolerance, allow_discard=True):
    """True if bounded_rtdp_policy has already been run and saved for this
    set of parameters."""
    return Path(_brtdp_filename(n, p, p_s, cutoff, tolerance, allow_discard)).exists()


def save_brtdp_data(n, p, p_s, cutoff, tolerance, v0_evol, state_info, exe_time,
                     allow_discard=True):
    """Save bounded_rtdp_policy data for this set of parameters. v0_evol here
    is a list of [lower_bound, upper_bound] pairs (one per trial), not a
    single value trace -- the error bar bounded_rtdp_policy offers over
    policy_iteration's labeled solved-or-not convergence."""
    os.makedirs('data_brtdp', exist_ok=True)
    filename = _brtdp_filename(n, p, p_s, cutoff, tolerance, allow_discard)
    data = {'v0_evol': v0_evol, 'state_info': _export_state_info(state_info, allow_discard),
            'exe_time': exe_time}
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_brtdp_data(n, p, p_s, cutoff, tolerance, allow_discard=True):
    """Load data saved by save_brtdp_data / bounded_rtdp_policy."""
    filename = _brtdp_filename(n, p, p_s, cutoff, tolerance, allow_discard)
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data['v0_evol'], data['state_info'], data['exe_time']


# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Policy iteration for the qubit-age cutoff model.')
    parser.add_argument('--n', type=int, default=3, help='Number of nodes.')
    parser.add_argument('--p', type=float, default=0.8,
                        help='Elementary link generation success probability.')
    parser.add_argument('--p_s', type=float, default=0.8,
                        help='Entanglement swap success probability.')
    parser.add_argument('--cutoff', type=int, default=5, help='Qubit-age cutoff.')
    parser.add_argument('--tol', type=float, default=1e-5, help='Value tolerance.')
    parser.add_argument('--policy', choices=['optimal', 'swap-asap', 'brtdp'], default='optimal',
                        help='"optimal" (policy iteration), "swap-asap" (baseline), or "brtdp" '
                             '(bounded real-time dynamic programming -- see bounded_rtdp_policy; '
                             'requires --discard).')
    parser.add_argument('--discard', nargs='?', const=True, default=False, type=int,
                        help='Allow the agent to discard occupied qubits each time slot. '
                             'Pass a number to cap how many qubits may be discarded per '
                             'time slot (e.g. --discard 1); with no number, there is no cap.')
    parser.add_argument('--method', choices=['iterative', 'direct'], default='iterative',
                        help='How to run each policy-evaluation step for --policy optimal: '
                             '"iterative" (default) matches the paper\'s own Bellman-backup '
                             'sweeps; "direct" solves each evaluation exactly via sparse LU '
                             'instead, substantially faster once --discard inflates the '
                             'action/state space (see policy_iteration\'s docstring). Ignored '
                             'for --policy swap-asap/brtdp.')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output.')
    args = parser.parse_args()

    if args.policy == 'optimal':
        if check_policyiter_data(args.n, args.p, args.p_s, args.cutoff, args.tol, args.discard):
            print('Data already exists!')
        else:
            v0_evol, state_info, exe_time = policy_iteration(
                args.n, args.p, args.p_s, args.cutoff,
                tolerance=args.tol, progress=not args.quiet, savedata=True,
                allow_discard=args.discard, method=args.method)
            print('Done in %.1fs! %d states. Expected delivery time: %.4f time slots.' % (
                exe_time, len(state_info), -(state_info[0]['value'] + 1)))
    elif args.policy == 'swap-asap':
        if check_swapasap_data(args.n, args.p, args.p_s, args.cutoff, args.tol, args.discard):
            print('Data already exists!')
        else:
            v0_evol, state_info, exe_time = policy_eval_swapasap(
                args.n, args.p, args.p_s, args.cutoff,
                tolerance=args.tol, progress=not args.quiet, savedata=True,
                allow_discard=args.discard)
            print('Done in %.1fs! %d states. Expected delivery time: %.4f time slots.' % (
                exe_time, len(state_info), -(state_info[0]['value'] + 1)))
    else:
        if not args.discard:
            parser.error('--policy brtdp requires --discard (True or an int cap)')
        if check_brtdp_data(args.n, args.p, args.p_s, args.cutoff, args.tol, args.discard):
            print('Data already exists!')
        else:
            v0_evol, state_info, exe_time = bounded_rtdp_policy(
                args.n, args.p, args.p_s, args.cutoff,
                tolerance=args.tol, progress=not args.quiet, savedata=True,
                allow_discard=args.discard)
            lb, ub = v0_evol[-1]
            print('Done in %.1fs! %d states. Expected delivery time: %.4f time slots '
                  '(bounds [%.4f, %.4f]).' % (
                exe_time, len(state_info), -(state_info[0]['value'] + 1), lb, ub))
