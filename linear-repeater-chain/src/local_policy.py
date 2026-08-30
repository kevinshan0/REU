"""Direct optimization of a strictly local, per-node policy for the qubit-age
cutoff model defined in environment.py / policy.py.

This sits between the two other local-policy approaches already in this repo:

- distill.py DISTILLS an already-solved global-knowledge policy (policy.py's
  policy_iteration) down to per-node tables, after the fact, by majority vote
  over every global state consistent with a node's own qubits. The result
  only approximates optimal local behavior -- it's whatever the global
  policy's decisions look like when projected onto one node's view.
- po_policy.py SOLVES a genuine decentralized POMDP directly, but lets each
  node track a running Bayesian belief over the full chain state, updated
  from its own observation history -- correct in general, but the belief
  space has to be rounded/pruned to stay finite, and every extra bit of
  belief precision multiplies the hyper-state space.

This module solves the decentralized problem directly too, but for the
simpler MEMORYLESS (reactive) case explicitly asked for: each node's decision
is a function of only its own current qubit state (age/hanging of the one or
two qubits it holds this time slot) -- no belief, no history, exactly what a
real node could compute from what it physically has in front of it right
now. That makes it a genuinely "local-knowledge" policy in the same sense
distill.py's local state (see node_qubits/decompose_action there) is local --
just optimized directly for self-consistency instead of copied from the
global solution.

Algorithm: self-consistent coordinate-ascent policy iteration
--------------------------------------------------------------
Each node i owns a table node_policy[i]: local_state -> one-hot local action,
where local_state is (age, hanging) for each of node i's own qubits (see
po_environment.observation_of) and a local action is (swap, discard_qubits)
(see po_environment.local_actions/combine_actions). The *same* table entry is
shared by every global state that happens to give node i that particular
local observation, so one node's decision is a single choice made across
many different global contexts at once -- that's the "self-consistency"
constraint an unconstrained global policy doesn't have.

Policy iteration proceeds over the global state graph exactly as in
policy.py (states discovered lazily via policy.cached_step, memoized
transitions shared with every other module in this repo that imports it):
    1. Policy evaluation: with every node_policy[i] table held fixed, the
       joint action at any global state is the combination of each node's
       independent lookup (_joint_action_distribution) -- evaluate every
       discovered state's value against that induced Markov chain by
       Bellman-backup sweeps (_evaluate_joint_policy), exactly
       policy.py's _evaluate_policy_iterative with the induced-joint-action
       lookup standing in for a stored per-state policy/action_space.
    2. Visitation: how often each global state is actually visited under the
       current joint policy (_visitation_counts) -- the same transposed
       linear-system trick as distill.py's _visitation_counts /
       po_policy.py's _hyperstate_visitation_counts, needed because a single
       (node, local_state) table entry must be optimized against the whole
       weighted mixture of global states sharing that local_state, not
       against just one.
    3. Policy improvement: for every (node, local_state) pair actually
       touched by a visited state, compute the visitation-weighted Q-value of
       every locally-valid candidate action for that node -- holding every
       other node's table fixed at its own current choice (the `override`
       argument to _joint_action_distribution) -- and keep the best
       (_improve_node_policies). This is a single-agent best-response step
       repeated per node per round, i.e. coordinate ascent on a shared
       objective; at a fixed point every node's rule is a best response to
       every other node's (including its own), which is as close to a Nash/
       Markov-perfect equilibrium for this cooperative team as
       po_policy.py's belief-based version, just without the belief
       dimension. Mirrors po_policy.py's _improve_node_policies almost
       exactly, with local_state standing in for belief and cached_step
       standing in for hyperstate_step (no belief update needed here).
    4. Stop under the same dual criterion as policy.py/po_policy.py: policy
       exactly stable for two consecutive outer rounds, OR every discovered
       state's value has moved by less than `tolerance` since the last round
       (see policy.policy_iteration's docstring for why both checks exist).

Output format
-------------
The saved pickle has exactly the same shape policy.py's save_policyiter_data
produces -- {'v0_evol', 'state_info', 'exe_time'}, each state_info entry with
'state' (paper-compatible node-indexed matrix), 'qubits' (raw qubit-age key),
'policy' (one-hot over 'action_space'), 'value' -- built by feeding this
module's own state_info (in policy.py's pre-export shape) through policy.py's
own _export_state_info. This is possible because, once node_policy converges
to one-hot tables (guaranteed for every local_state actually touched by
policy improvement; any local_state that's lazily initialized but never
visited under the final policy is force-collapsed to its highest-weight
candidate just before export -- see the "leftover uniform tables" step in
local_policy_iteration), the induced joint action at every discovered global
state is a single deterministic action, exactly like a converged policy.py
run. So the resulting file is a drop-in for anything that consumes
policy.load_policyiter_data's return shape -- see build_policy_json below,
a copy of simulate.py's own build_policy_json with only the loader swapped,
producing the same JSON simulate.jl consumes.
"""
from collections import defaultdict
from itertools import product
from pathlib import Path
import argparse
import json
import os
import pickle
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from environment import Configuration, State
from policy import (
    _discard_suffix,
    _export_state_info,
    _is_terminal,
    cached_step,
    key_to_qubits,
    qubits_to_key,
)
from distill import _check_local_policy_discard_support
from po_environment import combine_actions, local_actions, node_qubits, observation_of

# See policy.py / distill.py / po_policy.py / simulate.py: anchors relative
# data_* paths to this module's own directory regardless of the caller's CWD.
os.chdir(Path(__file__).resolve().parent)


# --------------------------------------------------------------------------- #
# ------------------------------  AGENT  ------------------------------------ #
# --------------------------------------------------------------------------- #

class Agent:
    """Bookkeeping for policy iteration over the global state graph: every
    distinct global state discovered so far (by qubit-age key, see
    policy.qubits_to_key) and its current value estimate. Unlike policy.py's
    own Agent, there's no per-state action_space/policy stored here -- the
    joint action at any state is instead derived on the fly from the much
    smaller per-node node_policy tables (see _joint_action_distribution),
    since a state's local observations (and hence its candidate local
    actions) are cheap to re-derive. Value convention matches policy.py:
    expected remaining time slots to delivery is -(value + 1)."""

    def __init__(self, n, parameters):
        s0 = qubits_to_key(State.initial(n, parameters).qubits)
        self._index = {s0: 0}
        self.state_list = [s0]
        self.values = [0.0]

    def observe(self, key):
        idx = self._index.get(key)
        if idx is None:
            idx = len(self.state_list)
            self._index[key] = idx
            self.state_list.append(key)
            self.values.append(0.0)
        return idx


# --------------------------------------------------------------------------- #
# ------------------------  JOINT ACTION FROM LOCAL TABLES  ------------------ #
# --------------------------------------------------------------------------- #

def _node_table(node_policy, node, local_state, own_qubits, discard_enabled):
    """The candidate-local-action distribution `node` currently uses at
    `local_state` -- lazily initialized to uniform-random over every locally
    valid candidate the first time this local_state is encountered, mirroring
    policy.py's Agent.observe/init_policy and po_policy.py's own
    _node_table (there keyed on belief; here keyed directly on the bare local
    observation)."""
    table = node_policy[node].get(local_state)
    if table is None:
        candidates = local_actions(local_state, own_qubits, discard_enabled)
        table = [(a, 1.0 / len(candidates)) for a in candidates]
        node_policy[node][local_state] = table
    return table


def _joint_action_distribution(key, own_qubits_by_node, node_policy, discard_enabled, override=None):
    """Every (global action, probability) pair reachable by each node
    independently sampling its own local action from its own current table at
    its own current local observation of `key` -- the joint of n independent
    per-node choices. Mirrors distill.py's _joint_action_distribution and
    po_policy.py's own version of the same idea, applied to bare local
    observations instead of a pre-built distilled table or a belief.

    `override`, if given, is a (node, local_action) pin: every other node
    still uses its own current table, but `node`'s choice is fixed to
    `local_action` at probability 1 -- used by policy improvement to ask "if
    this node deviated to this candidate, holding every other node fixed at
    its current policy, what would happen?"."""
    per_node = []
    for j, own_qubits in enumerate(own_qubits_by_node):
        if override is not None and override[0] == j:
            per_node.append([(override[1], 1.0)])
        else:
            local_state = observation_of(key, own_qubits)
            per_node.append(_node_table(node_policy, j, local_state, own_qubits, discard_enabled))

    combined = defaultdict(float)
    for combo in product(*per_node):
        prob = 1.0
        for _a, p in combo:
            prob *= p
        if prob == 0.0:
            continue
        action = combine_actions(own_qubits_by_node, [a for a, _p in combo], discard_enabled)
        combined[action] += prob

    return list(combined.items())


# --------------------------------------------------------------------------- #
# ---------------------------  POLICY EVALUATION  ----------------------------- #
# --------------------------------------------------------------------------- #

def _evaluate_joint_policy(agent, own_qubits_by_node, node_policy, parameters, tol, progress):
    """Evaluate the current node_policy tables by successive approximation
    over the global state graph -- exactly policy.py's
    _evaluate_policy_iterative, but with the joint action at each state
    derived from node_policy (_joint_action_distribution) rather than read
    from a stored per-state policy/action_space. New states are registered
    via agent.observe() and picked up on a later pass of the same sweep,
    since it re-reads len(agent.state_list) every iteration."""
    discard_enabled = parameters.discard_enabled
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

            value = agent.values[idx]
            v = 0.0
            for action, action_prob in _joint_action_distribution(key, own_qubits_by_node, node_policy,
                                                                    discard_enabled):
                if action_prob == 0.0:
                    continue
                next_keys, probs, _ = cached_step(parameters, key, action)
                for s_key, P in zip(next_keys, probs):
                    if P == 0.0:
                        continue
                    s_idx = agent.observe(s_key)
                    v += action_prob * P * (-1 + agent.values[s_idx])

            error = max(error, abs(value - v))
            agent.values[idx] = v
            idx += 1

        v0.append(agent.values[0])
        if progress:
            print("Local policy eval.: error = %.2e > %.2e = tolerance, %d states" % (
                error, tol, len(agent.state_list)) + " " * 10, end="\r")
    return v0


# --------------------------------------------------------------------------- #
# -----------------------------  VISITATION  ---------------------------------- #
# --------------------------------------------------------------------------- #

def _visitation_counts(agent, own_qubits_by_node, node_policy, parameters):
    """Expected number of times each known global state is visited, starting
    from the chain's initial state, under the current node_policy-induced
    joint policy -- the same transposed-linear-system trick as
    distill.py's _visitation_counts / po_policy.py's
    _hyperstate_visitation_counts, applied to the plain global state graph.
    Needed because a single (node, local_state) table entry is shared by many
    global states, so policy improvement has to weight each one by how often
    it's actually reached, not count every discovered state equally."""
    discard_enabled = parameters.discard_enabled
    index = {key: i for i, key in enumerate(agent.state_list)}
    coeffs = {}

    for i, key in enumerate(agent.state_list):
        if _is_terminal(key):
            continue
        for action, action_prob in _joint_action_distribution(key, own_qubits_by_node, node_policy,
                                                                discard_enabled):
            if action_prob == 0.0:
                continue
            next_keys, probs, _ = cached_step(parameters, key, action)
            for s_key, P in zip(next_keys, probs):
                if P == 0.0:
                    continue
                j = index.get(s_key)
                if j is None:
                    # Not (yet) discovered -- can happen right after a policy
                    # improvement round changes behavior; it'll get its own
                    # weight once the next evaluation sweep discovers it.
                    continue
                if not _is_terminal(s_key):
                    coeffs[(i, j)] = coeffs.get((i, j), 0.0) - action_prob * P

    non_terminal = [i for i, key in enumerate(agent.state_list) if not _is_terminal(key)]
    row_of = {idx: row for row, idx in enumerate(non_terminal)}
    size = len(non_terminal)

    rows, cols, data = [], [], []
    for (r_idx, c_idx), coeff in coeffs.items():
        rows.append(row_of[r_idx])
        cols.append(row_of[c_idx])
        data.append(coeff)
    for idx in non_terminal:
        rows.append(row_of[idx])
        cols.append(row_of[idx])
        data.append(1.0)

    A = coo_matrix((data, (rows, cols)), shape=(size, size)).tocsc()
    b = np.zeros(size)
    b[row_of[0]] = 1.0
    x = spsolve(A.transpose().tocsc(), b)

    counts = [0.0] * len(agent.state_list)
    for idx in non_terminal:
        counts[idx] = float(x[row_of[idx]])
    return counts


def _evaluate_joint_policy_direct(agent, own_qubits_by_node, node_policy, parameters):
    """Evaluate the current node_policy tables exactly, solving the linear
    system they define directly (sparse LU) instead of iterating a Bellman
    backup to convergence -- mirrors policy.py's _evaluate_policy_direct
    (V = -1 + sum_a policy(s,a) sum_s' P(s'|s,a) V(s') is |S| linear equations
    in |S| unknowns; terminal states are absorbing with V=0 and contribute no
    column), applied to the joint-action-induced chain
    (_joint_action_distribution) instead of a stored per-state policy.
    Substantially faster than _evaluate_joint_policy once discard inflates
    the branching factor per state -- see policy.py's policy_iteration
    docstring on 'direct' vs 'iterative', and distill.py's own 'method'
    parameter for the same rationale applied to distillation.

    Walks agent.state_list exactly like _evaluate_joint_policy (discovering
    new states via agent.observe() the same way, since the while loop
    re-reads len(agent.state_list) every iteration), but instead of
    accumulating value updates it accumulates (row, col) -> coefficient
    entries for the sparse matrix, then solves once.
    Returns v0: [initial state's value] (a single exact value, not a
    per-sweep trace -- there's no sweep-by-sweep convergence here)."""
    discard_enabled = parameters.discard_enabled
    coeffs = {}

    idx = 0
    while idx < len(agent.state_list):
        key = agent.state_list[idx]
        if _is_terminal(key):
            idx += 1
            continue

        for action, action_prob in _joint_action_distribution(key, own_qubits_by_node, node_policy,
                                                                discard_enabled):
            if action_prob == 0.0:
                continue
            next_keys, probs, _ = cached_step(parameters, key, action)
            for s_key, P in zip(next_keys, probs):
                if P == 0.0:
                    continue
                s_idx = agent.observe(s_key)
                if not _is_terminal(s_key):
                    coeff_key = (idx, s_idx)
                    coeffs[coeff_key] = coeffs.get(coeff_key, 0.0) - action_prob * P

        idx += 1

    non_terminal_indices = [i for i in range(len(agent.state_list)) if not _is_terminal(agent.state_list[i])]
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
        agent.values[agent_idx] = float(V[row])

    return [agent.values[0]]


# --------------------------------------------------------------------------- #
# ---------------------------  POLICY IMPROVEMENT  ---------------------------- #
# --------------------------------------------------------------------------- #

def _improve_node_policies(agent, node_policy, visitation, own_qubits_by_node, parameters):
    """One policy-improvement round: for every (node, local_state) pair
    touched by a visited, non-terminal global state, compute the
    visitation-weighted expected Q-value of every locally valid candidate
    action for that node (holding every other node's table fixed -- see
    _joint_action_distribution's `override`), and set that node's table at
    that local_state to the single best candidate (one-hot, matching
    policy.py's converged-policy convention). Mirrors po_policy.py's
    _improve_node_policies with local_state in place of belief and
    policy.cached_step in place of hyperstate_step (no belief to update).

    Returns True if any (node, local_state) table changed."""
    discard_enabled = parameters.discard_enabled
    n = len(own_qubits_by_node)
    q_accum = defaultdict(float)
    touched = set()

    for idx, key in enumerate(agent.state_list):
        if _is_terminal(key):
            continue
        w = visitation[idx]
        if w <= 0.0:
            continue

        for j in range(n):
            own_qubits = own_qubits_by_node[j]
            local_state = observation_of(key, own_qubits)
            candidates = local_actions(local_state, own_qubits, discard_enabled)
            if len(candidates) < 2:
                continue
            touched.add((j, local_state))

            for a in candidates:
                joint_dist = _joint_action_distribution(key, own_qubits_by_node, node_policy, discard_enabled,
                                                          override=(j, a))
                for action, action_prob in joint_dist:
                    if action_prob == 0.0:
                        continue
                    next_keys, probs, _ = cached_step(parameters, key, action)
                    for s_key, P in zip(next_keys, probs):
                        if P == 0.0:
                            continue
                        s_idx = agent.observe(s_key)
                        q_accum[(j, local_state, a)] += w * action_prob * P * (-1 + agent.values[s_idx])

    changed = False
    for (j, local_state) in touched:
        candidates = local_actions(local_state, own_qubits_by_node[j], discard_enabled)
        qs = [q_accum[(j, local_state, a)] for a in candidates]
        best = int(np.argmax(qs))
        current = node_policy[j].get(local_state)
        is_onehot_best = current is not None and len(current) == 1 and current[0][0] == candidates[best]
        if not is_onehot_best:
            changed = True
        node_policy[j][local_state] = [(candidates[best], 1.0)]

    return changed


# --------------------------------------------------------------------------- #
# ---------------------------  POLICY ITERATION  ----------------------------- #
# --------------------------------------------------------------------------- #

def local_policy_iteration(n, p, p_s, cutoff, tolerance=1e-5, tolerance_stability=1e-1,
                            progress=True, savedata=True, allow_discard=False, method="iterative"):
    """Find the optimal strictly-local (memoryless, per-node) swap policy for
    an n-node chain under the qubit-age cutoff model: each node decides its
    own action from only the (age, hanging) of its own one or two qubits,
    optimized directly (not distilled from a pre-solved global policy -- see
    distill.py -- and without belief tracking -- see po_policy.py) via
    self-consistent coordinate-ascent policy iteration (see module
    docstring).
        ---Inputs---
            · n, p, p_s, cutoff, tolerance, tolerance_stability, progress,
              savedata: see policy.policy_iteration -- identical meaning.
            · allow_discard: (bool) whether nodes may independently discard
                             their own occupied qubits each time slot. Must
                             be False or True (uncapped) -- a network-wide
                             discard cap can't be respected by independent
                             per-node decisions (see
                             distill._check_local_policy_discard_support,
                             reused here for the same reason).
            · method: (str) how to run each policy evaluation step --
                      "iterative" (default) matches policy.py's own default,
                      Bellman-backup sweeps to a tolerance
                      (_evaluate_joint_policy); "direct" solves each
                      evaluation exactly via sparse LU instead
                      (_evaluate_joint_policy_direct). Discard enabled
                      inflates the per-state branching factor enough that
                      "iterative" can become dramatically slower -- "direct"
                      barely notices -- exactly policy.py's own method
                      parameter, for the same reason.
        ---Outputs---
            · v0_evol, state_info, exe_time: see policy.policy_iteration --
              same shapes, so the result can be saved/loaded/exported through
              the same pipeline (see save_local_optimal_data,
              build_policy_json below)."""
    assert isinstance(n, int) and n >= 2, "n must be an integer >= 2"
    assert 0 <= p <= 1, "p must be between zero and one"
    assert 0 <= p_s <= 1, "p_s must be between zero and one"
    assert isinstance(cutoff, int) and cutoff > 0, "cutoff must be a positive integer"
    assert method in ("iterative", "direct"), 'method must be "iterative" or "direct"'
    _check_local_policy_discard_support(allow_discard)

    parameters = Configuration(p=p, p_s=p_s, cut=cutoff, allow_discard=allow_discard)
    own_qubits_by_node = [node_qubits(n, i) for i in range(n)]
    agent = Agent(n, parameters)
    node_policy = [dict() for _ in range(n)]

    start_time = time.time()
    v0_evol = []
    policy_is_stable = False
    policy_was_stable = False
    tol = tolerance_stability

    while True:
        prev_values = list(agent.values)
        prev_len = len(prev_values)

        if method == "direct":
            v0 = _evaluate_joint_policy_direct(agent, own_qubits_by_node, node_policy, parameters)
        else:
            v0 = _evaluate_joint_policy(agent, own_qubits_by_node, node_policy, parameters, tol, progress)
        v0_evol.append(v0)

        value_delta = 0.0
        for i in range(len(agent.state_list)):
            before = prev_values[i] if i < prev_len else 0.0
            value_delta = max(value_delta, abs(agent.values[i] - before))

        if (policy_is_stable and policy_was_stable) or value_delta < tolerance:
            break

        policy_was_stable = policy_is_stable
        policy_is_stable = True
        tol = tolerance

        visitation = _visitation_counts(agent, own_qubits_by_node, node_policy, parameters)
        changed = _improve_node_policies(agent, node_policy, visitation, own_qubits_by_node, parameters)
        if changed:
            policy_is_stable = False
            tol = tolerance_stability

        if progress:
            print("Local policy improvement: %d states" % len(agent.state_list) + " " * 30, end="\r")

    if progress:
        print(" " * 80, end="\r")

    # Any (node, local_state) table that was lazily created (uniform init)
    # but never touched by policy improvement -- i.e. never visited with
    # positive weight under the final policy -- is force-collapsed to its
    # highest-weight candidate here, so every table is one-hot before export.
    # Since these entries are, by construction, never actually reached by
    # the converged joint policy, this can't change any reported value; it
    # only guarantees the induced action at every discovered state is
    # deterministic (see module docstring's "Output format" section).
    for node_table in node_policy:
        for local_state, table in list(node_table.items()):
            if len(table) > 1:
                best_action = max(table, key=lambda kv: kv[1])[0]
                node_table[local_state] = [(best_action, 1.0)]

    exe_time = time.time() - start_time
    state_info = _build_state_info(agent, own_qubits_by_node, node_policy, parameters)

    if savedata:
        save_local_optimal_data(n, p, p_s, cutoff, tolerance, v0_evol, state_info, exe_time,
                                 allow_discard=allow_discard)

    return v0_evol, state_info, exe_time


# --------------------------------------------------------------------------- #
# -----------------------  EXPORT TO POLICY.PY'S SHAPE  ----------------------- #
# --------------------------------------------------------------------------- #

def _action_key(action, discard_enabled):
    """Hashable, type-normalized form of a raw action (as produced by
    State.get_actions(), a list when discard is disabled) or a combined
    action (as produced by po_environment.combine_actions, a tuple) -- so the
    two can be compared for equality regardless of which one produced it."""
    if discard_enabled:
        swap_nodes, discard_qubits = action
        return (tuple(swap_nodes), tuple(discard_qubits))
    return tuple(action)


def _induced_action(key, own_qubits_by_node, node_policy, discard_enabled):
    """The single deterministic global action every node's own (converged,
    one-hot) table combines to at global state `key`."""
    dist = _joint_action_distribution(key, own_qubits_by_node, node_policy, discard_enabled)
    return max(dist, key=lambda action_prob: action_prob[1])[0]


def _build_state_info(agent, own_qubits_by_node, node_policy, parameters):
    """Build state_info in policy.py's own pre-export shape (see Agent's
    state_info in policy.py: 'state', 'action_space', 'policy', 'value' per
    discovered global state), so it can be handed directly to policy.py's
    own _export_state_info. Terminal (already-delivered) states get the same
    trivial no-op action_space/policy policy.py's own Agent.__init__ uses --
    State.get_actions() isn't consulted for them, since it has no notion of
    "already delivered" and could otherwise return a nontrivial (and
    irrelevant) action space for one."""
    discard_enabled = parameters.discard_enabled
    no_op_action_space = [((), ())] if discard_enabled else [[]]

    state_info = []
    for idx, key in enumerate(agent.state_list):
        if _is_terminal(key):
            state_info.append({"state": key, "action_space": no_op_action_space, "policy": [1.0], "value": 0.0})
            continue

        qubits = key_to_qubits(key)
        raw_actions = State(qubits, parameters).get_actions()
        action = _induced_action(key, own_qubits_by_node, node_policy, discard_enabled)
        action_key = _action_key(action, discard_enabled)
        match = next(i for i, a in enumerate(raw_actions) if _action_key(a, discard_enabled) == action_key)

        policy_vec = [0.0] * len(raw_actions)
        policy_vec[match] = 1.0
        state_info.append({
            "state": key,
            "action_space": raw_actions,
            "policy": policy_vec,
            "value": agent.values[idx],
        })

    return state_info


# --------------------------------------------------------------------------- #
# --------------------------------  I/O  ------------------------------------ #
# --------------------------------------------------------------------------- #

def _localopt_filename(n, p, p_s, cutoff, tolerance, allow_discard=False):
    return 'data_localopt/n%s_p%.3f_ps%.3f_tc%s_tol%s%s_localopt' % (
        n, p, p_s, cutoff, tolerance, _discard_suffix(allow_discard))


def check_local_optimal_data(n, p, p_s, cutoff, tolerance, allow_discard=False):
    """True if local_policy_iteration has already been run and saved for this
    set of parameters."""
    return Path(_localopt_filename(n, p, p_s, cutoff, tolerance, allow_discard)).exists()


def save_local_optimal_data(n, p, p_s, cutoff, tolerance, v0_evol, state_info, exe_time, allow_discard=False):
    """Save local_policy_iteration's output in exactly policy.py's own
    save_policyiter_data pickle shape (see module docstring)."""
    os.makedirs('data_localopt', exist_ok=True)
    filename = _localopt_filename(n, p, p_s, cutoff, tolerance, allow_discard)
    data = {'v0_evol': v0_evol, 'state_info': _export_state_info(state_info, allow_discard),
            'exe_time': exe_time}
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_local_optimal_data(n, p, p_s, cutoff, tolerance, allow_discard=False):
    """Load data saved by save_local_optimal_data / local_policy_iteration."""
    filename = _localopt_filename(n, p, p_s, cutoff, tolerance, allow_discard)
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data['v0_evol'], data['state_info'], data['exe_time']


def _export_action(raw_action, allow_discard):
    """Serialize one raw (qubit-index) action for the JSON handoff. Mirrors
    simulate.py's own _export_action -- duplicated rather than imported to
    keep this pipeline decoupled from simulate.py's subprocess/CLI concerns
    (same rationale po_environment.py gives for duplicating node_qubits)."""
    if not allow_discard:
        return list(raw_action)
    swap_nodes, discard_qubits = raw_action
    return {"swap": list(swap_nodes), "discard": list(discard_qubits)}


def build_policy_json(n, p, p_s, cutoff, tolerance, allow_discard, out_path, method="iterative"):
    """Write the locally-optimal policy (computing it first if necessary) as
    JSON for simulate.jl to consume -- a copy of simulate.py's own
    build_policy_json with only the loader swapped (load_local_optimal_data
    instead of policy.load_policyiter_data), possible because the two
    on-disk pickles share the exact same shape (see module docstring).
    method -- see local_policy_iteration's docstring; irrelevant if the data
    already exists on disk."""
    parameters = Configuration(p=p, p_s=p_s, cut=cutoff, allow_discard=allow_discard)
    if not check_local_optimal_data(n, p, p_s, cutoff, tolerance, allow_discard):
        local_policy_iteration(n, p, p_s, cutoff, tolerance=tolerance, savedata=True,
                                allow_discard=allow_discard, method=method)
    _, state_info, _ = load_local_optimal_data(n, p, p_s, cutoff, tolerance, allow_discard)

    states = []
    for entry in state_info:
        qubits = key_to_qubits(entry["qubits"])
        raw_actions = State(qubits, parameters).get_actions()
        states.append({
            "qubits": entry["qubits"],
            "policy": entry["policy"],
            "actions": [_export_action(a, allow_discard) for a in raw_actions],
        })

    with open(out_path, "w") as handle:
        json.dump({"n": n, "states": states}, handle)


# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Directly optimize a strictly local (memoryless, per-node) policy '
                     'for the qubit-age cutoff model: each node decides its own action '
                     'from only its own current qubit state, via self-consistent '
                     'coordinate-ascent policy iteration -- see this module\'s docstring. '
                     'Takes the same model-defining arguments as policy.py, and saves '
                     'output in the same file format, directly loadable anywhere '
                     'policy.load_policyiter_data\'s return shape is expected.')
    parser.add_argument('--n', type=int, default=3, help='Number of nodes.')
    parser.add_argument('--p', type=float, default=0.8,
                        help='Elementary link generation success probability.')
    parser.add_argument('--p_s', type=float, default=0.8,
                        help='Entanglement swap success probability.')
    parser.add_argument('--cutoff', type=int, default=5, help='Qubit-age cutoff.')
    parser.add_argument('--tol', type=float, default=1e-5, help='Value tolerance.')
    parser.add_argument('--discard', nargs='?', const=True, default=False, type=int,
                        help='Allow each node to independently discard its own occupied '
                             'qubits each time slot. Must be omitted (off) or passed with '
                             'no number (uncapped) -- a network-wide discard cap can\'t be '
                             'respected by independent per-node decisions.')
    parser.add_argument('--method', choices=['iterative', 'direct'], default='iterative',
                        help='How to run the underlying policy-evaluation steps: '
                             '"iterative" (default) matches policy.py\'s own default; '
                             '"direct" solves each policy evaluation exactly via sparse LU '
                             'instead of Bellman-backup sweeps, which is substantially '
                             'faster once --discard inflates the state/action space. See '
                             'local_policy_iteration\'s docstring.')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output.')
    args = parser.parse_args()

    if check_local_optimal_data(args.n, args.p, args.p_s, args.cutoff, args.tol, args.discard):
        print('Data already exists!')
    else:
        v0_evol, state_info, exe_time = local_policy_iteration(
            args.n, args.p, args.p_s, args.cutoff,
            tolerance=args.tol, progress=not args.quiet, savedata=True,
            allow_discard=args.discard, method=args.method)
        print('Done in %.1fs! %d states. Expected delivery time: %.4f time slots.' % (
            exe_time, len(state_info), -(state_info[0]['value'] + 1)))
