"""Self-consistent, decentralized POMDP policy iteration for the qubit-age
cutoff model (see environment.py, policy.py).

Motivation
----------
policy_iteration in policy.py finds the optimal swap policy assuming a
central controller that observes the full chain state every time slot.
distill.py explores one way to make that policy realistic for a real
repeater network -- where no single node can see the whole chain -- by
DISTILLING the already-solved global policy down to independent per-node
tables (majority-vote over every global state consistent with a node's own
qubits).

This module instead SOLVES the decentralized problem directly. Each node
knows only the (age, hanging) of its own one or two qubits, exactly as in
distill.py, but additionally maintains a running BELIEF -- a probability
distribution over the full global state, Bayesian-filtered from what it has
personally observed (see po_environment.py) -- and its decision rule is a
function of that belief, not just of the bare current local observation.
This is a genuine (finite, cooperative) decentralized-POMDP planning
problem, solved here by policy iteration over an augmented "hyper-state"

    (true global state, node 0's belief, node 1's belief, ..., node n-1's belief)

-- see HyperAgent -- mirroring the structure of policy.py's own
Agent/policy_iteration (uniform-random initial policy, alternating
evaluation/improvement, dual stability/value convergence check), just with
a much larger per-step state, and with the "policy" now being n separate
per-node tables (belief -> local action) rather than one table per global
state.

Self-consistency, and why it doesn't require modeling nested beliefs
----------------------------------------------------------------------
A node's belief update needs to know what action was actually taken by
*every* node (its own qubits' fate depends on the full joint action, not
just its own local piece) -- but in real decentralized execution, no node
observes anyone else's action either. Naively this seems to demand an
infinite regress (node i must reason about node j's belief about node k's
belief about ...), which is exactly why Dec-POMDPs are worst-case
intractable.

This module sidesteps that regress because policy iteration here is run
OFFLINE by a planner with full visibility into the hyper-state -- including
every node's *actual* belief, not just its own. So at every hyper-state, the
joint action actually taken is simply "each node's policy applied to its own
actual current belief" (see _joint_action_distribution): no hypothesizing
about anyone else's belief is required, because the planner already has it.
Each node's own belief update (po_environment.predict_and_filter) then
reuses that SAME already-determined joint action across every state in its
belief's support -- exactly the standard Bayes-filter equation
b'(s') ~ sum_s b(s) P(s'|s,a) O(o|s'), just noting that `a` is a fact fixed
by the hyper-state here, not a per-hypothesis unknown. The policy that falls
out of this offline solve is genuinely executable online by each node
independently afterwards: the belief-update rule and the belief -> action
table are both precomputed, so a real node only ever needs its own
observation history, never anyone else's.

The "self-consistent" part is that every node's decision rule is improved
against the SAME evolving node_policy that determines everyone's behavior
(including its own, and including how every OTHER node's belief evolves) --
so at a fixed point, every node's rule is a best response to the rules
(including its own) that produced the hyper-state distribution it was
evaluated against: a Nash/Markov-perfect-style equilibrium for this
cooperative (shared-reward) team, found by a coordinate-ascent-like policy
iteration. As in policy.py, several actions can be numerically tied in
Q-value at some hyper-states, which is why the same dual
stability-or-value-tolerance stopping rule is used (see policy_iteration's
docstring there for why either check alone can be fooled).

The one further approximation beyond decentralization/self-consistency:
belief rounding
-----------------
A node left untouched for arbitrarily long keeps refining -- never exactly
repeating -- its belief about the rest of the chain, so the set of exactly
reachable beliefs is not generally finite. To keep the hyper-state space (and
hence this solve) computable, beliefs are rounded and small entries pruned
before being used as a lookup key (see po_environment.belief_key,
belief_precision/prob_floor below). This is the only place this module
trades exactness for tractability; everything else here is an exact solve of
the (rounded) POMDP it defines, in the same spirit as policy.py's exact
solves of the MDP it defines. max_hyperstates guards against the hyper-state
space blowing up past what's practical for a given n/cutoff/precision.
"""
from collections import defaultdict
from itertools import product
from pathlib import Path
import argparse
import os
import pickle
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from environment import Configuration, State
from policy import cached_step, qubits_to_key, _is_terminal
from po_environment import (
    belief_from_key,
    belief_key,
    belief_observation,
    combine_actions,
    initial_belief_key,
    local_actions,
    node_qubits,
    observation_of,
    predict_and_filter,
)

os.chdir(Path(__file__).resolve().parent)


def _check_pomdp_discard_support(allow_discard):
    """Independent per-node decisions can't respect a network-wide discard
    cap (no node knows whether another is also discarding this tick) --
    exactly distill.py's _check_local_policy_discard_support, for the same
    reason. Only off (False) or uncapped (True) are supported."""
    if allow_discard not in (False, True):
        raise ValueError(
            "the POMDP pipeline doesn't support capped discard (allow_discard=%r): "
            "independent per-node decisions can't respect a network-wide cap. Use "
            "allow_discard=False or True (uncapped) instead." % (allow_discard,))


def _discard_suffix(allow_discard):
    return '_discard' if allow_discard else ''


# --------------------------------------------------------------------------- #
# --------------------------  JOINT ACTION ASSEMBLY  ------------------------- #
# --------------------------------------------------------------------------- #

def _node_table(node_policy, node, bkey, own_qubits, discard_enabled):
    """The candidate-local-action distribution node `node` currently uses at
    belief `bkey` -- a list of (local_action, probability) pairs, lazily
    initialized to uniform-random over every locally valid candidate the
    first time this belief is encountered (mirrors policy.py's Agent.observe
    / init_policy: exploring every reachable choice before any improvement
    step has run)."""
    table = node_policy[node].get(bkey)
    if table is None:
        obs = belief_observation(bkey, own_qubits)
        candidates = local_actions(obs, own_qubits, discard_enabled)
        table = [(a, 1.0 / len(candidates)) for a in candidates]
        node_policy[node][bkey] = table
    return table


def _joint_action_distribution(n, own_qubits_by_node, betas, node_policy, discard_enabled, override=None):
    """Every (global action, probability) pair reachable by each node
    independently sampling its own local action from its own current table
    at its own current belief -- the joint of n independent per-node
    choices, exactly as distill.py's _joint_action_distribution, but keyed
    on belief rather than bare local state.

    `override`, if given, is a (node, local_action) pin: every other node
    still uses its own current table, but `node`'s choice is fixed to
    `local_action` at probability 1. Policy improvement uses this to ask "if
    this node deviated to this candidate action, holding every other node's
    behavior fixed at its current policy, what would happen?" -- the
    single-agent best-response step that drives the coordinate-ascent-style
    self-consistency loop (see module docstring)."""
    per_node = []
    for j in range(n):
        if override is not None and override[0] == j:
            per_node.append([(override[1], 1.0)])
        else:
            per_node.append(_node_table(node_policy, j, betas[j], own_qubits_by_node[j], discard_enabled))

    combined = defaultdict(float)
    for combo in product(*per_node):
        prob = 1.0
        actions = []
        for a, p in combo:
            actions.append(a)
            prob *= p
        if prob == 0.0:
            continue
        action = combine_actions(own_qubits_by_node, actions, discard_enabled)
        combined[action] += prob

    return list(combined.items())


# --------------------------------------------------------------------------- #
# ------------------------------  HYPER-STATE  ------------------------------- #
# --------------------------------------------------------------------------- #

def hyperstate_step(n, own_qubits_by_node, parameters, s_key, betas, action, precision, prob_floor):
    """Advance one hyper-state transition given the joint `action` actually
    taken: branches over every possible true-state outcome (via policy.py's
    cached_step), and for each branch, updates every node's belief by
    predict_and_filter-ing its prior belief through this same `action` and
    picking out the branch matching the observation the actual next state
    produces (see po_environment.predict_and_filter's docstring -- this
    module always knows the true next state exactly, so it never needs to
    sum over "which observation would occur," only look one up).

    Returns a list of (next_hyperstate, probability) pairs."""
    next_states, probs, _ = cached_step(parameters, s_key, action)
    node_predictions = [
        predict_and_filter(belief_from_key(betas[i]), own_qubits_by_node[i], action, parameters)
        for i in range(n)
    ]

    results = []
    for s_next, p in zip(next_states, probs):
        if p == 0.0:
            continue
        new_betas = []
        for i in range(n):
            obs = observation_of(s_next, own_qubits_by_node[i])
            _, posterior = node_predictions[i][obs]
            new_betas.append(belief_key(posterior, precision, prob_floor, keep=s_next))
        results.append(((s_next, tuple(new_betas)), p))
    return results


class HyperAgent:
    """Bookkeeping for hyper-state policy iteration: every distinct
    (true state, per-node beliefs) tuple discovered so far, and its current
    value estimate. Mirrors policy.py's Agent, but doesn't store an
    action_space/policy per hyper-state -- the joint action distribution is
    instead derived on the fly from the much smaller per-node node_policy
    tables (see _joint_action_distribution), since candidate local actions
    are cheap to re-derive from a belief's observation. Value convention
    matches policy.py: expected remaining time slots to delivery is
    -(value + 1)."""

    def __init__(self, n, parameters):
        s0 = qubits_to_key(State.initial(n, parameters).qubits)
        b0 = initial_belief_key(n, parameters)
        h0 = (s0, tuple(b0 for _ in range(n)))
        self._index = {h0: 0}
        self.state_list = [h0]
        self.values = [0.0]

    def observe(self, h):
        idx = self._index.get(h)
        if idx is None:
            idx = len(self.state_list)
            self._index[h] = idx
            self.state_list.append(h)
            self.values.append(0.0)
        return idx


# --------------------------------------------------------------------------- #
# ---------------------------  POLICY EVALUATION  ----------------------------- #
# --------------------------------------------------------------------------- #

def _evaluate_hyperstates(agent, node_policy, n, own_qubits_by_node, parameters, tol, precision, prob_floor,
                           max_hyperstates, progress):
    """Evaluate the current node_policy tables by successive approximation
    over the hyper-state graph -- exactly policy.py's
    _evaluate_policy_iterative, just with hyperstate_step / a HyperAgent in
    place of policy.py's cached_step / Agent."""
    v0 = []
    error = np.inf
    while error > tol:
        error = 0.0
        idx = 0
        while idx < len(agent.state_list):
            if len(agent.state_list) > max_hyperstates:
                raise RuntimeError(
                    "POMDP hyper-state space exceeded max_hyperstates=%d; try a coarser "
                    "belief_precision/prob_floor or a smaller n/cutoff." % max_hyperstates)

            s_key, betas = agent.state_list[idx]
            if _is_terminal(s_key):
                idx += 1
                continue

            value = agent.values[idx]
            v = 0.0
            for action, action_prob in _joint_action_distribution(
                    n, own_qubits_by_node, betas, node_policy, parameters.discard_enabled):
                if action_prob == 0.0:
                    continue
                for next_h, p in hyperstate_step(n, own_qubits_by_node, parameters, s_key, betas, action,
                                                  precision, prob_floor):
                    h_idx = agent.observe(next_h)
                    v += action_prob * p * (-1 + agent.values[h_idx])

            error = max(error, abs(value - v))
            agent.values[idx] = v
            idx += 1

        v0.append(agent.values[0])
        if progress:
            print("POMDP policy eval.: error = %.2e > %.2e = tolerance, %d hyperstates" % (
                error, tol, len(agent.state_list)) + " " * 10, end="\r")
    return v0


# --------------------------------------------------------------------------- #
# -----------------------  HYPER-STATE VISITATION  ---------------------------- #
# --------------------------------------------------------------------------- #

def _hyperstate_visitation_counts(agent, node_policy, n, own_qubits_by_node, parameters, precision, prob_floor):
    """Expected number of times each known hyper-state is visited, starting
    from the initial hyper-state, under the current node_policy tables --
    the same transposed-linear-system trick as distill.py's
    _visitation_counts, applied to the hyper-state graph instead of the
    global-state graph. Used to weight policy improvement so that a
    (node, belief) pair encountered across many hyper-states is optimized
    against how often each of those hyper-states actually arises, not
    counted once per hyper-state regardless of how (im)probable it is."""
    state_list = agent.state_list
    index = {h: i for i, h in enumerate(state_list)}
    coeffs = {}

    for i, (s_key, betas) in enumerate(state_list):
        if _is_terminal(s_key):
            continue
        for action, action_prob in _joint_action_distribution(
                n, own_qubits_by_node, betas, node_policy, parameters.discard_enabled):
            if action_prob == 0.0:
                continue
            for next_h, p in hyperstate_step(n, own_qubits_by_node, parameters, s_key, betas, action,
                                              precision, prob_floor):
                j = index.get(next_h)
                if j is None:
                    # Not discovered during evaluation (can happen right after a policy
                    # improvement round changes behavior); treated as zero visitation
                    # for now -- it'll be discovered and get its own weight once the
                    # next evaluation sweep runs.
                    continue
                if not _is_terminal(next_h[0]):
                    coeffs[(i, j)] = coeffs.get((i, j), 0.0) - action_prob * p

    non_terminal = [i for i, (s, _b) in enumerate(state_list) if not _is_terminal(s)]
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

    counts = [0.0] * len(state_list)
    for idx in non_terminal:
        counts[idx] = float(x[row_of[idx]])
    return counts


# --------------------------------------------------------------------------- #
# ---------------------------  POLICY IMPROVEMENT  ---------------------------- #
# --------------------------------------------------------------------------- #

def _improve_node_policies(agent, node_policy, visitation, n, own_qubits_by_node, parameters, precision, prob_floor):
    """One policy-improvement round: for every (node, belief) pair touched by
    a visited, non-terminal hyper-state, compute the visitation-weighted
    expected Q-value of every locally valid candidate action for that node
    (holding every other node's behavior fixed at its own current table --
    see _joint_action_distribution's `override`), and set that node's table
    at that belief to the single best candidate (one-hot, matching
    policy.py's converged-policy convention).

    A (node, belief) pair can appear in several distinct hyper-states
    (different true states, or different other-node beliefs, paired with
    the same belief for this node); visitation weights those hyper-states'
    contributions by how often the current policy actually reaches them,
    so the improvement step is a genuine expected-value maximization, not an
    unweighted average across however many hyper-states happen to share it.

    Returns True if any (node, belief) table changed -- the "not yet stable"
    signal pomdp_policy_iteration uses to decide whether another
    evaluation/improvement round is needed."""
    discard_enabled = parameters.discard_enabled
    q_accum = defaultdict(float)
    touched = set()

    for idx in range(len(agent.state_list)):
        s_key, betas = agent.state_list[idx]
        if _is_terminal(s_key):
            continue
        w = visitation[idx]
        if w <= 0.0:
            continue

        for i in range(n):
            bkey = betas[i]
            obs = belief_observation(bkey, own_qubits_by_node[i])
            candidates = local_actions(obs, own_qubits_by_node[i], discard_enabled)
            if len(candidates) < 2:
                continue
            touched.add((i, bkey))

            for a_i in candidates:
                joint_dist = _joint_action_distribution(
                    n, own_qubits_by_node, betas, node_policy, discard_enabled, override=(i, a_i))
                for action, action_prob in joint_dist:
                    if action_prob == 0.0:
                        continue
                    for next_h, p in hyperstate_step(n, own_qubits_by_node, parameters, s_key, betas, action,
                                                      precision, prob_floor):
                        h_idx = agent.observe(next_h)
                        q_accum[(i, bkey, a_i)] += w * action_prob * p * (-1 + agent.values[h_idx])

    changed = False
    for (i, bkey) in touched:
        obs = belief_observation(bkey, own_qubits_by_node[i])
        candidates = local_actions(obs, own_qubits_by_node[i], discard_enabled)
        qs = [q_accum[(i, bkey, a)] for a in candidates]
        best = int(np.argmax(qs))
        current = node_policy[i].get(bkey)
        is_onehot_best = current is not None and len(current) == 1 and current[0][0] == candidates[best]
        if not is_onehot_best:
            changed = True
        node_policy[i][bkey] = [(candidates[best], 1.0)]

    return changed


# --------------------------------------------------------------------------- #
# ---------------------------  POLICY ITERATION  ------------------------------ #
# --------------------------------------------------------------------------- #

def pomdp_policy_iteration(n, p, p_s, cutoff, tolerance=1e-5, tolerance_stability=1e-1,
                            belief_precision=3, prob_floor=1e-3, max_hyperstates=20000,
                            progress=True, savedata=True, allow_discard=False):
    """Find the self-consistent, decentralized (belief-conditioned) swap
    policy for an n-node chain under the qubit-age cutoff model -- see the
    module docstring for the planning algorithm and its approximations.
        ---Inputs---
            · n, p, p_s, cutoff, tolerance, tolerance_stability, allow_discard:
              see policy.policy_iteration -- identical meaning, applied to
              the hyper-state graph instead of the plain global-state graph.
              allow_discard must be False or True (uncapped) -- see
              _check_pomdp_discard_support.
            · belief_precision: (int) decimal places a belief is rounded to
              before being treated as a distinct belief-state (see
              po_environment.belief_key). Bounds an otherwise-unbounded
              belief space; coarser (smaller) values keep the hyper-state
              graph smaller at the cost of conflating similar beliefs.
            · prob_floor: (float) belief-support entries at or below this
              probability are pruned (renormalizing the rest) before the
              belief is used as a lookup key -- same role as distill.py's
              soft-distillation prob_floor.
            · max_hyperstates: (int) safety cap -- raises RuntimeError rather
              than silently running away if the hyper-state graph grows past
              this, since a genuinely unbounded belief space is always a
              possibility for larger n/cutoff.
            · progress, savedata: see policy.policy_iteration.
        ---Outputs---
            · v0_evol: (list of lists) evolution of the initial hyper-state's
              value across policy evaluation sweeps, one inner list per
              outer (evaluate, improve) round.
            · node_policy: (list of n dicts) node_policy[i] maps a belief_key
              (po_environment.belief_key's canonical form) to a one-hot
              [(local_action, 1.0)] list once converged -- this is the
              artifact a real, decentralized node would execute: at each
              time slot, update its belief (po_environment.predict_and_filter)
              from its own new observation, round it the same way
              (belief_precision/prob_floor), and look up its action here.
            · exe_time: (float) execution time in seconds."""
    assert isinstance(n, int) and n >= 2, "n must be an integer >= 2"
    assert 0 <= p <= 1, "p must be between zero and one"
    assert 0 <= p_s <= 1, "p_s must be between zero and one"
    assert isinstance(cutoff, int) and cutoff > 0, "cutoff must be a positive integer"
    _check_pomdp_discard_support(allow_discard)

    parameters = Configuration(p=p, p_s=p_s, cut=cutoff, allow_discard=allow_discard)
    own_qubits_by_node = [node_qubits(n, i) for i in range(n)]
    agent = HyperAgent(n, parameters)
    node_policy = [dict() for _ in range(n)]

    start_time = time.time()
    v0_evol = []
    policy_is_stable = False
    policy_was_stable = False
    tol = tolerance_stability

    while True:
        prev_values = list(agent.values)
        prev_len = len(prev_values)

        v0 = _evaluate_hyperstates(agent, node_policy, n, own_qubits_by_node, parameters, tol,
                                    belief_precision, prob_floor, max_hyperstates, progress)
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

        if len(agent.state_list) > max_hyperstates:
            raise RuntimeError(
                "POMDP hyper-state space exceeded max_hyperstates=%d; try a coarser "
                "belief_precision/prob_floor or a smaller n/cutoff." % max_hyperstates)

        visitation = _hyperstate_visitation_counts(agent, node_policy, n, own_qubits_by_node, parameters,
                                                     belief_precision, prob_floor)
        changed = _improve_node_policies(agent, node_policy, visitation, n, own_qubits_by_node, parameters,
                                          belief_precision, prob_floor)
        if changed:
            policy_is_stable = False
            tol = tolerance_stability

        if progress:
            print("POMDP policy improvement: %d hyperstates" % len(agent.state_list) + " " * 30, end="\r")

    if progress:
        print(" " * 80, end="\r")

    exe_time = time.time() - start_time

    if savedata:
        save_pomdppolicy_data(n, p, p_s, cutoff, tolerance, belief_precision, prob_floor, v0_evol, node_policy,
                               exe_time, allow_discard=allow_discard)

    return v0_evol, node_policy, exe_time


# --------------------------------------------------------------------------- #
# --------------------------------  I/O  ------------------------------------ #
# --------------------------------------------------------------------------- #

def _pomdppolicy_filename(n, p, p_s, cutoff, tolerance, belief_precision, prob_floor, allow_discard=False):
    return 'data_pomdppolicy/n%s_p%.3f_ps%.3f_tc%s_tol%s_bp%d_pf%s%s' % (
        n, p, p_s, cutoff, tolerance, belief_precision, prob_floor, _discard_suffix(allow_discard))


def check_pomdppolicy_data(n, p, p_s, cutoff, tolerance, belief_precision, prob_floor, allow_discard=False):
    """True if POMDP policy iteration has already been run and saved for this
    set of parameters."""
    return Path(_pomdppolicy_filename(n, p, p_s, cutoff, tolerance, belief_precision, prob_floor,
                                       allow_discard)).exists()


def save_pomdppolicy_data(n, p, p_s, cutoff, tolerance, belief_precision, prob_floor, v0_evol, node_policy,
                           exe_time, allow_discard=False):
    """Save POMDP policy iteration data for this set of parameters."""
    os.makedirs('data_pomdppolicy', exist_ok=True)
    filename = _pomdppolicy_filename(n, p, p_s, cutoff, tolerance, belief_precision, prob_floor, allow_discard)
    data = {
        'n': n, 'v0_evol': v0_evol, 'node_policy': node_policy, 'exe_time': exe_time,
        'belief_precision': belief_precision, 'prob_floor': prob_floor,
    }
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pomdppolicy_data(n, p, p_s, cutoff, tolerance, belief_precision, prob_floor, allow_discard=False):
    """Load POMDP policy iteration data saved by save_pomdppolicy_data /
    pomdp_policy_iteration."""
    filename = _pomdppolicy_filename(n, p, p_s, cutoff, tolerance, belief_precision, prob_floor, allow_discard)
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data['v0_evol'], data['node_policy'], data['exe_time']


# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Self-consistent, decentralized POMDP policy iteration for the '
                     'qubit-age cutoff model: each node conditions its own local action '
                     'on a Bayesian belief over the full chain state, rather than on the '
                     'full state itself (policy.py) or on its bare local state alone '
                     '(distill.py). See this module\'s docstring for the algorithm.')
    parser.add_argument('--n', type=int, default=3, help='Number of nodes.')
    parser.add_argument('--p', type=float, default=0.8,
                        help='Elementary link generation success probability.')
    parser.add_argument('--p_s', type=float, default=0.8,
                        help='Entanglement swap success probability.')
    parser.add_argument('--cutoff', type=int, default=5, help='Qubit-age cutoff.')
    parser.add_argument('--tol', type=float, default=1e-5, help='Value tolerance.')
    parser.add_argument('--belief-precision', type=int, default=3,
                        help='Decimal places beliefs are rounded to before being treated as '
                             'distinct belief-states -- bounds the otherwise-unbounded belief '
                             'space. Smaller is coarser/faster but conflates more beliefs.')
    parser.add_argument('--prob-floor', type=float, default=1e-3,
                        help='Prune belief-support entries at or below this probability '
                             '(renormalizing what remains) before keying on the belief.')
    parser.add_argument('--max-hyperstates', type=int, default=20000,
                        help='Safety cap on the number of distinct hyper-states discovered.')
    parser.add_argument('--discard', action='store_true',
                        help='Allow each node to independently discard its own occupied '
                             'qubits (uncapped only -- a shared cap can\'t be respected by '
                             'independent nodes, see distill.py).')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output.')
    args = parser.parse_args()

    if check_pomdppolicy_data(args.n, args.p, args.p_s, args.cutoff, args.tol,
                              args.belief_precision, args.prob_floor, args.discard):
        print('Data already exists!')
    else:
        v0_evol, node_policy, exe_time = pomdp_policy_iteration(
            args.n, args.p, args.p_s, args.cutoff, tolerance=args.tol,
            belief_precision=args.belief_precision, prob_floor=args.prob_floor,
            max_hyperstates=args.max_hyperstates, progress=not args.quiet, savedata=True,
            allow_discard=args.discard)
        counts = ', '.join('node %d: %d belief-states' % (i, len(t)) for i, t in enumerate(node_policy))
        print('Done in %.1fs! Expected delivery time: %.4f time slots. %s' % (
            exe_time, -(v0_evol[-1][-1] + 1), counts))
