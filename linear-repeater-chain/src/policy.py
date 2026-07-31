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
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import argparse
import os
import pickle
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

def step(parameters, qubits, action):
    """Advance one full time slot from a decision state (the state just
    before an action is applied), given `action`. Composition matches the
    paper's Environment.step: apply the action (swap, and discard if
    parameters.discard_enabled), apply cutoffs, age the qubits, then attempt
    entanglement generation -- unless the action already produced an
    end-to-end link, in which case the chain is delivered and evolution
    stops there.

    Returns (next_states, probabilities, action_spaces): one entry per
    possible outcome, each next_state a qubit list and each action_space the
    valid actions from that next state (a single no-op action for a
    delivered/terminal state, in whichever shape get_actions() uses for
    this Configuration).
    """
    swap_states, swap_probs = State(deepcopy(qubits), parameters) \
        .get_transitions_for_action(action)

    no_op_action_space = [((), ())] if parameters.discard_enabled else [[]]

    next_states, probs, action_spaces = [], [], []
    for s_qubits, p in zip(swap_states, swap_probs):
        if p == 0.0:
            continue

        post = State(s_qubits, parameters)
        if post.check_e2e_link():
            next_states.append(s_qubits)
            probs.append(p)
            action_spaces.append(no_op_action_space)
            continue

        post.cutoff()
        post.age_qubits()
        gen_states, gen_probs = post.get_entanglement_generation_transitions()
        for g_qubits, gp in zip(gen_states, gen_probs):
            if gp == 0.0:
                continue
            next_states.append(g_qubits)
            probs.append(p * gp)
            action_spaces.append(State(g_qubits, parameters).get_actions())

    return next_states, probs, action_spaces


@lru_cache(maxsize=None)
def _is_terminal(qubits_key):
    """Whether a state (given by its qubit-age key) already has an
    end-to-end link. Depends only on link structure, not on parameters, so
    it's cached independent of (p, p_s, cutoff)."""
    return State(key_to_qubits(qubits_key), None).check_e2e_link()


@lru_cache(maxsize=None)
def _cached_step(p, p_s, cutoff, allow_discard, qubits_key, action):
    """Memoized transition lookup, keyed on hashable (state, action,
    parameters). The transition structure out of a given state under a given
    action never changes across policy-evaluation sweeps -- only the value
    estimates propagating through it do -- so recomputing it every sweep
    (deepcopy-ing qubit lists and re-expanding every success/failure
    combination) is pure waste. This turns that into "compute once per
    (state, action) the first time it's visited, then reuse." allow_discard
    is part of the key (not just baked into a shared Configuration) because
    it changes both the action shape and the transition semantics for the
    same (p, p_s, cutoff).
    Returns (next_keys, probs, action_spaces) where next_keys are qubit-age
    keys and action_spaces are tuples of tuples (both hashable, unlike the
    Qubit-list/list-of-lists shapes step() returns)."""
    parameters = Configuration(p=p, p_s=p_s, cut=cutoff, allow_discard=allow_discard)
    next_states, probs, action_spaces = step(parameters, key_to_qubits(qubits_key), list(action))
    next_keys = tuple(qubits_to_key(s) for s in next_states)
    action_spaces = tuple(tuple(tuple(a) for a in a_space) for a_space in action_spaces)
    return next_keys, tuple(probs), action_spaces


def cached_step(parameters, qubits_key, action):
    """Convenience wrapper around _cached_step taking a Configuration and a
    qubit-age key directly."""
    return _cached_step(parameters.p, parameters.p_s, parameters.cut,
                         parameters.allow_discard, qubits_key, tuple(action))


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
    parser.add_argument('--policy', choices=['optimal', 'swap-asap'], default='optimal',
                        help='"optimal" (policy iteration) or "swap-asap" (baseline).')
    parser.add_argument('--discard', nargs='?', const=True, default=False, type=int,
                        help='Allow the agent to discard occupied qubits each time slot. '
                             'Pass a number to cap how many qubits may be discarded per '
                             'time slot (e.g. --discard 1); with no number, there is no cap.')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output.')
    args = parser.parse_args()

    if args.policy == 'optimal':
        if check_policyiter_data(args.n, args.p, args.p_s, args.cutoff, args.tol, args.discard):
            print('Data already exists!')
        else:
            v0_evol, state_info, exe_time = policy_iteration(
                args.n, args.p, args.p_s, args.cutoff,
                tolerance=args.tol, progress=not args.quiet, savedata=True,
                allow_discard=args.discard)
            print('Done in %.1fs! %d states. Expected delivery time: %.4f time slots.' % (
                exe_time, len(state_info), -(state_info[0]['value'] + 1)))
    else:
        if check_swapasap_data(args.n, args.p, args.p_s, args.cutoff, args.tol, args.discard):
            print('Data already exists!')
        else:
            v0_evol, state_info, exe_time = policy_eval_swapasap(
                args.n, args.p, args.p_s, args.cutoff,
                tolerance=args.tol, progress=not args.quiet, savedata=True,
                allow_discard=args.discard)
            print('Done in %.1fs! %d states. Expected delivery time: %.4f time slots.' % (
                exe_time, len(state_info), -(state_info[0]['value'] + 1)))
