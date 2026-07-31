"""Distillation of a strictly local, per-node policy from the global-knowledge
policy computed by policy.py.

policy_iteration's policy conditions each decision on the full chain state
(every qubit's age/hanging, network-wide) -- unrealistic for a real repeater
network, where each node only observes the qubits it physically holds. This
module distills that global policy into n independent per-node tables, one
per node, each keyed only on that node's own qubit(s). Every local table
uses the same representation regardless of which method built it: for a
local state, a dict mapping candidate local actions to the probability of
taking them (always summing to 1). A degenerate one-entry dict is simply a
deterministic local action.

Two distillation methods are provided:

- distill_local_policy (greedy): for a node and local state, collect every
  global state consistent with it, look at the (one-hot, once converged --
  see policy.py's Agent docstring) action each prescribes, decompose it into
  that node's share, and keep whichever local action was prescribed most
  often -- one vote per distinct global state. This is simple but has a
  systematic bias: swap-asap is the exactly-optimal action in the large
  majority of enumerated global states (states deep enough into a trajectory
  that waiting can't help), so it dominates the vote almost everywhere,
  drowning out the comparatively rare states where the global policy is
  actually doing something non-trivial (delaying a swap). The resulting
  local policy ends up close to swap-asap itself.

- distill_local_policy_soft (probabilistic): addresses that bias differently.
  Every matching global state still casts a single hard vote for its own
  recorded optimal action (same as the greedy method), but two things
  change. First, each state's vote is weighted by its expected visitation
  count under the optimal policy (starting from the chain's initial state),
  not by 1 -- so a node's local distribution reflects how often a local
  state is actually encountered in practice, not how many combinatorially-
  distinct global microstates happen to share it (visitation_weighted=False
  restores the plain one-vote-per-state count instead). Second, the
  resulting weighted-majority proportions are temperature-scaled:
  P(action) ∝ proportion(action)^(1/temperature) -- temperature=1 leaves
  the honest weighted-vote proportions alone, temperature->0 collapses to a
  hard pick of the majority action (recovering distill_local_policy's own
  vote, just visitation- rather than count-weighted), and temperature->inf
  flattens toward uniform over whichever actions appear at all. This
  softens or sharpens *disagreement between states* directly, rather than
  each state's own individual confidence (which, being a converged one-hot
  policy, has no notion of "confidence" to begin with -- every state is
  already maximally confident in its own recorded action).
"""
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import argparse
import json
import os
import pickle

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from environment import Configuration, State
from policy import (
    _discard_suffix,
    _is_terminal,
    _policyiter_filename,
    cached_step,
    check_policyiter_data,
    key_to_qubits,
    load_policyiter_data,
    policy_iteration,
    qubits_to_key,
)

SRC_DIR = Path(__file__).resolve().parent
# See simulate.py: anchors relative data_policyiter/data_localpolicy paths to
# this module's location regardless of the caller's CWD.
os.chdir(SRC_DIR)


def _check_local_policy_discard_support(allow_discard):
    """Local/distilled execution assembles a global action from every node's
    own independently-made choice (see _joint_action_distribution / node
    ownership never overlapping), with no cross-node coordination at all --
    that's the entire premise of "strictly local". A network-wide discard
    *cap* (allow_discard as an int) is a constraint on the sum across every
    node's choices, which independent nodes fundamentally cannot enforce on
    their own (neither node knows whether the other is also discarding this
    tick), and two nodes discarding at once can jointly land in a state
    nothing in this pipeline has an entry for. So it's unsupported here --
    only off (False) or uncapped (True), where every node's own choice is
    already valid on its own, are."""
    if allow_discard is False or allow_discard is True:
        return
    raise ValueError(
        "local/distilled policies don't support capped discard (allow_discard=%r): "
        "independent per-node decisions can't respect a network-wide cap. Use "
        "allow_discard=False or allow_discard=True (uncapped) instead." % (allow_discard,))


# --------------------------------------------------------------------------- #
# --------------------------  NODE / LOCAL STATE  ---------------------------- #
# --------------------------------------------------------------------------- #

def node_qubits(n, node):
    """Flat qubit indices (0-indexed, environment.py's convention) owned by
    `node` (0-indexed): a single qubit for the two end nodes, two qubits
    (left, right) for every middle node."""
    if node == 0:
        return (0,)
    if node == n - 1:
        return (2 * n - 3,)
    return (2 * node - 1, 2 * node)


def local_state_key(qubits, own_qubits):
    """A node's local state: (age, hanging) for each qubit it owns, in
    ascending flat-index order."""
    return tuple((qubits[q].age, qubits[q].hanging) for q in own_qubits)


def decompose_action(action, own_qubits, discard_enabled):
    """A node's share of a global action: whether it performs its own swap
    (only possible for a middle node, whose own left qubit is exactly the
    swap-action index it would appear under -- see State.get_actions), and
    which of its own qubits (if any) it discards."""
    if discard_enabled:
        swap_nodes, discard_qubits = action
    else:
        swap_nodes, discard_qubits = action, ()
    swap = len(own_qubits) == 2 and own_qubits[0] in swap_nodes
    discard = tuple(sorted(q for q in discard_qubits if q in own_qubits))
    return (swap, discard)


# --------------------------------------------------------------------------- #
# ------------------------------  DISTILLATION  ------------------------------ #
# --------------------------------------------------------------------------- #

def distill_local_policy(n, state_info, parameters):
    """Greedy majority-vote distillation. Returns a list of n dicts (one per
    node, 0-indexed), each mapping a local state (see local_state_key) to a
    single deterministic local action (see module docstring for the shared
    {action: probability} representation) -- whichever local action was
    prescribed most often across every global state consistent with that
    local state."""
    own_qubits_by_node = [node_qubits(n, node) for node in range(n)]
    tallies = [defaultdict(Counter) for _ in range(n)]

    for entry in state_info:
        qubits = key_to_qubits(entry["qubits"])
        state = State(qubits, parameters)
        if state.check_e2e_link():
            # Never consulted in simulation: run_episode! only looks up the
            # policy for a state reached at the top of its loop, which is
            # always non-terminal by construction (see simulate.jl).
            continue

        raw_actions = state.get_actions()
        chosen_action = raw_actions[int(np.argmax(entry["policy"]))]

        for node, own_qubits in enumerate(own_qubits_by_node):
            local_state = local_state_key(qubits, own_qubits)
            local_action = decompose_action(chosen_action, own_qubits, parameters.discard_enabled)
            tallies[node][local_state][local_action] += 1

    return [
        {local_state: {counter.most_common(1)[0][0]: 1.0} for local_state, counter in node_tally.items()}
        for node_tally in tallies
    ]


# --------------------------------------------------------------------------- #
# ---------------------  VISITATION, SOFT DISTILLATION  ---------------------- #
# --------------------------------------------------------------------------- #

def _temperature_scale(proportions, temperature):
    """Temperature-scale a categorical distribution `proportions` (a dict of
    value -> probability, summing to 1): P(v) ∝ proportions[v]^(1/temperature).
    Scale-invariant -- unlike scaling raw un-normalized weights, temperature
    means the same thing regardless of the absolute magnitude of whatever
    produced `proportions`. temperature=1 leaves it unchanged; temperature<=0
    is the hard limit, collapsing to the mode (ties split evenly);
    temperature->inf flattens toward uniform over every value present."""
    if temperature <= 0:
        best = max(proportions.values())
        winners = [v for v, p in proportions.items() if p == best]
        return {v: 1.0 / len(winners) for v in winners}
    scaled = {v: p ** (1.0 / temperature) for v, p in proportions.items()}
    total = sum(scaled.values())
    return {v: s / total for v, s in scaled.items()}


def _visitation_counts(n, parameters, state_info):
    """Expected number of times each state is visited, starting from the
    chain's initial (empty) state, under the converged (one-hot) global
    policy recorded in state_info. This is the same linear system
    policy.py's _evaluate_policy_direct solves for the value function --
    (I - P) x = b, P built from the recorded policy's transitions -- but
    solved transposed against an indicator at the initial state instead of
    against an all-ones vector: visitation counts satisfy x^T (I - P) = e_0^T,
    i.e. (I - P)^T x = e_0."""
    index = {entry["qubits"]: i for i, entry in enumerate(state_info)}
    s0 = qubits_to_key(State.initial(n, parameters).qubits)

    coeffs = {}
    for i, entry in enumerate(state_info):
        key = entry["qubits"]
        if _is_terminal(key):
            continue

        qubits = key_to_qubits(key)
        raw_actions = State(qubits, parameters).get_actions()
        for action, prob in zip(raw_actions, entry["policy"]):
            if prob == 0.0:
                continue
            next_keys, probs, _ = cached_step(parameters, key, action)
            for s_key, P in zip(next_keys, probs):
                if P == 0.0:
                    continue
                j = index[s_key]
                if not _is_terminal(s_key):
                    coeff_key = (i, j)
                    coeffs[coeff_key] = coeffs.get(coeff_key, 0.0) - prob * P

    non_terminal_indices = [i for i in range(len(state_info)) if not _is_terminal(state_info[i]["qubits"])]
    row_of = {idx: row for row, idx in enumerate(non_terminal_indices)}
    size = len(non_terminal_indices)

    rows, cols, data = [], [], []
    for (r_idx, c_idx), coeff in coeffs.items():
        rows.append(row_of[r_idx])
        cols.append(row_of[c_idx])
        data.append(coeff)
    for idx in non_terminal_indices:
        rows.append(row_of[idx])
        cols.append(row_of[idx])
        data.append(1.0)

    A = coo_matrix((data, (rows, cols)), shape=(size, size)).tocsc()
    b = np.zeros(size)
    b[row_of[index[s0]]] = 1.0
    x = spsolve(A.transpose().tocsc(), b)

    counts = {entry["qubits"]: 0.0 for entry in state_info}
    for idx in non_terminal_indices:
        counts[state_info[idx]["qubits"]] = float(x[row_of[idx]])
    return counts


def distill_local_policy_soft(n, state_info, parameters, temperature, prob_floor=1e-3,
                               visitation_weighted=True, visitation_floor=1e-9):
    """Probabilistic distillation (see module docstring). Returns the same
    shape as distill_local_policy -- a list of n dicts, local state ->
    {action: probability} -- but generally with more than one action per
    local state.

    Every matching global state casts a single hard vote for its own
    recorded optimal action -- exactly distill_local_policy's vote, via the
    same argmax(entry["policy"]) -- weighted by visitation_weighted (see
    below) and grouped by each node's own share of that action
    (decompose_action). That gives, per local state, a weighted-majority
    proportion for each candidate local action; _temperature_scale then
    sharpens or flattens *those proportions*: temperature=1 leaves the
    honest weighted-vote shares alone, temperature->0 collapses to the
    single majority action (recovering distill_local_policy's own vote, just
    visitation- rather than count-weighted -- see visitation_weighted),
    temperature->inf flattens toward uniform over whichever actions appear
    at all for that local state.

    prob_floor prunes candidate actions below that probability (renormalizing
    what's left) before they're stored: an unpruned long tail of near-zero
    candidates would needlessly blow up the joint action space
    evaluate_distilled_policy has to enumerate later (the product of every
    node's own candidate count).

    visitation_weighted controls how much each matching global state's hard
    vote counts for -- the *only* thing that differs from
    distill_local_policy's own aggregation once temperature->0 (see above).
    True (default) weights each state by its expected visitation count under
    the optimal policy (see _visitation_counts): a node's distribution then
    reflects how often a local state is actually encountered in practice,
    not how many combinatorially-distinct global microstates happen to share
    it. False weights every state equally (1 each, matching
    distill_local_policy's own one-vote-per-distinct-state convention) --
    useful for isolating the effect of temperature-scaling alone from the
    effect of visitation weighting, or if visitation weighting's bias toward
    the optimal policy's own operating regime is undesirable for your use.

    visitation_floor (only used when visitation_weighted=True) guarantees
    every state policy_iteration discovered still casts *some* vote, even
    one the optimal policy itself never visits (visitation count exactly 0
    -- roughly half of them, typically: states only reachable by an action
    the optimal policy never takes). Without this floor those states -- and
    the local states only they exhibit -- would be silently absent from the
    table; but the *distilled* policy is not the optimal policy and can
    easily wander into exactly those states (e.g. one node's own local
    choice puts a neighboring node into a combination the global policy
    would have steered away from), so evaluate_soft_distilled_policy needs
    an entry to look up regardless. The floor is tiny relative to any
    visitation count that matters, so it doesn't perturb the distribution
    for states that genuinely are visited. When visitation_weighted=False
    every state already contributes a nonzero (equal) vote, so this
    coverage concern doesn't arise and the floor is unused."""
    own_qubits_by_node = [node_qubits(n, node) for node in range(n)]
    visits = _visitation_counts(n, parameters, state_info) if visitation_weighted else None

    tallies = [defaultdict(lambda: defaultdict(float)) for _ in range(n)]

    for entry in state_info:
        key = entry["qubits"]
        if _is_terminal(key):
            continue
        weight = (visits[key] + visitation_floor) if visitation_weighted else 1.0

        qubits = key_to_qubits(key)
        raw_actions = State(qubits, parameters).get_actions()
        chosen_action = raw_actions[int(np.argmax(entry["policy"]))]

        for node, own_qubits in enumerate(own_qubits_by_node):
            local_state = local_state_key(qubits, own_qubits)
            local_action = decompose_action(chosen_action, own_qubits, parameters.discard_enabled)
            tallies[node][local_state][local_action] += weight

    local_policy = []
    for node_tally in tallies:
        table = {}
        for local_state, action_weights in node_tally.items():
            total = sum(action_weights.values())
            proportions = {a: w / total for a, w in action_weights.items()}
            scaled = _temperature_scale(proportions, temperature)
            pruned = {a: p for a, p in scaled.items() if p >= prob_floor}
            renorm = sum(pruned.values())
            table[local_state] = {a: p / renorm for a, p in pruned.items()}
        local_policy.append(table)
    return local_policy


def distill(n, p, p_s, cutoff, tolerance=1e-5, allow_discard=False, progress=True, savedata=True,
            method="iterative"):
    """Ensure the global-knowledge policy exists (running policy iteration if
    not), then greedily distill it into a local per-node policy. method --
    see policy_iteration: "direct" (exact sparse-linear-system policy
    evaluation) is substantially faster than the default "iterative" once
    allow_discard inflates the state/action space, without changing the
    result -- see policy.py's _evaluate_policy_direct docstring."""
    _check_local_policy_discard_support(allow_discard)
    if not check_policyiter_data(n, p, p_s, cutoff, tolerance, allow_discard):
        policy_iteration(n, p, p_s, cutoff, tolerance=tolerance, progress=progress,
                          savedata=True, allow_discard=allow_discard, method=method)

    _, state_info, _ = load_policyiter_data(n, p, p_s, cutoff, tolerance, allow_discard)
    parameters = Configuration(p=p, p_s=p_s, cut=cutoff, allow_discard=allow_discard)
    local_policy = distill_local_policy(n, state_info, parameters)

    if savedata:
        save_local_policy_data(n, p, p_s, cutoff, tolerance, local_policy, allow_discard=allow_discard)

    return local_policy


def distill_soft(n, p, p_s, cutoff, temperature, tolerance=1e-5, allow_discard=False,
                  prob_floor=1e-3, visitation_weighted=True, progress=True, savedata=True,
                  method="iterative"):
    """Ensure the global-knowledge policy exists (running policy iteration if
    not), then probabilistically distill it into a local per-node policy at
    the given temperature (see distill_local_policy_soft). visitation_weighted
    toggles whether each matching global state's vote is weighted by its
    expected visitation count under the optimal policy (default) or counted
    equally (see distill_local_policy_soft's docstring). method -- see
    distill's docstring."""
    _check_local_policy_discard_support(allow_discard)
    if not check_policyiter_data(n, p, p_s, cutoff, tolerance, allow_discard):
        policy_iteration(n, p, p_s, cutoff, tolerance=tolerance, progress=progress,
                          savedata=True, allow_discard=allow_discard, method=method)

    _, state_info, _ = load_policyiter_data(n, p, p_s, cutoff, tolerance, allow_discard)
    parameters = Configuration(p=p, p_s=p_s, cut=cutoff, allow_discard=allow_discard)
    local_policy = distill_local_policy_soft(n, state_info, parameters, temperature, prob_floor=prob_floor,
                                              visitation_weighted=visitation_weighted)

    if savedata:
        save_local_policy_data(n, p, p_s, cutoff, tolerance, local_policy, allow_discard=allow_discard,
                                temperature=temperature, visitation_weighted=visitation_weighted)

    return local_policy


# --------------------------------------------------------------------------- #
# --------------------------------  I/O  ------------------------------------ #
# --------------------------------------------------------------------------- #

def _localpolicy_filename(n, p, p_s, cutoff, tolerance, allow_discard=False, temperature=None,
                           visitation_weighted=True):
    if temperature is None:
        soft_suffix = ''
    else:
        weighting_suffix = '' if visitation_weighted else '_unweighted'
        soft_suffix = '_soft_tau%.3f%s' % (temperature, weighting_suffix)
    return 'data_localpolicy/n%s_p%.3f_ps%.3f_tc%s_tol%s%s%s_local' % (
        n, p, p_s, cutoff, tolerance, _discard_suffix(allow_discard), soft_suffix)


def check_local_policy_data(n, p, p_s, cutoff, tolerance, allow_discard=False, temperature=None,
                             visitation_weighted=True):
    """True if a local policy has already been distilled and saved for this
    set of parameters."""
    return Path(_localpolicy_filename(n, p, p_s, cutoff, tolerance, allow_discard, temperature,
                                       visitation_weighted)).exists()


def save_local_policy_data(n, p, p_s, cutoff, tolerance, local_policy, allow_discard=False, temperature=None,
                            visitation_weighted=True):
    os.makedirs('data_localpolicy', exist_ok=True)
    filename = _localpolicy_filename(n, p, p_s, cutoff, tolerance, allow_discard, temperature,
                                      visitation_weighted)
    with open(filename, 'wb') as handle:
        pickle.dump({'n': n, 'local_policy': local_policy}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_local_policy_data(n, p, p_s, cutoff, tolerance, allow_discard=False, temperature=None,
                            visitation_weighted=True):
    filename = _localpolicy_filename(n, p, p_s, cutoff, tolerance, allow_discard, temperature,
                                      visitation_weighted)
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data['local_policy']


def _load_or_distill(n, p, p_s, cutoff, tolerance, allow_discard, progress=True, method="iterative"):
    """The greedily-distilled local policy for this set of parameters, from
    disk if it's already been computed, or freshly distilled (and cached)
    otherwise. method -- see distill's docstring; irrelevant if the policy
    iteration data already exists on disk."""
    _check_local_policy_discard_support(allow_discard)
    if check_local_policy_data(n, p, p_s, cutoff, tolerance, allow_discard):
        return load_local_policy_data(n, p, p_s, cutoff, tolerance, allow_discard)
    return distill(n, p, p_s, cutoff, tolerance=tolerance, allow_discard=allow_discard, progress=progress,
                    method=method)


def _load_or_distill_soft(n, p, p_s, cutoff, tolerance, allow_discard, temperature,
                           prob_floor=1e-3, visitation_weighted=True, progress=True, method="iterative"):
    """The probabilistically-distilled local policy at this temperature, from
    disk if it's already been computed, or freshly distilled (and cached)
    otherwise. method -- see distill's docstring; irrelevant if the policy
    iteration data already exists on disk."""
    _check_local_policy_discard_support(allow_discard)
    if check_local_policy_data(n, p, p_s, cutoff, tolerance, allow_discard, temperature=temperature,
                                visitation_weighted=visitation_weighted):
        return load_local_policy_data(n, p, p_s, cutoff, tolerance, allow_discard, temperature=temperature,
                                       visitation_weighted=visitation_weighted)
    return distill_soft(n, p, p_s, cutoff, temperature, tolerance=tolerance, allow_discard=allow_discard,
                         prob_floor=prob_floor, visitation_weighted=visitation_weighted, progress=progress,
                         method=method)


def build_local_policy_json(n, p, p_s, cutoff, tolerance, allow_discard, out_path,
                             temperature=None, prob_floor=1e-3, visitation_weighted=True,
                             method="iterative"):
    """Write the distilled local policy (distilling it first if necessary) as
    JSON for simulate.jl to consume: one table per node, each entry giving a
    local state and its candidate local actions with their probabilities --
    a single candidate at probability 1 for the greedy method
    (temperature=None, the default), generally several for the probabilistic
    method (temperature given). visitation_weighted only applies to the
    probabilistic method -- see distill_local_policy_soft. method -- see
    distill's docstring; irrelevant if the policy iteration data already
    exists on disk."""
    if temperature is None:
        local_policy = _load_or_distill(n, p, p_s, cutoff, tolerance, allow_discard, method=method)
    else:
        local_policy = _load_or_distill_soft(n, p, p_s, cutoff, tolerance, allow_discard, temperature,
                                              prob_floor=prob_floor, visitation_weighted=visitation_weighted,
                                              method=method)

    nodes = []
    for node_table in local_policy:
        entries = []
        for local_state, action_probs in node_table.items():
            entries.append({
                "qubits": [[age, hanging] for age, hanging in local_state],
                "actions": [
                    {"swap": swap, "discard": list(discard), "prob": prob}
                    for (swap, discard), prob in action_probs.items()
                ],
            })
        nodes.append(entries)

    with open(out_path, "w") as handle:
        json.dump({"n": n, "nodes": nodes}, handle)


# --------------------------------------------------------------------------- #
# ----------------------  EXACT DISTILLED-POLICY EVAL  ----------------------- #
# --------------------------------------------------------------------------- #

def _joint_action_distribution(qubits, own_qubits_by_node, local_policy, discard_enabled):
    """Every (global action, probability) pair reachable by each node
    independently sampling its own local action from its own local
    distribution -- the joint of n independent per-node choices (nodes never
    coordinate, by construction of "local" execution). Since qubit ownership
    never overlaps between nodes, every combination is internally consistent;
    for allow_discard=False every resulting action is also always a valid
    member of that state's State.get_actions() (any subset of the currently-
    eligible swap positions is valid, and a node's own eligibility -- both
    its qubits occupied and non-hanging -- is exactly what its local table is
    keyed on). For a greedy (single-candidate-per-state) local policy this
    always collapses to exactly one combination at probability 1, so this is
    a strict generalization of distilled-policy evaluation's old
    single-action-per-state lookup."""
    per_node_candidates = [
        list(local_policy[node][local_state_key(qubits, own_qubits)].items())
        for node, own_qubits in enumerate(own_qubits_by_node)
    ]

    combined = {}
    for combo in product(*per_node_candidates):
        swap_nodes = []
        discard_qubits = []
        joint_prob = 1.0
        for own_qubits, ((swap, discard), prob) in zip(own_qubits_by_node, combo):
            if swap:
                swap_nodes.append(own_qubits[0])
            discard_qubits.extend(discard)
            joint_prob *= prob
        if joint_prob == 0.0:
            continue
        action_key = (tuple(sorted(swap_nodes)), tuple(sorted(discard_qubits)))
        combined[action_key] = combined.get(action_key, 0.0) + joint_prob

    return [
        ((swap_nodes, discard_qubits) if discard_enabled else list(swap_nodes), prob)
        for (swap_nodes, discard_qubits), prob in combined.items()
    ]


def _evaluate_local_policy(n, p, p_s, cutoff, allow_discard, local_policy, progress=True):
    """Exact expected delivery time of a local policy (in the shared
    {local_state: {action: probability}} representation -- see module
    docstring), solved directly over the same discrete MDP policy_iteration /
    policy_eval_swapasap evaluate against (see policy.py's
    _evaluate_policy_direct) -- unlike simulate.py's Monte Carlo physical
    simulation via simulate.jl, this is noise-free and directly comparable to
    policy_iteration's optimal value and policy_eval_swapasap's swap-asap
    value.

    Walks the state graph the local policy itself induces -- at each state,
    the joint distribution over global actions from every node's independent
    local choice (_joint_action_distribution), generalizing
    _evaluate_policy_direct's sum over a single stored action_space/policy to
    a sum over that joint -- registering states as they're discovered and
    accumulating (I - P) V = -1 for a direct sparse solve."""
    parameters = Configuration(p=p, p_s=p_s, cut=cutoff, allow_discard=allow_discard)
    own_qubits_by_node = [node_qubits(n, node) for node in range(n)]

    s0 = qubits_to_key(State.initial(n, parameters).qubits)
    state_list = [s0]
    index = {s0: 0}
    coeffs = {}  # (row_idx, col_idx) -> accumulated coefficient, as in _evaluate_policy_direct

    idx = 0
    while idx < len(state_list):
        key = state_list[idx]
        if _is_terminal(key):
            idx += 1
            continue

        qubits = key_to_qubits(key)
        for action, action_prob in _joint_action_distribution(qubits, own_qubits_by_node, local_policy,
                                                                parameters.discard_enabled):
            next_keys, probs, _ = cached_step(parameters, key, action)
            for s_key, P in zip(next_keys, probs):
                if P == 0.0:
                    continue
                s_idx = index.get(s_key)
                if s_idx is None:
                    s_idx = len(state_list)
                    index[s_key] = s_idx
                    state_list.append(s_key)
                if not _is_terminal(s_key):
                    coeff_key = (idx, s_idx)
                    coeffs[coeff_key] = coeffs.get(coeff_key, 0.0) - action_prob * P

        idx += 1
        if progress:
            print("Local-policy eval: state %d/%d" % (idx, len(state_list)) + " " * 20, end="\r")

    non_terminal_indices = [i for i in range(len(state_list)) if not _is_terminal(state_list[i])]
    row_of = {s_idx: row for row, s_idx in enumerate(non_terminal_indices)}
    size = len(non_terminal_indices)

    rows, cols, data = [], [], []
    for (r_idx, c_idx), coeff in coeffs.items():
        rows.append(row_of[r_idx])
        cols.append(row_of[c_idx])
        data.append(coeff)
    for s_idx in non_terminal_indices:
        rows.append(row_of[s_idx])
        cols.append(row_of[s_idx])
        data.append(1.0)

    A = coo_matrix((data, (rows, cols)), shape=(size, size)).tocsc()
    b = -np.ones(size)
    V = spsolve(A, b)

    if progress:
        print(" " * 80, end="\r")

    return -(float(V[row_of[0]]) + 1)


def evaluate_distilled_policy(n, p, p_s, cutoff, tolerance=1e-5, allow_discard=False, progress=True,
                               method="iterative"):
    """Exact expected delivery time of the greedily-distilled local policy
    (see distill_local_policy), computing/caching it first if necessary.
    method -- see distill's docstring; irrelevant if the policy iteration
    data already exists on disk."""
    local_policy = _load_or_distill(n, p, p_s, cutoff, tolerance, allow_discard, progress=progress,
                                     method=method)
    return _evaluate_local_policy(n, p, p_s, cutoff, allow_discard, local_policy, progress=progress)


def evaluate_soft_distilled_policy(n, p, p_s, cutoff, temperature, tolerance=1e-5, allow_discard=False,
                                    prob_floor=1e-3, visitation_weighted=True, progress=True,
                                    method="iterative"):
    """Exact expected delivery time of the probabilistically-distilled local
    policy at the given temperature (see distill_local_policy_soft),
    computing/caching it first if necessary. visitation_weighted -- see
    distill_local_policy_soft. method -- see distill's docstring; irrelevant
    if the policy iteration data already exists on disk."""
    local_policy = _load_or_distill_soft(n, p, p_s, cutoff, tolerance, allow_discard, temperature,
                                          prob_floor=prob_floor, visitation_weighted=visitation_weighted,
                                          progress=progress, method=method)
    return _evaluate_local_policy(n, p, p_s, cutoff, allow_discard, local_policy, progress=progress)


# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Distill a local per-node policy from the global-knowledge '
                     'policy computed by policy.py -- greedily by default, or '
                     'probabilistically (temperature-scaled weighted-majority vote) '
                     'if --temperature is given.')
    parser.add_argument('--n', type=int, default=3, help='Number of nodes.')
    parser.add_argument('--p', type=float, default=0.8,
                        help='Elementary link generation success probability.')
    parser.add_argument('--p_s', type=float, default=0.8,
                        help='Entanglement swap success probability.')
    parser.add_argument('--cutoff', type=int, default=5, help='Qubit-age cutoff.')
    parser.add_argument('--tol', type=float, default=1e-5, help='Value tolerance.')
    parser.add_argument('--discard', nargs='?', const=True, default=False, type=int,
                        help='Allow the agent to discard occupied qubits each time slot. '
                             'Must match a previously computed policy\'s setting.')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Enable probabilistic distillation, temperature-scaling the '
                             'weighted-majority vote proportions: 1 leaves them alone, '
                             '<=0 collapses to the hard majority, >1 flattens toward '
                             'uniform. Omit for the deterministic greedy majority-vote method.')
    parser.add_argument('--prob-floor', type=float, default=1e-3,
                        help='Probabilistic distillation only: prune candidate local '
                             'actions below this probability before storing them.')
    parser.add_argument('--uniform-states', action='store_true',
                        help='Probabilistic distillation only: weight every matching global '
                             'state\'s vote equally (1 each) instead of by its expected '
                             'visitation count under the optimal policy. See '
                             'distill_local_policy_soft\'s docstring -- note this is '
                             'independent of --temperature, including at --temperature 0.')
    parser.add_argument('--method', choices=['iterative', 'direct'], default='iterative',
                        help='How to run the underlying policy-iteration evaluation steps, if '
                             'the global-knowledge policy still needs to be computed: '
                             '"iterative" (default) matches policy.py\'s own default; "direct" '
                             'solves each policy evaluation exactly via sparse LU instead of '
                             'Bellman-backup sweeps, which is substantially faster once '
                             '--discard inflates the state/action space. See '
                             'policy.py\'s policy_iteration docstring.')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output.')
    args = parser.parse_args()

    if args.temperature is None:
        if check_local_policy_data(args.n, args.p, args.p_s, args.cutoff, args.tol, args.discard):
            print('Local policy already distilled!')
        else:
            local_policy = distill(args.n, args.p, args.p_s, args.cutoff, tolerance=args.tol,
                                    allow_discard=args.discard, progress=not args.quiet, savedata=True,
                                    method=args.method)
            counts = ', '.join('node %d: %d local states' % (j, len(t)) for j, t in enumerate(local_policy))
            print('Done! %s' % counts)
    else:
        visitation_weighted = not args.uniform_states
        if check_local_policy_data(args.n, args.p, args.p_s, args.cutoff, args.tol, args.discard,
                                    temperature=args.temperature, visitation_weighted=visitation_weighted):
            print('Local policy already distilled!')
        else:
            local_policy = distill_soft(args.n, args.p, args.p_s, args.cutoff, args.temperature,
                                         tolerance=args.tol, allow_discard=args.discard,
                                         prob_floor=args.prob_floor, visitation_weighted=visitation_weighted,
                                         progress=not args.quiet, savedata=True, method=args.method)
            counts = ', '.join(
                'node %d: %d local states, avg %.2f candidates/state' % (
                    j, len(t), np.mean([len(a) for a in t.values()]) if t else 0.0)
                for j, t in enumerate(local_policy)
            )
            print('Done! %s' % counts)
