"""Belief-state primitives for the decentralized POMDP formulation of the
qubit-age cutoff model defined in environment.py / policy.py.

A node in the chain only ever observes the (age, hanging) of the one or two
qubits it physically holds -- never the rest of the network. All it can do
about the states it can't see is keep a running BELIEF: a probability
distribution over which full global state (in policy.py's qubits_to_key
sense) is actually true, updated by Bayesian filtering as its own qubits
evolve. po_policy.py runs policy iteration directly over (true state,
per-node belief) tuples using the primitives defined here -- see that
module's docstring for the planning algorithm and the self-consistency
argument that makes it tractable.

This module only reads from environment.py / policy.py (imports their
state/transition machinery) and never modifies them.
"""
from collections import defaultdict
from itertools import combinations

from environment import Configuration, State
from policy import cached_step, qubits_to_key, _is_terminal


# --------------------------------------------------------------------------- #
# ------------------------  NODE OWNERSHIP / OBSERVATION  -------------------- #
# --------------------------------------------------------------------------- #

def node_qubits(n, node):
    """Flat qubit indices (0-indexed, environment.py's convention) owned by
    `node` (0-indexed): a single qubit for the two end nodes, two qubits
    (left, right) for every middle node. Mirrors distill.py's node_qubits --
    duplicated here (rather than imported) to keep this belief-state
    pipeline decoupled from the unrelated greedy/soft distillation pipeline
    in distill.py."""
    if node == 0:
        return (0,)
    if node == n - 1:
        return (2 * n - 3,)
    return (2 * node - 1, 2 * node)


def observation_of(state_key, own_qubits):
    """A node's own observation of a global state -- (age, hanging) for each
    qubit it owns, in ascending flat-index order -- read directly off the
    state's hashable qubits_to_key() form (a tuple of (age, hanging) pairs
    indexed by flat qubit index), without materializing Qubit objects."""
    return tuple(state_key[q] for q in own_qubits)


def local_actions(observation, own_qubits, discard_enabled):
    """Every local action a node can independently decide on given only its
    own observation -- no global state needed, exactly what a decentralized
    node can compute for itself. A (swap, discard) pair, mirroring
    distill.py's decompose_action shape: swap is only offered to a middle
    node (two owned qubits) when both are occupied and non-hanging;
    discard, when enabled, is any subset of the node's own currently
    occupied qubits, except when swapping -- swapping consumes both of a
    middle node's qubits, so discard must be empty alongside it (matching
    environment.py's own rule that swapping and discarding the same qubit
    in the same time slot is undefined)."""
    occupied = tuple(q for q, (age, _hanging) in zip(own_qubits, observation) if age != -1)
    can_swap = (
        len(own_qubits) == 2
        and observation[0][0] != -1 and not observation[0][1]
        and observation[1][0] != -1 and not observation[1][1]
    )

    if not discard_enabled:
        return [(False, ())] + ([(True, ())] if can_swap else [])

    non_swap_discards = [
        combo for k in range(len(occupied) + 1) for combo in combinations(occupied, k)
    ]
    actions = [(False, d) for d in non_swap_discards]
    if can_swap:
        actions.append((True, ()))
    return actions


def combine_actions(own_qubits_by_node, local_actions_chosen, discard_enabled):
    """Assemble one global action (in whichever shape State.get_actions()
    uses for this Configuration) from each node's own, independently chosen
    local action -- mirrors _joint_action_distribution's combination step in
    distill.py. Qubit ownership never overlaps between nodes, so every
    combination is internally consistent by construction."""
    swap_nodes = sorted(
        own_qubits[0]
        for own_qubits, (swap, _discard) in zip(own_qubits_by_node, local_actions_chosen)
        if swap
    )
    if not discard_enabled:
        return tuple(swap_nodes)
    discard_qubits = sorted(q for _swap, discard in local_actions_chosen for q in discard)
    return (tuple(swap_nodes), tuple(discard_qubits))


# --------------------------------------------------------------------------- #
# -------------------------------  BELIEFS  ----------------------------------#
# --------------------------------------------------------------------------- #
# A belief is, at the finest grain, a dict {global_state_key: probability}
# (probabilities summing to 1) over every global state consistent with a
# node's own current observation. Since the exact set of reachable beliefs
# is not generally finite (a node left untouched keeps refining -- never
# exactly repeating -- its estimate of the rest of the chain for as long as
# nothing resets its own qubits), po_policy.py's planner rounds beliefs to
# `precision` decimal places before using them as dict/table keys, so the
# reachable *rounded* belief space stays finite. This is the one
# approximation this module makes beyond the self-consistency assumption
# documented in po_policy.py: it bounds an otherwise-unbounded belief space,
# it doesn't prove finiteness.

def belief_key(belief, precision, prob_floor=0.0, keep=None):
    """Canonical, hashable, rounded form of a belief dict: round every
    probability to `precision` decimals, drop entries at or below
    `prob_floor` (renormalizing what's left -- exactly like distill.py's
    soft-distillation prob_floor), and return a tuple of (state_key,
    probability) pairs sorted by state_key. Every state in a belief's
    support shares the same value for the owning node's own qubits (that's
    what conditioning on self-observation means), so this key also doubles
    as that node's current observation -- see belief_observation.

    `keep`, if given, is a state_key that po_policy.py knows is the actual
    ground-truth continuation for this branch: it is exempted from the
    prob_floor prune (rounding to 0 would round it up to the smallest
    representable value at this precision instead), so it can never be
    pruned out of the belief entirely. Without this, a true state that
    happens to be a minority hypothesis within its own observation bucket
    could be discarded by rounding/pruning, breaking the invariant
    po_policy.hyperstate_step relies on -- that the actual next global state
    always remains in the belief support that's carried forward -- which
    would otherwise surface later as a missing-observation KeyError."""
    quantum = 10 ** (-precision)
    rounded = {}
    for s, p in belief.items():
        r = round(p, precision)
        if s == keep and r <= 0.0:
            r = quantum
        rounded[s] = r
    pruned = {s: p for s, p in rounded.items() if p > 0.0 and (p > prob_floor or s == keep)}
    total = sum(pruned.values())
    if total == 0.0:
        # Every entry rounded/pruned to zero -- shouldn't happen for a
        # normalized belief under a sane precision/floor, but fall back to
        # the single most likely state rather than leave the belief
        # undefined.
        s = max(belief, key=belief.get)
        return ((s, 1.0),)
    return tuple(sorted((s, p / total) for s, p in pruned.items()))


def belief_from_key(key):
    """Belief dict from its canonical belief_key() form."""
    return dict(key)


def belief_observation(key, own_qubits):
    """The (age, hanging)-per-qubit observation every state in this belief
    shares -- see belief_key's docstring."""
    any_state = key[0][0]
    return observation_of(any_state, own_qubits)


def initial_belief_key(n, parameters):
    """The belief every node starts with: certainty in the chain's known
    initial (empty) configuration."""
    s0 = qubits_to_key(State.initial(n, parameters).qubits)
    return ((s0, 1.0),)


def predict_and_filter(belief, own_qubits, action, parameters):
    """Advance a belief through one hyper-state transition given the joint
    `action` actually taken. po_policy.py's offline planner knows this
    action exactly, since it tracks every node's belief centrally -- see
    that module's docstring for why using it here doesn't require modeling
    other nodes' beliefs recursively.

    For every global state in the belief's support, branches over every
    possible outcome of `action` (the environment's own stochastics, via
    policy.py's memoized cached_step), weights by prior belief * transition
    probability, and groups the result by this node's own resulting
    observation -- the only thing it actually gets to see.

    Returns a dict: this node's possible next observation -> (probability of
    seeing it, un-rounded posterior belief dict over next global states
    consistent with it). po_policy.py looks up the branch matching whichever
    observation actually occurs along each true-state transition it's
    enumerating; it never needs to sum over "which observation," since it
    already knows the true next state exactly."""
    grouped = defaultdict(lambda: defaultdict(float))
    for state_key, prior in belief.items():
        if prior == 0.0:
            continue
        if _is_terminal(state_key):
            # Delivered states are absorbing -- nothing left to predict.
            grouped[observation_of(state_key, own_qubits)][state_key] += prior
            continue
        next_keys, probs, _ = cached_step(parameters, state_key, action)
        for s_key, p in zip(next_keys, probs):
            if p == 0.0:
                continue
            obs = observation_of(s_key, own_qubits)
            grouped[obs][s_key] += prior * p

    result = {}
    for obs, dist in grouped.items():
        total = sum(dist.values())
        result[obs] = (total, {s: p / total for s, p in dist.items()})
    return result
