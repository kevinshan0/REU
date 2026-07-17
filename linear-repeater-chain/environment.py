from typing import List, Set, Union
from copy import deepcopy
from itertools import combinations
from dataclasses import dataclass

@dataclass
class Configuration:
    p: float = 0.8
    p_s: float = 0.8
    cut: int = 5
    allow_discard: Union[bool, int] = False # False disables discard; True enables it
                                             # with no per-timestep cap; an int enables
                                             # it capped at that many qubits per timestep.

    @property
    def discard_enabled(self) -> bool:
        return self.allow_discard is not False

    @property
    def discard_cap(self):
        """Max qubits that may be discarded in a single time slot, or None
        for no cap. Only meaningful when discard_enabled is True."""
        if self.allow_discard is True:
            return None
        return int(self.allow_discard)

@dataclass
class Qubit:
    """
    age is set to 0 during the timeslot it is initialized.
    if age is not -1, the qubit is entangled.
    hanging is True when this qubit's former link partner has been
    discarded out from under it: the qubit is still occupied (has an age)
    but is no longer part of any link, so it's skipped when links are
    parsed (see State.links_of).
    """
    age: int
    hanging: bool = False

    def clear(self):
        """Reset to empty: unentangled and not hanging."""
        self.age = -1
        self.hanging = False

@dataclass
class State:
    def __init__(self, qubits: List[Qubit], parameters: Configuration):
        self.parameters = parameters
        self.qubits = qubits # qubits labeled 0 to n - 1

    @staticmethod
    def initial(n: int, parameters: Configuration) -> "State":
        """Fresh n-node chain with no entangled links: the two end nodes
        hold a single qubit each, and the n-2 middle nodes hold a left
        and a right qubit."""
        assert n >= 2, "a chain needs at least two nodes"
        return State([Qubit(-1) for _ in range(2 * n - 2)], parameters)

    @property
    def num_nodes(self) -> int:
        return (len(self.qubits) + 2) // 2

    def check_e2e_link(self) -> bool:
        """True if the two end nodes share a virtual entangled link."""
        if self.qubits[0].age == -1 or self.qubits[-1].age == -1:
            return False
        return self.link_partner(self.qubits, 0) == len(self.qubits) - 1

    def age_qubits(self):
        for qubit in self.qubits:
            if qubit.age != -1:
                qubit.age += 1

    def get_entanglement_generation_transitions(self):
        """
        returns all possible states resultant of entanglement generation attempts.
        also returns the probability to reach each state
        """
        transition_space: List[List[Qubit]] = []
        probabilities: List[float] = []

        possible: List[int] = [] # list of indices i of qubits where i and i+1 can attempt entanglement generation
        for i in range(0, len(self.qubits) - 1, 2):
            if self.qubits[i].age == -1 and self.qubits[i+1].age == -1:
                possible.append(i)

        # do for all number of possible successful entanglement generation attempts
        for success_num in range(0, len(possible) + 1):
            # do for all combinations of possible generations with a total of success_num successful generations
            for combination in combinations(possible, success_num):
                copy = deepcopy(self.qubits)
                for success_index in combination:
                    copy[success_index].age = 0
                    copy[success_index+1].age = 0
                transition_space.append(copy)
                probabilities.append((self.parameters.p ** success_num) * ((1 - self.parameters.p) ** (len(possible) - success_num)))

        return transition_space, probabilities

    @staticmethod
    def links_of(qubits):
        """Reconstruct links (sets of two qubit indices) from any qubit list,
        using the non-crossing even-is-left / odd-is-right structure.
        Hanging qubits (occupied, but orphaned by a discarded partner) are
        skipped entirely -- neither pushed nor popped -- so the remaining
        qubits still pair up correctly around the gap."""
        links: List[Set[int]] = []
        stack: List[int] = [] # stores the indices of entangled qubits
        for index, qubit in enumerate(qubits):
            if qubit.age == -1 or qubit.hanging:
                continue
            if index % 2 == 0: # right qubit
                stack.append(index)
            else:
                links.append({stack.pop(), index})

        return links

    def get_links(self):
        """
        returns a list of sets contain the indices of entangled qubits.
        tuples contain two integers, ordered least to greatest.
        """
        return self.links_of(self.qubits)

    @staticmethod
    def link_partner(qubits, q):
        """Index of the qubit entangled with q on this list, or None if q is free."""
        if qubits[q].age == -1:
            return None
        for link in State.links_of(qubits):
            if q in link:
                a, b = tuple(link)
                return b if a == q else a
        return None

    def get_actions(self):
        """
        Returns the valid actions from this state. By default (when
        self.parameters.discard_enabled is False) an action is a flat list
        of swap mid-qubit indices, exactly as before. When discard is
        enabled, an action is instead a (swap_nodes, discard_qubits) pair --
        swap_nodes as before, discard_qubits any subset (up to
        self.parameters.discard_cap qubits, or unbounded if discard_cap is
        None) of currently occupied qubit indices -- restricted to
        combinations where the two sets don't share a qubit (swapping and
        discarding the same qubit in the same time slot is undefined).
        """
        possible_swaps: List[int] = [ # indices of qubits where entanglement swap can be attempted between i and i+1
            i for i in range(1, len(self.qubits) - 1, 2)
            if self.qubits[i].age != -1 and not self.qubits[i].hanging
            and self.qubits[i + 1].age != -1 and not self.qubits[i + 1].hanging
        ]
        swap_actions: List[List[int]] = [
            list(combination)
            for k in range(len(possible_swaps) + 1)
            for combination in combinations(possible_swaps, k)
        ]

        if not self.parameters.discard_enabled:
            return swap_actions

        occupied: List[int] = [i for i, qubit in enumerate(self.qubits) if qubit.age != -1]
        cap = self.parameters.discard_cap
        max_discard = len(occupied) if cap is None else min(cap, len(occupied))
        discard_actions: List[tuple] = [
            combination
            for k in range(max_discard + 1)
            for combination in combinations(occupied, k)
        ]

        actions = []
        for swap_nodes in swap_actions:
            used_qubits = {q for node in swap_nodes for q in (node, node + 1)}
            valid_discards = [d for d in discard_actions if used_qubits.isdisjoint(d)]
            # Reversed so the empty discard set (always valid) comes last within
            # each swap group: swap_actions is itself built smallest-to-largest,
            # so the very last action overall ends up (every possible swap, no
            # discard) -- the "swap-asap" convention policy_eval_swapasap relies
            # on when picking action_space[-1].
            for discard_qubits in reversed(valid_discards):
                actions.append((tuple(swap_nodes), discard_qubits))

        return actions
    
    def get_entanglement_swap_transitions_for_action(self, action: List[int]):
        transition_space: List[List[Qubit]] = []
        probabilities: List[float] = []
        p_s = self.parameters.p_s

        for success_num in range(0, len(action) + 1):
            for successes in combinations(action, success_num):
                successes = set(successes)
                copy = deepcopy(self.qubits)

                for node in action: 
                    # ascending order matters
                    qL, qR = node, node + 1
                    left_ok  = copy[qL].age != -1 and not copy[qL].hanging
                    right_ok = copy[qR].age != -1 and not copy[qR].hanging

                    if left_ok and right_ok:
                        if node in successes:
                            # success: consume only the middle node. The two outer
                            # qubits keep their own ages (qubit-age convention) and
                            # are rejoined into one link by links_of.
                            copy[qL].clear()
                            copy[qR].clear()
                        else:
                            # failure: discard BOTH current input links entirely
                            left  = self.link_partner(copy, qL)
                            right = self.link_partner(copy, qR)
                            copy[qL].clear()
                            copy[qR].clear()
                            if left  is not None: copy[left].clear()
                            if right is not None: copy[right].clear()
                    else:
                        # guard: a neighbour's failure already consumed the shared
                        # link, so this node has at most one link left. The paper
                        # destroys that remaining link whether this swap is marked a
                        # success or a failure, so we do the same.
                        for q in (qL, qR):
                            if copy[q].age != -1:
                                partner = self.link_partner(copy, q)
                                copy[q].clear()
                                if partner is not None: copy[partner].clear()

                transition_space.append(copy)
                probabilities.append((p_s ** success_num) * ((1 - p_s) ** (len(action) - success_num)))

        return transition_space, probabilities

    def get_transitions_for_action(self, action):
        """
        Dispatches to get_entanglement_swap_transitions_for_action, matching
        whichever action shape get_actions() produces for this State's
        Configuration. When discard is disabled, action is the plain
        swap-node list and this is a pure passthrough. When discard is
        enabled, action is the (swap_nodes, discard_qubits) pair get_actions()
        produces: swaps are resolved probabilistically first, exactly as
        without discard, then discards are applied deterministically to
        every resulting branch -- discard has no failure mode, it's simply
        the agent choosing to free a qubit. Discarding a qubit never
        discards its partner; the partner (if any) becomes hanging instead.
        """
        if not self.parameters.discard_enabled:
            return self.get_entanglement_swap_transitions_for_action(action)

        swap_nodes, discard_qubits = action
        transition_space, probabilities = self.get_entanglement_swap_transitions_for_action(list(swap_nodes))

        for qubits in transition_space:
            for q in discard_qubits:
                partner = self.link_partner(qubits, q)
                qubits[q].clear()
                if partner is not None:
                    qubits[partner].hanging = True

        return transition_space, probabilities

    def cutoff(self):
        """
        Checks if any qubit's age equals the cutoff. A qubit that's part of
        a link is discarded along with its partner. A hanging qubit has no
        partner (its own former partner was already discarded out from
        under it, which is why it's hanging), so it's discarded on its own
        -- otherwise, being excluded from get_links(), it would never be
        seen here and would age forever.
        """
        links = self.get_links()
        involved_qubits: List[int] = []
        for index, qubit in enumerate(self.qubits):
            if qubit.age == self.parameters.cut:
                if qubit.hanging:
                    involved_qubits.append(index)
                else:
                    for link in links:
                        if index in link:
                            involved_qubits += list(link)

        for qubit_index in involved_qubits:
            self.qubits[qubit_index].clear()
