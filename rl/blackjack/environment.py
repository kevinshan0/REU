from dataclasses import dataclass
import random

# blackjack but with inifinite deck, no splitting, and ace is worth 1
# if agent_sum in [12, 20] they have they choice between hit or stand
# anything less, they must hit, anything more (21) they must stand
# implementing on policy first-visit mc control method

@dataclass
class Blackjack:
    def __init__(self):
        self.deck = [10 if x > 10 else x for _ in range(0, 4) for x in range(1, 14)]

    def draw_cards(self, n: int):
        return [self.deck[random.randint(0, len(self.deck) - 1)] for _ in range(0, n)]

        
@dataclass
class State:
    def __init__(self):
        self.draw = Blackjack().draw_cards
        self.agent_sum = sum(self.draw(2))
        self.dealer_sum = self.draw(1)[0]
        self.stood = False

        while self.agent_sum < 12:
            self.play(True)

    def is_valid_non_terminal(self):
        if self.stood:
            return False
        if self.agent_sum >= 12 and self.agent_sum <= 21:
            return True
        else:
            return False
        
    def play(self, hit: bool):
    # True for hit, False for stand
        if hit:
            self.agent_sum += self.draw(1)[0]
        else:
            self.stood = True
            while self.dealer_sum < 17:
                self.dealer_sum += self.draw(1)[0]

    def evaluate(self):
        if self.agent_sum > 21:
            return -1
        if self.dealer_sum > 21:
            return 1
        if self.agent_sum > self.dealer_sum:
            return 1
        if self.agent_sum == self.dealer_sum:
            return 0
        
        return -1

@dataclass
class Episode:
    def __init__(self):
        self.state = State()

    def run_episode(self, policy: dict[tuple[tuple[int, int], bool], int], action_values: dict[tuple[tuple[int, int], bool], float], returns: dict[tuple[tuple[int, int], bool], list[int]], epsilon):
        visited = []
        while self.state.is_valid_non_terminal():
            p_hit = policy[((self.state.agent_sum, self.state.dealer_sum), True)]
            action = random.random() < p_hit
            visited.append(((self.state.agent_sum, self.state.dealer_sum), action))
            self.state.play(action)

        reward = self.state.evaluate()
        for state, action in visited:
            returns[(state, action)].append(reward)
            action_values[(state, action)] = sum(returns[(state, action)]) / len(returns[(state, action)])

        for state, _ in visited:
            max_action = True if action_values[(state, True)] > action_values[(state, False)] else False
            policy[(state, max_action)] = 1 - epsilon + epsilon / 2
            policy[(state, not max_action)] = epsilon / 2

        return reward
