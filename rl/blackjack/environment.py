from dataclasses import dataclass
import random

# blackjack but with inifinite deck,  no splitting, and ace is worth 1

@dataclass
class InfiniteDeck:
    def __init__(self):
        self.deck = [10 if x > 10 else x for _ in range(0, 4) for x in range(1, 14)]

    def draw_cards(self, n: int):
        return [self.deck[random.randint(0, len(self.deck) - 1)] for _ in range(0, n)]

@dataclass
class State:
    def __init__(self, dealer_card: list[int], agent_cards: list[int]):
        self.dealer_sum = sum(dealer_card)
        self.agent_sum = sum(agent_cards)

@dataclass
class Episode:
    def __init__(self):
        self.deck = InfiniteDeck()
        self.state = State(self.deck.draw_cards(1), self.deck.draw_cards(2))
        self.value = 
    
    def hit(self):
        self.state.agent_sum += self.deck.draw_cards(1)[0]

    def stand(self):
        

