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
class Game:
    def __init__(self):
        self.deck = InfiniteDeck()
        self.dealer = self.deck.draw_cards(1)[0]
        self.agent = sum(self.deck.draw_cards(2))

