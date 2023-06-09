from abc import ABC, abstractmethod
from game import *

class Placer(ABC):
    @abstractmethod
    def place_ships(self) -> Board:
        pass

class Shooter(ABC):
    @abstractmethod
    def shoot(self, board) -> tuple:
        pass

    def shoot_many(self, boards) -> list:
        return [self.shoot(board) for board in boards]
