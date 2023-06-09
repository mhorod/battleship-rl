from abc import ABC, abstractmethod

class Placer(ABC):
    @abstractmethod
    def place_ships(self):
        pass

class Shooter(ABC):
    @abstractmethod
    def shoot(self, board) -> tuple:
        pass

