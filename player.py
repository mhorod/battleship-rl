from abc import ABC, abstractmethod

import numpy as np

from game import  *

class Placer(ABC):
    '''
    Returns a board with ships placed on it.
    '''
    @abstractmethod
    def place_ships(self) -> Board:
        pass

class Shooter(ABC):
    '''
    Returns a position to shoot at.
    '''
    @abstractmethod
    def shoot(self, board) -> tuple:
        pass

    def shoot_many(self, boards) -> list:
        return [self.shoot(board) for board in boards]

class ShipPredictor(ABC):
    '''
    Agent that for each field returns likelihood of it containing a ship.
    '''

    @abstractmethod
    def predict_ships(self, board) -> np.ndarray:
        '''
        For each field return a value in [0, 1] where 1 means high confidence
        that the field contains a ship and 0 means high confidence that it is empty.
        '''
        pass

    def predict_ships_many(self, boards) -> list:
        return [self.predict_ships(board) for board in boards]