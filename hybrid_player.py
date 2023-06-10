from player import *
from humanlike_player import HumanLikeShooter

class HybridShooter(Shooter):
    def __init__(self, model):
        self.model = model
        self.humanlike = HumanLikeShooter()

    def shoot(self, board):
        if board.count(Tile.HIT) == 0:
            return self.model.shoot(board)
        else:
            return self.humanlike.shoot(board)
