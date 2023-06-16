from player import *

class PredictionShooter(Shooter):
    '''
    Selects the best position to shoot based on a prediction of where the ships are.
    '''
    def __init__(self, predictor: ShipPredictor):
        self.predictor = predictor

    def shoot(self, board) -> tuple:
        return self.shoot_many([board])[0]

    def shoot_many(self, boards) -> list:
        return [self.select_best_pos(board) for board in boards]

    def predict_raw(self, board) -> np.ndarray:
        return self.predictor.predict_ships(board)

    def predict_masked(self, board) -> np.ndarray:
        likelihoods = self.predict_raw(board)
        for pos in board_positions(board.config):
            if board[pos] != Tile.EMPTY:
                likelihoods[pos] = -10000
        return likelihoods

    def select_best_pos(self, board) -> tuple:
        likelihoods = self.predict_masked(board)
        max = np.max(likelihoods)
        indices = np.argwhere(likelihoods == max)
        return tuple(indices[np.random.choice(len(indices))])