from visualisation import *
from pathlib import Path

from random_player import *
from hunter_player import *
from pichal_placer import *
from q_shooter import *
from policy_gradient_shooter import *
from hybrid_q_shooter import *
from configs import *
from monte_carlo_player import *
from hybrid_no_q_shooter import *

board_config = TINY_CONFIG
config_name = "tiny"

hqs = HybridQShooter(board_config)
hqs.load(f"models/{config_name}/hybrid_q_shooter")

hnqs = HybridNoQShooter(board_config)
hnqs.load(f"models/{config_name}/hybrid_no_q_shooter")

mc = MonteCarloShooter(1000)

def draw_pattern(shooter: Shooter):
    empty_board = Board(ShipBoard(board_config))
    for _ in range(board_config.size ** 2 // 3):
        empty_board.shoot(shooter.shoot(empty_board))
    return np.argmax(empty_board.get_repr(), axis = 2)

pattern = draw_pattern(hnqs)
plt.imshow(pattern)
path = f'plots/patterns/{config_name}'
Path(path).mkdir(parents=True, exist_ok=True)
plt.savefig(f"plots/patterns/{config_name}/hybrid_no_q_shooter.png")
