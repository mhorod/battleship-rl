import matplotlib.pyplot as plt

from random_player import *
from stats import *
from tf_player import *


config = TINY_BOARD_CONFIG
config = DEFAULT_BOARD_CONFIG
placer = RandomPlayer(config)
shooter = load_model('models/default/cnn.model')

game_lengths = compare_placer_with_shooter(
    placer,
    shooter,
    100
)

print(f"Average game length: {sum(game_lengths) / len(game_lengths)}")
plt.hist(game_lengths)
plt.show()
