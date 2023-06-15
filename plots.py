import matplotlib.pyplot as plt

from random_player import *
from stats import *

game_lengths = compare_placer_with_shooter(
    RandomPlacer(),
    RandomShooter(),
    1000
)

plt.plot(game_lengths)
plt.show()

plt.hist(game_lengths)
plt.show()
