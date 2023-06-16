from pathlib import Path

import matplotlib.pyplot as plt

from configs import *
from stats import *

from neural_network_player import *
from random_player import *
from hunter_player import *
from monte_carlo_player import *
from actor_critic_player import *
from naive_bayes_player import *
from actor_critic_player import *


tiny_models = [
    (load_actor_critic_shooter("models/tiny/actor_critic"), "Actor Critic"),
    (load_actor_critic_shooter("models/tiny/hybrid_actor_critic"), "Hybrid Actor Critic"),

    (load_naive_bayes_model("models/tiny/naive_bayes.model"), "Naive Bayes"),

    (RandomShooter(), "Random"),

    (HunterShooter(), "Hunter"),
    (EvenHunterShooter(), "Even Hunter"),

    (load_neural_network_model("models/tiny/neural_network/perceptron.model"), "Perceptron"),
    (load_neural_network_model("models/tiny/neural_network/dense.model"), "Dense"),
    (load_neural_network_model("models/tiny/neural_network/cnn.model"), "CNN"),

    (MonteCarloShooter(1000), "Monte Carlo"),
]

small_models = [
    (load_actor_critic_shooter("models/small/actor_critic"), "Actor Critic"),
    (load_actor_critic_shooter("models/small/hybrid_actor_critic"), "Hybrid Actor Critic"),

    (load_naive_bayes_model("models/small/naive_bayes.model"), "Naive Bayes"),

    (RandomShooter(), "Random"),

    (HunterShooter(), "Hunter"),
    (EvenHunterShooter(), "Even Hunter"),

    (load_neural_network_model("models/small/neural_network/perceptron.model"), "Perceptron"),
    (load_neural_network_model("models/small/neural_network/dense.model"), "Dense"),
    (load_neural_network_model("models/small/neural_network/cnn.model"), "CNN"),

    (MonteCarloShooter(1000), "Monte Carlo"),
]

standard_models = [
    (load_actor_critic_shooter("models/standard/actor_critic"), "Actor Critic"),
    (load_actor_critic_shooter("models/standard/hybrid_actor_critic"), "Hybrid Actor Critic"),

    (load_naive_bayes_model("models/standard/naive_bayes.model"), "Naive Bayes"),

    (RandomShooter(), "Random"),

    (HunterShooter(), "Hunter"),
    (EvenHunterShooter(), "Even Hunter"),

    (load_neural_network_model("models/standard/neural_network/perceptron.model"), "Perceptron"),
    (load_neural_network_model("models/standard/neural_network/dense.model"), "Dense"),
    (load_neural_network_model("models/standard/neural_network/cnn.model"), "CNN"),

    (MonteCarloShooter(1000), "Monte Carlo"),
]

all_models = [
    (tiny_models, TINY_CONFIG, 'tiny'),
    (small_models, SMALL_CONFIG, 'small'),
    (standard_models, STANDARD_CONFIG, 'standard'),
]

def plot_models(models, config, board_name):
    path = f'plots/histograms/{board_name}'
    Path(path).mkdir(parents=True, exist_ok=True)

    placer = RandomPlacer(config)
    for shooter, model_name in models:
        lengths = compare_placer_with_shooter(placer, shooter, 100)
        mean = np.mean(lengths)
        std = np.std(lengths)
        print(f'{model_name} model on {board_name} board: mean = {mean}, std = {std}')

        fig, ax = plt.subplots()
        max_moves = config.size * config.size
        ax.hist(lengths, bins=range(0, max_moves, 5))
        ax.set_xlabel('Game length')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Game length distribution for {model_name} model on {board_name} board')
        fig.tight_layout()
        fig.savefig(f'{path}/{model_name}.png')


def plot_all_models():
    for models, config, board_name in all_models:
        plot_models(models, config, board_name)

plot_all_models()