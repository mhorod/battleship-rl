
from pathlib import Path

import matplotlib.pyplot as plt

from configs import *
from stats import *

from pichal_placer import *
from neural_network_player import *
from random_player import *
from hunter_player import *
from monte_carlo_player import *
from actor_critic_player import *
from naive_bayes_player import *
from actor_critic_player import *
from q_shooter import *
from policy_gradient_shooter import *
from hybrid_q_shooter import *
from configs import *
from hybrid_no_q_shooter import *


tiny_hqs = HybridQShooter(TINY_CONFIG)
tiny_hqs.load(f"models/tiny/hybrid_q_shooter")

tiny_qs = QShooter(TINY_CONFIG)
tiny_qs.load(f"models/tiny/q_shooter")

tiny_hnqs = HybridNoQShooter(TINY_CONFIG)
tiny_hnqs.load(f"models/tiny/hybrid_no_q_shooter")

small_hqs = HybridQShooter(SMALL_CONFIG)
small_hqs.load(f"models/small/hybrid_q_shooter")

small_qs = QShooter(SMALL_CONFIG)
small_qs.load(f"models/small/q_shooter")

small_hnqs = HybridNoQShooter(SMALL_CONFIG)
small_hnqs.load(f"models/small/hybrid_no_q_shooter")

standard_hqs = HybridQShooter(STANDARD_CONFIG)
standard_hqs.load(f"models/standard/hybrid_q_shooter")

standard_qs = QShooter(STANDARD_CONFIG)
standard_qs.load(f"models/standard/q_shooter")

standard_hnqs = HybridNoQShooter(STANDARD_CONFIG)
standard_hnqs.load(f"models/standard/hybrid_no_q_shooter")

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

    (tiny_qs, "Q Shooter"),
    (tiny_hqs, "Hybrid Q Shooter"),
    (tiny_hnqs, "Hybrid No Q Shooter"),
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

    (small_qs, "Q Shooter"),
    (small_hqs, "Hybrid Q Shooter"),
    (small_hnqs, "Hybrid No Q Shooter"),
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

    (standard_qs, "Q Shooter"),
    (standard_hqs, "Hybrid Q Shooter"),
    (standard_hnqs, "Hybrid No Q Shooter"),
]

all_models = [
    (tiny_models, TINY_CONFIG, 'tiny'),
    (small_models, SMALL_CONFIG, 'small'),
    (standard_models, STANDARD_CONFIG, 'standard'),
]

def plot_models(models, config, board_name):
    path = f'plots/pichal/{board_name}'
    Path(path).mkdir(parents=True, exist_ok=True)

    random_placer = RandomPlacer(config)
    for shooter, model_name in models:
        placer = PichalPlacer(config)
        placer.train(shooter)
        lengths = compare_placer_with_shooter(placer, shooter, 100)
        random_lengths = compare_placer_with_shooter(random_placer, shooter, 100)
        print(f'{model_name} model on {board_name} board with pichal: mean = {np.mean(lengths)}, std = {np.std(lengths)}')
        print(f'{model_name} model on {board_name} board with random: mean = {np.mean(random_lengths)}, std = {np.std(random_lengths)}')

        fig, ax = plt.subplots()
        max_moves = config.size * config.size
        ax.hist(lengths, bins=range(0, (max_moves//15 + 1)*15, max_moves//15), alpha=0.5)
        ax.hist(random_lengths, bins=range(0, (max_moves//15 + 1)*15, max_moves//15), alpha=0.5)
        ax.set_xlabel('Game length')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Game length distribution for {model_name} model on {board_name} board')
        fig.tight_layout()
        fig.savefig(f'{path}/{model_name}.png')


def plot_all_models():
    for models, config, board_name in all_models:
        plot_models(models, config, board_name)

plot_all_models()