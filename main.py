from visualisation import *
from configs import *

from random_player import *
from monte_carlo_player import *
from neural_network_player import *
from hunter_player import *
from naive_bayes_player import *
from actor_critic_player import *

from pichal_placer import *
from q_shooter import *
from policy_gradient_shooter import *
from hybrid_q_shooter import *
from hybrid_no_q_shooter import *

def run_tiny():
    board_config = TINY_CONFIG
    config_name = "tiny"
    display_config = BoardDisplayConfig(board_config, tile_size = 40, margin = 40, info_height = 100)

    shooters = [
        (load_neural_network_model(f"models/{config_name}/neural_network/perceptron.model"), "Perceptron"),
        (load_neural_network_model(f"models/{config_name}/neural_network/dense.model"), "Dense"),
        (load_neural_network_model(f"models/{config_name}/neural_network/cnn.model"), "CNN"),

        (load_actor_critic_shooter(f"models/{config_name}/actor_critic/"), "Actor Critic"),
        (load_actor_critic_shooter(f"models/{config_name}/hybrid_actor_critic"), "Hybrid Actor Critic"),

        (load_hybrid_no_q_shooter(board_config, f"models/{config_name}/hybrid_no_q_shooter"), "Hybrid No Q Shooter"),

        (EvenHunterShooter(), "Even Hunter Shooter"),
        (MonteCarloShooter(1000), "Monte Carlo Shooter"),
    ]
    placer = RandomPlacer(board_config)

    visualization = Visualization(display_config, placer, shooters)
    visualization.run()

def run_small():
    board_config = SMALL_CONFIG
    config_name = "small"
    display_config = BoardDisplayConfig(board_config, tile_size = 40, margin = 40, info_height = 100)

    shooters = [
        (load_neural_network_model(f"models/{config_name}/neural_network/perceptron.model"), "Perceptron"),
        (load_neural_network_model(f"models/{config_name}/neural_network/dense.model"), "Dense"),
        (load_neural_network_model(f"models/{config_name}/neural_network/cnn.model"), "CNN"),

        (load_actor_critic_shooter(f"models/{config_name}/actor_critic/"), "Actor Critic"),
        (load_actor_critic_shooter(f"models/{config_name}/hybrid_actor_critic"), "Hybrid Actor Critic"),

        (EvenHunterShooter(), "Even Hunter Shooter"),
        (MonteCarloShooter(1000), "Monte Carlo Shooter"),
    ]
    placer = RandomPlacer(board_config)

    visualization = Visualization(display_config, placer, shooters)
    visualization.run()


def run_standard():
    board_config = STANDARD_CONFIG
    config_name = "standard"
    display_config = BoardDisplayConfig(board_config, tile_size = 40, margin = 40, info_height = 100)

    shooters = [
        (load_neural_network_model(f"models/{config_name}/neural_network/perceptron.model"), "Perceptron"),
        (load_neural_network_model(f"models/{config_name}/neural_network/dense.model"), "Dense"),
        (load_neural_network_model(f"models/{config_name}/neural_network/cnn.model"), "CNN"),

        (load_actor_critic_shooter(f"models/{config_name}/actor_critic/"), "Actor Critic"),
        (load_actor_critic_shooter(f"models/{config_name}/hybrid_actor_critic"), "Hybrid Actor Critic"),

        (EvenHunterShooter(), "Even Hunter Shooter"),
        (MonteCarloShooter(1000), "Monte Carlo Shooter"),
    ]
    placer = RandomPlacer(board_config)

    visualization = Visualization(display_config, placer, shooters)
    visualization.run()


#run_tiny()
#run_small()
run_standard()