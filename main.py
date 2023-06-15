from visualisation import *

from random_player import *
from montecarlo_player import *
from tf_player import *
from hunter_player import *

board_config = DEFAULT_BOARD_CONFIG
#board_config = TINY_BOARD_CONFIG
display_config = BoardDisplayConfig(board_config, tile_size = 40, margin = 40, info_height = 100)

placer = RandomPlacer(board_config)

shooter = EvenHunterShooter()
shooter_name = "Even Hunter"

shooter = load_model("models/default/cnn.model")
shooter_name = "CNN"

visualization = Visualization(display_config, placer, shooter, shooter_name)
visualization.run()