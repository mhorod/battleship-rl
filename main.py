from visualisation import *

from random_player import *
from montecarlo_player import *
from tf_player import *
from hunter_player import *
from pichal_placer import *
from q_shooter import *
from policy_gradient_shooter import *
from hybrid_q_shooter import *
from hybrid_no_q_shooter import *

board_config = TINY_BOARD_CONFIG
#board_config = TINY_BOARD_CONFIG
display_config = BoardDisplayConfig(board_config, tile_size = 40, margin = 40, info_height = 100)


# shooter = QShooter(board_config)
# shooter_name = "Deep Q Shooter"

shooter = HybridNoQShooter(board_config)
shooter_name = "Hybrid No Q Shooter"

placer = RandomPlacer(board_config)
shooter.load("models/tiny/hybrid_no_q_shooter")
# shooter.train(placer)
# shooter.save("models/tiny/hybrid_no_q_shooter")

# shooter = load_model("models/default/cnn.model")
# shooter_name = "CNN"

# pichal_placer = PichalPlacer(board_config)

# print(pichal_placer.train(shooter))
# print(pichal_placer.train(shooter))
# print(pichal_placer.weights)

# plt.hist(compare_placer_with_shooter(pichal_placer, shooter))
# plt.hist(compare_placer_with_shooter(RandomPlacer(board_config), shooter))
# plt.show()



visualization = Visualization(display_config, placer, shooter, shooter_name)
visualization.run()