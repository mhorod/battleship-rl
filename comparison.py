
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

hqs = HybridQShooter(board_config)
hqs.load("models/tiny/hybrid_q_shooter")

qs = QShooter(board_config)
qs.load("models/tiny/q_shooter")

hnqs = HybridNoQShooter(board_config)
hnqs.load("models/tiny/hybrid_no_q_shooter")

hunter = HunterShooter()
even_hunter = EvenHunterShooter()

result = compare_placer_with_shooter(RandomPlacer(board_config), hqs)
print("Hybrid Q Shooter:\tavg={}\tstd={}".format(np.average(result), np.std(result)))

result = compare_placer_with_shooter(RandomPlacer(board_config), qs)
print("Q Shooter:\tavg={}\tstd={}".format(np.average(result), np.std(result)))

result = compare_placer_with_shooter(RandomPlacer(board_config), hnqs)
print("Hybrid No Q Shooter:\tavg={}\tstd={}".format(np.average(result), np.std(result)))

result = compare_placer_with_shooter(RandomPlacer(board_config), hunter)
print("Hunter Shooter:\tavg={}\tstd={}".format(np.average(result), np.std(result)))

result = compare_placer_with_shooter(RandomPlacer(board_config), even_hunter)
print("Even Hunter Shooter:\tavg={}\tstd={}".format(np.average(result), np.std(result)))
