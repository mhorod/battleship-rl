
from visualisation import *

from random_player import *
from hunter_player import *
from pichal_placer import *
from q_shooter import *
from policy_gradient_shooter import *
from hybrid_q_shooter import *
from configs import *
from hybrid_no_q_shooter import *

board_config = TINY_CONFIG
config_name = "tiny"

hqs = HybridQShooter(board_config)
hqs.load(f"models/{config_name}/hybrid_q_shooter")

qs = QShooter(board_config)
qs.load(f"models/{config_name}/q_shooter")

hnqs = HybridNoQShooter(board_config)
hnqs.load(f"models/{config_name}/hybrid_no_q_shooter")

hunter = HunterShooter()
even_hunter = EvenHunterShooter()

result = compare_placer_with_shooter(RandomPlacer(board_config), hqs)
print("Hybrid Q Shooter:\tavg={}\tstd={}".format(np.average(result), np.std(result)))
plt.hist(result)
plt.xlabel('Game length')
plt.ylabel('Frequency')
plt.title(f'Game length distribution for Hybrid Q Shooter model on {config_name} board')
plt.savefig(f"plots/histograms/{config_name}/Hybrid Q Shooter.png")
plt.clf()

result = compare_placer_with_shooter(RandomPlacer(board_config), qs)
print("Q Shooter:\tavg={}\tstd={}".format(np.average(result), np.std(result)))
plt.hist(result)
plt.xlabel('Game length')
plt.ylabel('Frequency')
plt.title(f'Game length distribution for Q Shooter model on {config_name} board')
plt.savefig(f"plots/histograms/{config_name}/Q Shooter.png")
plt.clf()

result = compare_placer_with_shooter(RandomPlacer(board_config), hnqs)
print("Hybrid No Q Shooter:\tavg={}\tstd={}".format(np.average(result), np.std(result)))
plt.hist(result)
plt.xlabel('Game length')
plt.ylabel('Frequency')
plt.title(f'Game length distribution for Hybrid No Q Shooter model on {config_name} board')
plt.savefig(f"plots/histograms/{config_name}/Hybrid No Q Shooter.png")
plt.clf()