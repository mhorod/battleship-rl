from game import *

STANDARD_CONFIG = BoardConfig(10, [ShipConfig(5, 1), ShipConfig(4, 1), ShipConfig(3, 2), ShipConfig(2, 1)])
SMALL_CONFIG = BoardConfig(7, [ShipConfig(5, 1), ShipConfig(4, 1), ShipConfig(3, 1)])
TINY_CONFIG = BoardConfig(5, [ShipConfig(3, 1), ShipConfig(2, 1)])

CONFIGS = [TINY_CONFIG, SMALL_CONFIG, STANDARD_CONFIG]