import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sc2rl
from sc2rl.config.ConfigBase import ConfigBase

if __name__ == "__main__":

    conf = sc2rl.config.ConfigBase()