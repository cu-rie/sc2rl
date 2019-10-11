# Helper dictionary for encoding unit types: python-sc2 unit_typeid -> one-hot position

type2onehot = {9: 0,  # Baneling
               105: 1,  # Zergling
               48: 2,  # Marine
               51: 3,  # Marauder
               33: 4,  # Siegetank - tank mode
               32: 5,  # Siegetank - siege mode
               54: 6,  # Medivac
               73: 7,  # Zealot
               53: 8  # Hellion
               }

NUM_TOTAL_TYPES = len(type2onehot)

type2cost = {'Marine': [50, 0, 1.0],  # [Mineral, Vespene, Supply]
             'Zergling': [25, 0, 0.5],  # 1/2 of Zergling production cost
             'Baneling': [50, 25, 1.0],
             'Zealot': [100, 0, 2.0],
             'Hellion': [100, 0, 2.0]}
