# sc2rl

## Known issues and fixes
1. 'ValueError: 3794 is not a valid AbilityId' from pytho-sc2
- Comment 'assert self.id != 0' in game_data.py of python-sc2
2. AssertionError: Unsupported pixel density
- Comment assert self.bits_per_pixel % 8 == 0, "Unsupported pixel density"

## wandb (Weight AND Bias) setup 
```
$ pip install --upgrade wandb
$ wandb login eae3a2bffdf3f5cf9980f47b2232247823d8d5b5

```

## training scenario description (../Maps/) 

Training Scenario 1:
Training Scenario 2: