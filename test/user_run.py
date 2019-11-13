import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer, Human


class WorkerRushBot(sc2.BotAI):
    async def on_step(self, iteration):
        if iteration == 0:
            for worker in self.workers:
                await self.do(worker.attack(self.enemy_start_locations[0]))


if __name__ == "__main__":
    map_name = "training_scenario_5_Human"

run_game(maps.get(map_name), [
    Human(Race.Terran),
    Computer(Race.Terran, Difficulty.Medium)
], realtime=True)
