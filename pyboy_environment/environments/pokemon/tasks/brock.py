from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc

class PokemonBrock(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:
        # Initialize the set to track discovered locations
        self.discovered_locations = set()
        self.discovered_maps = set()
        self.previous_location = None  # Track the previous location
        self.previous_num_locs = 0  # Track the previous number of discovered locations

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]

        super().__init__(
            act_freq=act_freq,
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )

    def _get_state(self) -> np.ndarray:
        # Retrieve the current game state
        game_stats = self._generate_game_stats()
        # print(game_stats["game_area"])
        num_locs = len(self.discovered_locations)
        
        # Return 0 if the number of locations hasn't changed
        if num_locs == self.previous_num_locs:
            return [0]
        else:
            self.previous_num_locs = num_locs  # Update the previous number of locations
            return [num_locs]

    def _calculate_reward(self, new_state: dict) -> float:
        # Get the current location
        current_location = new_state["location"]
        location_tuple = (current_location["x"], current_location["y"], current_location["map"])

        # if location hasn't changed, penalise
        if current_location == self.previous_location:
            reward = -10.0
        # if location already visited, penalise
        elif location_tuple in self.discovered_locations:
            reward = -1.0  # Penalize for visiting the same location
        else:
            # Reward logic for discovering new map or new location that is not oak's lab
            if current_location["map"] not in self.discovered_maps and current_location["map"] is not "OAKS_LAB,":
                self.discovered_maps.add(current_location["map"])
                print(f"Discovered new map: {current_location['map']}")
                reward = 10000.0  # Reward for discovering a new map
            else:
                self.discovered_locations.add(location_tuple)
                print(f"Discovered new location: {location_tuple}")
                reward = 1000.0  # Reward for discovering a new location

        # Update the previous location
        self.previous_location = current_location
        return reward


    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if the agent beats the first gym
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # End the game if too many steps are taken
        return self.steps >= 1000
