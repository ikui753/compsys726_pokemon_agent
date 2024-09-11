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
        return [game_stats["badges"]]

    def _calculate_reward(self, new_state: dict) -> float:
        # Get the current location
        current_location = new_state["location"]

        # Compare with previous location
        if self.previous_location is not None and current_location == self.previous_location:
            reward = -1.0  # Penalize if staying in the same location

        # if location already visited, penalise
        # if current_location in self.discovered_locations:
        #     reward = -1.0
        else:
            # Reward logic for discovering new map or new location
            location_tuple = (current_location["x"], current_location["y"], current_location["map"])

            if current_location["map"] not in self.discovered_maps:
                self.discovered_maps.add(current_location["map"])
                print(f"Discovered new map: {current_location['map']}")
                reward = 500.0  # Reward for discovering a new map
            elif location_tuple not in self.discovered_locations:
                self.discovered_locations.add(location_tuple)
                print(f"Discovered new location: {location_tuple}")
                reward = 100.0  # Reward for discovering a new location
            else:
                reward = 0.0  # No reward if the location is not new

            # Update the previous location
            self.previous_location = current_location

        return reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if the agent beats the first gym
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # End the game if too many steps are taken
        return self.steps >= 1000
