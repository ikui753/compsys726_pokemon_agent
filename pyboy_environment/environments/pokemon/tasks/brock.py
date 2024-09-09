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
        # Implement your state retrieval logic here
        location = self._get_location()
        game_stats = self._generate_game_stats()
        game_area = self.game_area()
        return [game_stats["badges"]]

    def _calculate_reward(self, new_state: dict) -> float:
        # Retrieve the current location and store in a tuple 
        # input("pause")       
        current_location = self._get_location()
        location_tuple = (current_location["x"], current_location["y"], current_location["map"])
        # print(f"{current_location['x']}, {current_location['y']} - {current_location['map']}")

        # Check if the agent is in a new map
        if current_location["map"] not in self.discovered_maps:
            self.discovered_maps.add(current_location["map"])
            print(f"{current_location['map']}")
            return 10.0  # Reward for discovering a new map

        # If the current map is "OAKS_LAB"
        if current_location['map'] == 'OAKS_LAB,':
            # input("pause")
            # Get the maximum y value from discovered locations in "OAKS_LAB"
            max_y = 0
            for loc in self.discovered_locations:
                if loc[2] == "OAKS_LAB,":
                    max_y = max(max_y, loc[1])
            
            # If current y is less than the minimum y value discovered
            if current_location["y"] > max_y:
                # Add location to discovered locations
                self.discovered_locations.add(location_tuple)
                print("Found new lowest loc")
                return 3.0  # Higher reward for discovering a new lowest location
        
        elif current_location["map"] == 'PALLET_TOWN,':
            min_y = 0
            for loc in self.discovered_locations:
                if loc[2] == "PALLET_TOWN,":
                    min_y = min(min_y, loc[1])
            if current_location["y"] < min_y:
                self.discovered_locations.add(location_tuple)
                print("Found new highest loc")
                return 3.0 # higher reward for discovering a new highest location
            
        # Check if the specific location (x, y) on the map is new
        if location_tuple not in self.discovered_locations:
            self.discovered_locations.add(location_tuple)
            return 1.0  # Reward for discovering a new location within the same map

        # If neither the map nor the location is new, return no reward
        return 0.0

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here
        return self.steps >= 1000
