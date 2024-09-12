from functools import cached_property
import numpy as np
from pyboy.utils import WindowEvent
from pyboy_environment.environments.pokemon.pokemon_environment import PokemonEnvironment
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
        self.start_location = None  # Track the start location
        self.previous_location = None  # Track the previous location
        self.previous_num_locs = 0  # Track the previous number of discovered locations
        self.max_dist = 0
        self.prev_distance = 0

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
        current_location = game_stats["location"]
        # If the start_location is None (first discovery of a new map), set it
        if self.start_location is None:
            self.start_location = (current_location["x"], current_location["y"])

        # Compute Euclidean distance from the start location
        if self.start_location:
            start_x, start_y = self.start_location
            current_x, current_y = current_location["x"], current_location["y"]
            distance = np.sqrt((current_x - start_x) ** 2 + (current_y - start_y) ** 2)
        else:
            distance = 0.0  # Default to 0 if there's no start location

        # Convert game_stats to a flat list or array and prepend the distance
        state_array = [
            distance, # distance from start
            game_stats["party_size"],
            len(game_stats["pokemon"]),  # Assuming pokemon is a list
            # np.mean(game_stats["levels"]),  # Example: mean level of pokemon
            # np.mean(game_stats["hp"]),  # Example: mean hp of pokemon
            # np.mean(game_stats["xp"]),  # Example: mean xp of pokemon
            # np.mean(game_stats["status"]),  # Example: mean status of pokemon
            game_stats["badges"],
            game_stats["caught_pokemon"],
            game_stats["seen_pokemon"],
            game_stats["money"]
        ]
        
        return state_array

    def _calculate_reward(self, new_state: dict) -> float:
        # Get the current location
        current_location = new_state["location"]
        location_tuple = (current_location["x"], current_location["y"], current_location["map"])

        reward = 0.0 # initialise reward as 0

        # Calculate distance traveled if start_location is set
        if self.start_location:
            start_x, start_y = self.start_location
            current_x, current_y = current_location["x"], current_location["y"]
            distance = np.sqrt((current_x - start_x) ** 2 + (current_y - start_y) ** 2)
        else:
            distance = 0.0

        # if location hasn't changed, penalise
        if current_location == self.previous_location:
            reward = -0.1
        # if location already visited, penalise
        elif location_tuple in self.discovered_locations:
            reward = -0.01  # Penalize for visiting the same location
        else:
            # Reward logic for discovering new map or new location
            if current_location["map"] not in self.discovered_maps and current_location["map"] != "OAKS_LAB,":
                self.discovered_maps.add(current_location["map"])
                print(f"Discovered new map: {current_location['map']}")
                # Set the start_location for the new map
                self.start_location = (current_location["x"], current_location["y"])
                print(self.start_location)
                self.max_dist = 0 # reset distance
                reward = 10.0  # Reward for discovering a new map

        # Update the previous location
        self.previous_location = current_location
        
        # reward greater distance, penalise same distance
        if distance == self.prev_distance:
            reward = -1.0
            print(self.start_location)
            print(current_location)
            print(distance)
        elif distance > self.max_dist:
            reward += 10.0 # achieved new distance, give reward
            reward += distance # still reward bigger distances
        elif distance < self.max_dist:
            reward += 5.0
            reward += distance

        self.prev_distance = distance
        return reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if the agent beats the first gym
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # End the game if too many steps are taken
        return self.steps >= 1000
