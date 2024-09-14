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
        self.discovered_locations_episode = set()
        self.discovered_maps = set()
        self.discovered_maps_episode = set()
        self.start_location = None  # Track the start location
        self.previous_locations = []  # Track the previous three locations
        self.max_dist_episode = 0
        self.max_dist = 0
        self.prev_distance = 0

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            # WindowEvent.PRESS_BUTTON_START,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            # WindowEvent.RELEASE_BUTTON_START,
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

        # get map constant, reverse dictionary
        reverse_map_locations = {v: k for k, v in pkc.map_locations.items()}
        map_loc = reverse_map_locations.get(current_location["map"], -1)
        # Convert game_stats to a flat list or array and prepend the distance
        state_array = [
            distance, # distance from start
            map_loc, # map location ie OAK'S LAB, PALLETTOWN
            # game_stats["seen_pokemon"]# seen pokemon here
        ]
        
        return state_array

    def _calculate_reward(self, new_state: dict) -> float:
        # Get the current location and turn into a tuple
        current_location = new_state["location"]
        location_tuple = (current_location["x"], current_location["y"], current_location["map"])
        reward = 0.0  # Initialize reward as 0
        distance = 0
        # Calculate distance traveled if start_location is set
        if self.start_location:
            start_x, start_y = self.start_location
            current_x, current_y = current_location["x"], current_location["y"]
            distance = np.sqrt((current_x - start_x) ** 2 + (current_y - start_y) ** 2)

        if location_tuple in self.previous_locations:
            reward -= 0.2  # Penalize for revisiting any of the last three locations

        # Penalize if location already visited
        elif location_tuple in self.discovered_locations_episode:
            reward -= 0.01  # Penalize for revisiting
    
        # add new map (across episodes)
        if current_location["map"] not in self.discovered_maps_episode:
            print(f"============ {current_location['map']} discovered in episode ============")
            self.discovered_maps_episode.add(current_location["map"])
            reward += 100.0 * len(self.discovered_maps_episode)

            if current_location["map"] not in self.discovered_maps:
                print(f"============ {current_location['map']} discovered FOR THE FIRST TIME ============")
                self.discovered_maps.add(current_location["map"])
                print(distance)
                reward += 300.0 * len(self.discovered_maps)
                print(f"discovered maps: {len(self.discovered_maps)}")
            
            # update start location
            self.start_location = (current_location["x"], current_location["y"])
            distance = 0
            print(self.start_location)
            self.max_dist_episode = 0
        
        if location_tuple not in self.discovered_locations_episode:
            self.discovered_locations_episode.add(location_tuple) # record location this episode
            reward += 10.0
        if location_tuple not in self.discovered_locations:
            # print("found new location")
            reward += 50.0 # reward for finding a new location, never before seen in any episode
            self.discovered_locations.add(location_tuple)

        # Distance-based rewards
        if distance == self.prev_distance:
            reward -= 1.0  # Penalize if distance from start hasn't increased
        elif distance > self.max_dist_episode:
            # greater reward if distance is greater than all episodes
            if distance > self.max_dist:
                reward += 100.0
                reward += distance * 2
                self.max_dist = distance
            else:
                reward += 50.0  # Reward for achieving a new max distance in this episode
                reward += distance
            self.max_dist_episode = distance
        elif distance < self.max_dist_episode:
            reward -= 1.0  # Reward for moving but getting closer
            reward += distance # still reward for being further away

        # Update the previous distance and location
        self.prev_distance = distance
        self.previous_location = current_location

        # Update previous locations list (keep only the last three)
        if len(self.previous_locations) >= 3:
            self.previous_locations.pop(0)  # Remove the oldest location
        self.previous_locations.append(location_tuple)

        return reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        if game_stats["badges"] > self.prior_game_stats["badges"]:
            self.reset_episode()
            return True
        return False

    def _check_if_truncated(self, game_stats: dict) -> bool:
        if self.steps >= 1000:
            self.reset_episode()
            return True
        return False
    
    def reset_episode(self):
        print("resetting episode")
        self.discovered_locations_episode.clear()
        self.discovered_maps_episode.clear()
        self.max_dist_episode = 0
        print(f"{self.max_dist}")
