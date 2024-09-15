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
        self.start_location = [None] * 248  # Track the start location of each map/ room
        self.previous_locations = []  # Track the previous three locations
        self.max_dist_episode = np.zeros(248)  # Record max distance per map/ room
        self.max_dist = np.zeros(248)          # Record max distance per map/ room
        self.prev_distance = 0
        self.current_location = None # record current location
        self.prev_state = None

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
        self.current_location = game_stats["location"]
        # get map number
        map_loc = self.current_location["map_id"]
        # If the start_location is None (first discovery of a new map), set it
        if self.start_location[map_loc] is None:
            self.start_location[map_loc] = (self.current_location["x"], self.current_location["y"])

        # Compute Euclidean distance from the start location
        if self.start_location[map_loc]:
            start_x, start_y = self.start_location[map_loc]
            current_x, current_y = self.current_location["x"], self.current_location["y"]
            distance = np.sqrt((current_x - start_x) ** 2 + (current_y - start_y) ** 2)
        else:
            distance = 0.0  # Default to 0 if there's no start location

        # Convert game_stats to a flat list or array and prepend the distance
        state_array = [
            distance, # distance from start
            map_loc, # map location ie OAK'S LAB, PALLETTOWN
            game_stats["seen_pokemon"]# seen pokemon here
        ]
        
        self.prev_state = game_stats # remember previous state
        return state_array

    def _calculate_reward(self, new_state: dict) -> float:
        # ========== LOCATION LOGIC ==========  
        # Get the current location and turn into a tuple
        # current_location = new_state["location"]
        map_loc = self.current_location["map_id"]
        location_tuple = (self.current_location["x"], self.current_location["y"], self.current_location["map"], self.current_location["map_id"])
        reward = 0.0  # Initialize reward as 0
        distance = 0
        # Calculate distance traveled if start_location is set for each map 
        if self.start_location[map_loc]:
            start_x, start_y = self.start_location[map_loc]
            current_x, current_y = self.current_location["x"], self.current_location["y"]
            distance = np.sqrt((current_x - start_x) ** 2 + (current_y - start_y) ** 2)

        if location_tuple in self.previous_locations:
            reward -= 0.2  # Penalize for revisiting any of the last three locations
        elif location_tuple in self.discovered_locations_episode:  # Penalize if location already visited
            reward -= 0.01  # Penalize for revisiting
    
        # add new map (across episodes)
        if self.current_location["map"] not in self.discovered_maps_episode:
            print(f"============ {self.current_location['map']} discovered in episode ============")
            self.discovered_maps_episode.add(self.current_location["map"])
            reward += 100.0 * len(self.discovered_maps_episode)

            if self.current_location["map"] not in self.discovered_maps:
                print(f"============ {self.current_location['map']} discovered FOR THE FIRST TIME ============")
                self.discovered_maps.add(self.current_location["map"])
                print(distance)
                reward += 300.0 * len(self.discovered_maps)
                print(f"discovered maps: {len(self.discovered_maps)}")
            
            # update start location for a new map and reset max distances
            self.start_location[map_loc] = (self.current_location["x"], self.current_location["y"])
            distance = 0
            self.max_dist_episode[map_loc] = 0
            self.max_dist[map_loc] = 0
            print(self.start_location[map_loc])
        
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
        elif distance > self.max_dist_episode[map_loc]:
            self.max_dist_episode[map_loc] = distance
            # greater reward if distance is greater than all episodes
            if distance > self.max_dist[map_loc]:
                reward += 100.0 + (distance * 2)
                self.max_dist[map_loc] = distance
            else:
                reward += 50.0 + distance  # Reward for achieving a new max distance in this episode
        elif distance < self.max_dist_episode[map_loc]:
            reward -= 1.0 #  distance # Reward for moving but getting closer

         # Update previous locations list (keep only the last three)
        if len(self.previous_locations) >= 3:
            self.previous_locations.pop(0)  # Remove the oldest location
        self.previous_locations.append(location_tuple)

        # ========== EXPLORATION LOGIC ==========
        
        # ========== UPDATE LOGIC ==========
        # Update the previous distance and location
        self.prev_distance = distance
        self.previous_location = self.current_location

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
        self.max_dist_episode = np.zeros(248) # reset all max distances this episode 
        print(f"max distance: {self.max_dist[self.current_location['map_id']]}")
