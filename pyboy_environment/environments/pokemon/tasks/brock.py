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
        self.prev_distance_t = 0
        self.current_location = None # record current location
        self.currrent_state = None
        self.prev_state = None
        self.found_map = False
        self.seen_pokemon_episode = 0
        self.seen_pokemon = 0
        self.target_loc = [None] * 248

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
        self.current_state = game_stats 
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
            # distance, # distance from start
            map_loc, # map location ie OAK'S LAB, PALLETOWN
            # game_stats["seen_pokemon"], # seen pokemon here
            self.current_location["x"],
            self.current_location["y"]
        ]
        
        return state_array

    def _calculate_reward(self, new_state: dict) -> float:
        # ========== LOCATION LOGIC ==========  
        # Get the current location and turn into a tuple
        map_loc = self.current_location["map_id"]
        location_tuple = (self.current_location["x"], self.current_location["y"], self.current_location["map"], self.current_location["map_id"])
        reward = 0.0  # Initialize reward as 0
        distance = 0
        distance_t = 0
        
        # Calculate distance traveled if start_location is set for each map 
        if self.start_location[map_loc]:
            start_x, start_y = self.start_location[map_loc]
            current_x, current_y = self.current_location["x"], self.current_location["y"]
            distance = np.sqrt((current_x - start_x) ** 2 + (current_y - start_y) ** 2)

        # print(self._read_events())
        # calculate location and map rewards
        if self.target_loc[map_loc] != (0, 0):
            reward += self.check_location_rewards(map_loc, location_tuple, distance)
        else:
            distance_t = self.calculate_distance(self.target_loc[map_loc][0], self.target_loc[map_loc][1], location_tuple[0], location_tuple[1])
            reward += self.target_rewards(map_loc, distance_t, self.prev_distance_t)

        # Penalize swapping between the same two maps
        reward += self.check_map_swap(map_loc)

        # Calculate Pokemon related rewards
        reward += self.check_pokemon_rewards()

        print(f"start: {self.start_location[map_loc]}")
        print(f"current: {self.current_location}")
        print(f"distance: {distance}")
        print(f"distance_t: {distance_t}")
        print(f"target_loc: {self.target_loc[map_loc]}")
        print(f"max dist {map_loc}: {self.max_dist_episode[map_loc]}")
        input("pause")
            
        # ========== UPDATE LOGIC ==========
        # Update the previous distance and location
        self.prev_distance = distance
        self.previous_location = self.current_location
        self.prev_state = self.current_state
        self.prev_distance_t = distance_t
        return reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        if game_stats["badges"] > self.prior_game_stats["badges"]:
            self.reset_episode()
            return True
        return False

    def _check_if_truncated(self, game_stats: dict) -> bool:
        if self.steps >= 2000:
            self.reset_episode()
            return True
        return False
    
    def reset_episode(self):
        print("resetting episode")
        self.discovered_locations_episode.clear()
        self.discovered_maps_episode.clear()
        self.seen_pokemon_episode = 0
        print(f"max distance: {self.max_dist[self.current_location['map_id']]}")
        self.max_dist_episode = np.zeros(248) # reset all max distances this episode 

    def check_location_rewards(self, map_loc, location_tuple, distance):
        reward = 0

        if self.previous_locations:
            # penalise for the same position, except when on grass
            if self.current_location == self.previous_locations[0] and self._is_grass_tile() == False:
                reward -= 5.0
        
        # Handle new map discovery (across episodes)
        if self.current_location["map"] not in self.discovered_maps_episode or self.found_map:
            # update target loc- use previous location
            if map_loc != 40:
                self.target_loc[self.previous_locations[1][3]] = (self.previous_locations[1][0], self.previous_locations[1][1])
                print(f"target loc: {self.target_loc[self.previous_locations[1][3]]}")

            if not self.found_map:
                self.found_map = True  # Set the flag to wait for the next tick
                self.max_dist_episode[map_loc] = 0 
            else:
                self.found_map = False  # Reset the flag
                self.discovered_maps_episode.add(self.current_location["map"])
                reward += 100.0 * len(self.discovered_maps_episode)

                # First discovery of this map across all episodes
                if self.current_location["map"] not in self.discovered_maps:
                    print(f"============ {self.current_location['map']} discovered FOR THE FIRST TIME ============")
                    self.discovered_maps.add(self.current_location["map"])
                    reward += 300.0 ** len(self.discovered_maps)

                # Update start location for the new map
                self.start_location[map_loc] = (self.current_location["x"], self.current_location["y"])
                self.max_dist_episode[map_loc] = 0
                print(f"New start location set for map {map_loc}: {self.start_location[map_loc]}")

        # Handle location discovery within the map
        if location_tuple not in self.discovered_locations_episode:
            self.discovered_locations_episode.add(location_tuple)
            distance_bonus = distance * 0.1  # Add a bonus based on distance
            reward += 10.0 + distance_bonus  # Reward for finding a new location with distance bonus

        if location_tuple not in self.discovered_locations:
            distance_bonus = distance * 0.2  # Larger bonus for all-time discoveries
            reward += 50.0 + distance_bonus  # Reward for finding a new location never seen in any episode
            self.discovered_locations.add(location_tuple)
            if distance > self.max_dist[map_loc]:
                self.max_dist[map_loc] = distance # set distance as max dist
                reward += 50.0

        # no reward for already visited locations

        # Update previous locations (keeping only the last three)
        if len(self.previous_locations) >= 3:
            self.previous_locations.pop(0)  # Remove the oldest location
        
        self.previous_locations.append(location_tuple)
        return reward

    def check_map_swap(self, map_loc):
        reward = 0
        # Penalize swapping between the same two maps
        if len(self.previous_locations) >= 3:
            first_map = self.previous_locations[0][3]
            second_map = self.previous_locations[1][3]  # Get map_id from two steps ago
            third_map = self.previous_locations[2][3]  # Get map_id from the previous step
            if first_map == third_map and first_map != second_map:
                reward -= 100.0  # Penalize for swapping maps back and forth
                print("Swapping between the same two maps detected")
        return reward
    
    def check_pokemon_rewards(self):
        reward = 0

        # Ensure prev_state is available for comparison
        if self.prev_state:
            # Found new Pokémon in the current state- counts unique pokemon
            if self._is_grass_tile():
                reward += 1.0
            
            if self.current_state["seen_pokemon"] > self.prev_state["seen_pokemon"]:
                print(f"currrent state seen: {self.current_state['seen_pokemon']}")
                print(f"previous state seen: {self.prev_state['seen_pokemon']}")
                print(f"episode seen: {self.seen_pokemon_episode}")
                print(f"max seen: {self.seen_pokemon}")

                reward += 1000.0
                print("============== Found new UNIQUE Pokémon ==============")
                
                # Update the number of Pokémon seen in the current episode
                self.seen_pokemon_episode = self.current_state["seen_pokemon"]
                
                # If the number of Pokémon seen in this episode exceeds the global record
                if self.seen_pokemon_episode > self.seen_pokemon:
                    reward += 1000.0
                    self.seen_pokemon = self.seen_pokemon_episode
                    print("============== Max new UNIQUE Pokémon, giving reward ==============")
        
        # Ensure that `seen_pokemon_episode` is always updated correctly for future comparisons
        self.seen_pokemon_episode = max(self.seen_pokemon_episode, self.current_state["seen_pokemon"])

        if self.prev_state:
            # CATCH POKEMON REWARDS
            if self.current_state["caught_pokemon"] > self.prev_state["caught_pokemon"]:
                reward += 10000.0
                print("========== Caught a Pokemon! ==========")

        return reward

    def target_rewards(map_loc, distance, prev_dist):
        if distance > prev_dist:
            reward += 100.0
            return reward
        return 0

    def calculate_distance(target_x, target_y, current_x, current_y):
        distance = np.sqrt((current_x - target_x) ** 2 + (current_y - target_y) ** 2)
        return distance