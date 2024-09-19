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
        self.target_location = [(0,0)] * 248 # record target locations
        self.currrent_state = None
        self.prev_state = None
        self.found_map = False

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
            distance, # distance from start
            map_loc, # map location ie OAK'S LAB, PALLETOWN
            # game_stats["seen_pokemon"], # seen pokemon here
            # self.current_location["x"],
            # self.current_location["y"]
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
            distance = self.get_distance(self.start_location[map_loc][0], self.start_location[map_loc][1], self.current_location["x"], self.current_location["y"])

        print(f"target location {map_loc}: {self.target_location[map_loc]}")
        if self.target_location[map_loc] != (0,0):
            # get distance to target location
            distance = self.get_distance(self.start_location[map_loc][0], self.start_location[map_loc][1], self.target_location[map_loc][0], self.target_location[map_loc][1])
            print(f"distance to target ({self.target_location[map_loc]}): {distance}")
            reward += self.check_target_rewards(map_loc, location_tuple, distance)
        # calculate location and map rewards
        reward += self.check_location_rewards(map_loc, location_tuple, distance)

        # Penalize swapping between the same two maps
        reward += self.check_map_swap(map_loc)


        # ========== EXPLORATION LOGIC ==========
        if self.prev_state is not None and self.current_state["seen_pokemon"] > self.prev_state["seen_pokemon"]:
            # found pokemon
            reward += 200.0 * self.current_state["seen_pokemon"]
            print("========== Pokemon found! ==========")
            
        # ========== UPDATE LOGIC ==========
        # Update the previous distance and location
        self.prev_distance = distance
        self.previous_location = self.current_location

        print(f"start: {self.start_location[map_loc]}")
        print(f"current: {self.current_location}")
        print(distance)
        print(f"max dist {self.current_location['map']}: {self.max_dist_episode[map_loc]}") # this is incorrect
        print(f"reward: {reward}")
        input("pause")

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
        print(f"max distance: {self.max_dist[self.current_location['map_id']]}")
        self.max_dist_episode = np.zeros(248) # reset all max distances this episode 

    def check_location_rewards(self, map_loc, location_tuple, distance):
        reward = 0

        if self.previous_locations:
            # penalise for the same position
            if self.current_location == self.previous_locations[0]:
                reward -= 5.0

        # NOVEL EXPLORATION PHASE
        # Handle new map discovery (across episodes)
        if self.current_location["map"] not in self.discovered_maps_episode or self.found_map:

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

                    # update target location for this map
                    # store previous location for future runs to get to (target location):
                    if len(self.previous_locations) > 0:
                        # print("getting new target location")
                        self.target_location[self.previous_locations[0][3]] = self.previous_locations[0]
                        # print(f"target loc: {self.target_location[self.previous_locations[0][3]]}")

                # Update start location for the new map
                self.start_location[map_loc] = (self.current_location["x"], self.current_location["y"])
                self.max_dist_episode[map_loc] = 0
                print(f"New start location set for map {map_loc}: {self.start_location[map_loc]}")        

        # Handle location discovery within the map
        if location_tuple not in self.discovered_locations_episode:
            self.discovered_locations_episode.add(location_tuple)
            distance_bonus = distance * 0.1  # Add a bonus based on distance
            reward += 10.0 + distance_bonus  # Reward for finding a new location with distance bonus
            if distance > self.max_dist_episode[map_loc]:
                self.max_dist_episode[map_loc] = distance

            if location_tuple not in self.discovered_locations:
                distance_bonus = distance * 0.2  # Larger bonus for all-time discoveries
                reward += 50.0 # + distance_bonus  # Reward for finding a new location never seen in any episode
                self.discovered_locations.add(location_tuple)
                if distance > self.max_dist[map_loc]:
                    print("setting new max_dist for map_loc")
                    self.max_dist[map_loc] = distance # set distance as max dist
                    reward += distance_bonus

        # no reward for already visited locations

        # # Max distance reward logic
        # if distance > self.max_dist_episode[map_loc]:
        #     self.max_dist_episode[map_loc] = distance
        # if distance > self.max_dist[map_loc]:
        #     reward += 20.0 + (distance * 2)  # Greater reward for exceeding max distance across all episodes
        #     self.max_dist[map_loc] = distance
        # else:
        #     reward += distance  # Reward for achieving a new max distance in this episode

        # Update previous locations (keeping only the last three)
        if len(self.previous_locations) >= 3:
            self.previous_locations.pop(0)  # Remove the oldest location
        self.previous_locations.append(location_tuple)

        return reward


    def calculate_distance_rewards(self, distance, map_loc, location_tuple):
        reward = 0
        # Distance-based rewards
        if distance == self.prev_distance:
            reward -= 1.0  # Penalize if distance from start hasn't increased
        elif distance > self.max_dist_episode[map_loc]:
            self.max_dist_episode[map_loc] = distance
            # greater reward if distance is greater than all episodes- only reward for greater distances
            if distance > self.max_dist[map_loc]:
                reward += 20.0 + (distance * 2)
                self.max_dist[map_loc] = distance
            else:
                reward += distance  # Reward for achieving a new max distance in this episode
        # elif distance < self.max_dist_episode[map_loc]:
        #     reward -= 1.0 #  distance # Reward for moving but getting closer

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
    
    def get_distance(self, curr_x, curr_y, target_x, target_y):
        distance = np.sqrt((curr_x - target_x) ** 2 + (curr_y - target_y) ** 2)
        return distance
    
    def check_target_rewards(self, map_loc, location_tuple, distance):
        return 0