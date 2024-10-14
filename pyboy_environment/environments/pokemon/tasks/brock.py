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
        self.currrent_state = None
        self.prev_state = None
        self.found_map = False
        self.seen_pokemon_episode = 0
        self.seen_pokemon = 0
        self.in_battle = False
        self.in_fight = False

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
        
        return state_array

    def _calculate_reward(self, new_state: dict) -> float:
        # ========== LOCATION LOGIC ==========  
        # Get the current location and turn into a tuple
        map_loc = self.current_location["map_id"]
        location_tuple = (self.current_location["x"], self.current_location["y"], self.current_location["map"], self.current_location["map_id"])
        frame = PokemonEnvironment.grab_frame(self)
        game_area = PokemonEnvironment.game_area(self)

        reward = 0.0  # Initialize reward as 0
        distance = 0
        
        # Calculate distance traveled if start_location is set for each map 
        if self.start_location[map_loc]:
            start_x, start_y = self.start_location[map_loc]
            current_x, current_y = self.current_location["x"], self.current_location["y"]
            distance = np.sqrt((current_x - start_x) ** 2 + (current_y - start_y) ** 2)

        # calculate location and map rewards
        reward += self.check_location_rewards(map_loc, location_tuple, distance)

        # Penalize swapping between the same two maps
        reward += self.check_map_swap(map_loc)

        # Calculate Pokemon related rewards
        reward += self.check_pokemon_rewards(frame)

        if self.in_battle:
            reward += self.battle_rewards(game_area)

        if self.in_fight:
            print(f"start: {self.start_location[map_loc]}")
            print(f"current: {self.current_location}")
            print(distance)
            print(f"max dist {map_loc}: {self.max_dist_episode[map_loc]}")
            print(f"{game_area}")
            print(frame)
            input("pause")
            

        # ========== XP Rewards ==========
         # ========== XP REWARD LOGIC ==========


        # ========== UPDATE LOGIC ==========
        # Update the previous distance and location
        self.prev_distance = distance
        self.previous_location = self.current_location
        self.prev_state = self.current_state
        
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
        elif self.current_location["map"] not in self.discovered_maps_episode or self.found_map:
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
    
    def check_pokemon_rewards(self, frame):
        reward = 0

        # Ensure prev_state is available for comparison
        if self.prev_state:
            if self._is_grass_tile():
                if np.all(frame == 0):
                    print("========= starting pokemon battle =========")
                    self.in_battle = True
            # Found new Pokémon in the current state- counts unique pokemon
            if self.current_state["seen_pokemon"] > self.prev_state["seen_pokemon"]:
                print(f"currrent state seen: {self.current_state['seen_pokemon']}")
                print(f"previous state seen: {self.prev_state['seen_pokemon']}")
                print(f"episode seen: {self.seen_pokemon_episode}")
                print(f"max seen: {self.seen_pokemon}")

                reward += 1000.0
                print("============== Found new Pokémon ==============")
                
                # Update the number of Pokémon seen in the current episode
                self.seen_pokemon_episode = self.current_state["seen_pokemon"]
                
                # If the number of Pokémon seen in this episode exceeds the global record
                if self.seen_pokemon_episode > self.seen_pokemon:
                    reward += 1000.0
                    self.seen_pokemon = self.seen_pokemon_episode
                    print("============== Max new Pokémon, giving reward ==============")
        
        # Ensure that `seen_pokemon_episode` is always updated correctly for future comparisons
        self.seen_pokemon_episode = max(self.seen_pokemon_episode, self.current_state["seen_pokemon"])

        return reward

    def battle_rewards(self, game_area):
        reward = 0
        fight_menu = np.array([
    [383, 311, 318, 325, 332, 339, 346, 353, 383, 367, 374, 374, 374, 374, 374, 374, 374, 374, 375, 383],
    [377, 378, 378, 378, 378, 378, 378, 378, 377, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 379],
    [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
    [380, 383, 383, 383, 383, 383, 383, 383, 380, 237, 133, 136, 134, 135, 147, 383, 225, 226, 383, 380],
    [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
    [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 136, 147, 132, 140, 383, 383, 145, 148, 141, 380],
    [381, 378, 378, 378, 378, 378, 378, 378, 381, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 382]
])
        fight_state = np.array([
    [380, 383, 383, 383, 383, 249, 251, 243, 249, 251, 380, 374, 374, 374, 374, 374, 374, 374, 375, 383],
    [381, 378, 378, 378, 378, 378, 378, 378, 378, 378, 382, 378, 378, 378, 378, 378, 378, 378, 378, 379],
    [380, 383, 383, 383, 380, 237, 147, 128, 130, 138, 139, 132, 383, 383, 383, 383, 383, 383, 383, 380],
    [380, 383, 383, 383, 380, 383, 147, 128, 136, 139, 383, 150, 135, 136, 143, 383, 383, 383, 383, 380],
    [380, 383, 383, 383, 380, 383, 227, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
    [380, 383, 383, 383, 380, 383, 227, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
    [381, 378, 378, 378, 381, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 382]
])
        flee_state = np.array([
    [383, 311, 318, 325, 332, 339, 346, 353, 383, 367, 374, 374, 374, 374, 374, 374, 374, 374, 375, 383],
    [377, 378, 378, 378, 378, 378, 378, 378, 377, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 379],
    [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
    [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 133, 136, 134, 135, 147, 383, 225, 226, 383, 380],
    [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
    [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 136, 147, 132, 140, 383, 237, 145, 148, 141, 380],
    [381, 378, 378, 378, 378, 378, 378, 378, 381, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 382]
])      
        tackle_state = np.array([
    [383, 143, 136, 131, 134, 132, 152, 383, 383, 383, 383, 383, 256, 263, 270, 277, 284, 291, 298, 383],
    [383, 383, 383, 383, 366, 249, 383, 383, 383, 383, 383, 383, 257, 264, 271, 278, 285, 292, 299, 383],
    [383, 371, 369, 354, 363, 363, 363, 363, 363, 363, 364, 383, 258, 265, 272, 279, 286, 293, 300, 383],
    [383, 372, 374, 374, 374, 374, 374, 374, 374, 374, 376, 383, 259, 266, 273, 280, 287, 294, 301, 383],
    [383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 260, 267, 274, 281, 288, 295, 302, 383],
    [383, 305, 312, 319, 326, 333, 340, 347, 383, 383, 383, 383, 261, 268, 275, 282, 289, 296, 303, 383],
    [383, 306, 313, 320, 327, 334, 341, 348, 383, 383, 383, 383, 262, 269, 276, 283, 290, 297, 304, 383],
    [383, 307, 314, 321, 328, 335, 342, 349, 383, 383, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
    [383, 308, 315, 322, 329, 336, 343, 350, 383, 383, 383, 383, 383, 383, 366, 252, 383, 383, 383, 383],
    [383, 309, 316, 323, 330, 337, 344, 351, 383, 383, 369, 354, 363, 363, 363, 363, 363, 363, 365, 383],
    [383, 310, 317, 324, 331, 338, 345, 352, 383, 383, 383, 383, 248, 248, 243, 383, 248, 248, 371, 383],
    [383, 311, 318, 325, 332, 339, 346, 353, 383, 367, 374, 374, 374, 374, 374, 374, 374, 374, 375, 383],
    [377, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 379],
    [380, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
    [380, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 383, 383, 383, 383, 383, 383, 383, 383, 380],
    [380, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
    [380, 180, 178, 164, 163, 383, 147, 128, 130, 138, 139, 132, 231, 383, 383, 383, 383, 383, 383, 380],
    [381, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 382]
])
        if np.array_equal(game_area[-7:, :], fight_menu):
            print("on fight menu")
            # self.in_fight = True
            reward += 100.0
            
        elif np.array_equal(game_area[-7:, :], fight_state):
            print("on attack menu")
            reward += 100.0

        elif np.array_equal(game_area[-7:, :], tackle_state):
            print("tackle action")
            reward += 200.0
            
        elif np.array_equal(game_area[-7:, :], flee_state):
            print("on flee menu")
            reward -= 100.0
        return reward

    def xp_rewards(self):
        reward = 0
        if self.prev_state:
            prev_xp = self.prev_state['xp']
            current_xp = self.current_state['xp']

            # Reward for XP increase
            xp_diff = current_xp - prev_xp
            if xp_diff > 0:
                print("========== xp gained! ==========")
                reward += xp_diff * 100  # Linear reward proportional to the XP increase
            
        return reward;
