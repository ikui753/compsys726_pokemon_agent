from functools import cached_property
import numpy as np
import torch
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
        self.prev_loc = None  # Track the previous location
        self.max_dist_episode = np.zeros(248)  # Record max distance per map/ room
        self.max_dist = np.zeros(248)          # Record max distance per map/ room
        self.prev_distance = 0
        self.cumulative_distance = 0  # Track cumulative distance for the episode
        self.loc = None # record current location
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
        game_area = np.array(self.game_area())
        game_area = self.process_game_area(game_area) # process game area and shape correctly for model
        self.loc = game_stats["location"]
        map_loc = self.loc["map_id"]
        self.current_state = game_stats 

        # If the start_location is None (first discovery of a new map), set it
        if self.start_location[map_loc] is None:
            self.start_location[map_loc] = (self.loc["x"], self.loc["y"])

        return game_area

    def _calculate_reward(self, new_state: dict) -> float:
        # initialization logic
        frame = PokemonEnvironment.grab_frame(self)
        game_area = PokemonEnvironment.game_area(self)

        reward = 0.0  # Initialize reward as 0
        distance = 0
        # ========== DISTANCE LOGIC ==========  
        # Calculate distance traveled if start_location is set for each map 
        if self.start_location[self.loc["map_id"]]:
            start_x, start_y = self.start_location[self.loc["map_id"]]
            current_x, current_y = self.loc["x"], self.loc["y"]
            # distance = np.sqrt((current_x - start_x) ** 2 + (current_y - start_y) ** 2)
            distance = abs(current_x-start_x) + abs(current_y-start_y)

        # distance += self.cumulative_distance # Add cumulative distance
        if self.prev_distance > 0 and distance != self.prev_distance and self.in_battle == True: 
            self.in_battle = False
            print("exiting battle mode")
        # calculate location and map rewards
        if self.in_battle: 
            reward += self.battle_rewards(game_area) # Calculate battle rewards if in battle
        else:
            reward += self.check_distance_rewards(distance)
            # reward += self.check_loc_rewards(distance, self.loc, self.prev_loc)
            reward += self.check_map_rewards(self.loc["map_id"])
            reward += self.xp_rewards()
            reward += self.check_pokemon_rewards(frame) # Calculate Pokemon related rewards

        # ========== UPDATE LOGIC ========== 
        # Update the previous distance and location
        self.prev_distance = distance
        self.prev_loc = self.loc
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
        self.discovered_locations.clear()
        self.discovered_maps.clear()
        self.seen_pokemon_episode = 0
        self.max_dist = np.zeros(248) # reset all max distances this episode 
        self.in_battle = False

    def check_distance_rewards(self, distance):
        reward = 0
        if self.prev_distance > 0:
            if distance > self.max_dist[self.loc["map_id"]]:
                reward += distance
                self.max_dist[self.loc["map_id"]] = distance
            if distance > self.prev_distance:
                reward += distance - self.prev_distance  # Reward for the increased distance
        return reward
    
    def check_loc_rewards(self, distance, loc, prev_loc):
        reward = 0
        loc_tuple = (loc["map_id"], loc["x"], loc["y"])
        if prev_loc:
            prev_loc_tuple = (prev_loc["map_id"], prev_loc["x"], prev_loc["y"])
            if loc_tuple not in self.discovered_locations:
                self.discovered_locations.add(loc_tuple)
                reward += distance # quality of new loc determined by distance
                if distance > self.max_dist[self.loc["map_id"]]:
                    reward += distance # double reward if new loc is beyond max dist
            elif loc_tuple == prev_loc_tuple:
                reward -= 1.0 # if stuck at same location
        return reward
    
    def check_map_rewards(self, current_map):
        reward = 0
        if self.prev_loc:
            if self.loc["map_id"] != self.prev_loc["map_id"]:
                print("Changed map.")
        if current_map not in self.discovered_maps and current_map not in {37, 38, 39, 40, 41}:
            print(f"Discovered {pkc.get_map_location(current_map)}!")
            reward += 1000.0 * (len(self.discovered_maps) + 1)
            self.discovered_maps.add(current_map)
        return reward
    
    def check_pokemon_rewards(self, frame):
        reward = 0
        # Ensure prev_state is available for comparison
        if self.prev_state:
            if self._is_grass_tile():
                if np.all(frame == 0):
                    print("========= starting pokemon battle =========")
                    reward += 1000.0 # reward for starting battle
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
        attack_menu = np.array([
    [380, 383, 383, 383, 383, 249, 251, 243, 249, 251, 380, 374, 374, 374, 374, 374, 374, 374, 375, 383],
    [381, 378, 378, 378, 378, 378, 378, 378, 378, 378, 382, 378, 378, 378, 378, 378, 378, 378, 378, 379],
    [380, 383, 383, 383, 380, 237, 147, 128, 130, 138, 139, 132, 383, 383, 383, 383, 383, 383, 383, 380],
    [380, 383, 383, 383, 380, 383, 147, 128, 136, 139, 383, 150, 135, 136, 143, 383, 383, 383, 383, 380],
    [380, 383, 383, 383, 380, 383, 227, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
    [380, 383, 383, 383, 380, 383, 227, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
    [381, 378, 378, 378, 381, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 382]
])
        flee_menu = np.array([
    [383, 311, 318, 325, 332, 339, 346, 353, 383, 367, 374, 374, 374, 374, 374, 374, 374, 374, 375, 383],
    [377, 378, 378, 378, 378, 378, 378, 378, 377, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 379],
    [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
    [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 133, 136, 134, 135, 147, 383, 225, 226, 383, 380],
    [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
    [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 136, 147, 132, 140, 383, 237, 145, 148, 141, 380],
    [381, 378, 378, 378, 378, 378, 378, 378, 381, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 382]
])      
        tackle_state = np.array([
    [383, 311, 318, 325, 332, 339, 346, 353, 383, 367, 374, 374, 374, 374, 374, 374, 374, 374, 375, 383],
    [377, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 379],
    [380, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
    [380, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 383, 383, 383, 383, 383, 383, 383, 383, 380],
    [380, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
    [380, 180, 178, 164, 163, 383, 147, 128, 130, 138, 139, 132, 231, 383, 383, 383, 383, 383, 383, 380],
    [381, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 382]
])
        
        if np.array_equal(game_area[-7:, :], fight_menu):
            # input("pause")
            print("on fight menu")
            reward += 100.0
            
        elif np.array_equal(game_area[-7:, :], attack_menu):
            print("choosing attack")
            # input("pause")
            reward += 200.0

        elif np.array_equal(game_area[-7:, :], tackle_state):
            print("performing tackle action")
            reward += 300.0
            
        elif np.array_equal(game_area[-7:, :], flee_menu):
            print("on flee menu")
            reward -= 100.0
        return reward

    def xp_rewards(self):
        reward = 0
        if self.prev_state:
            prev_xp = self.prev_state['xp']
            current_xp = self.current_state['xp']

            if current_xp > prev_xp:
                print("========== Gained XP! ==========")
                reward += 10000
            
        return reward;
    
    def process_game_area(self, game_area):
        while game_area.size > 256:
            game_area = game_area[1:-1, 1:-1]  # Remove first and last row, first and last column

        flattened_area = game_area.flatten()  # Flatten the existing game_area

        # If the flattened area has less than 256 elements, pad with zeros
        if flattened_area.size < 256:
            padding = np.zeros(256 - flattened_area.size, dtype=flattened_area.dtype)
            # Concatenate the flattened area with padding
            combined = np.concatenate((flattened_area, padding))
        else:
            combined = flattened_area

        combined = np.array(combined, dtype=np.float32)  # Correctly create the array
        return combined # return numpy array to be converted to tensor later