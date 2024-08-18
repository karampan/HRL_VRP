#--------------CONTENTS-----------------------

# 1. Configuration
    # 1.1 Imports
    # 1.2 Experiment configuration
    # 1.3 Constants
    # 1.4 Actions
    # 1.5 Objects
    # 1.6 Grid
# 2. Environment
    # 2.1 Minigrid Environment
    # 2.2 Manual Control
# 3. Main 
    # 3.1 Manual run mode
    # 3.2 Dynamic Programming run mode
    # 3.3 Reinforcement Learning run mode
    # 3.4 Hierarchical Reinforcement Learning run mode


#1--------------CONFIGURATION-----------------------------------------------------------------------------------------------------------------------------------------------------------

#1.1--------------IMPORTS------------------------

from __future__ import annotations
from minigrid.core.constants import (
    COLOR_NAMES,
    COLOR_TO_IDX,
    COLORS,
    OBJECT_TO_IDX,
    TILE_PIXELS)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from enum import IntEnum
from abc import abstractmethod
from typing import Any, Iterable, SupportsFloat, TypeVar, Callable
import gymnasium as gym
import pygame
import pygame.freetype
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from gymnasium import Env
from prompt_toolkit import prompt
import pandas as pd
from itertools import product
from collections import defaultdict, deque
import time
import sys
import random
from abc import abstractmethod
from typing import TYPE_CHECKING, Tuple
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    downsample,
    fill_coords,
    point_in_rect,
    point_in_circle)

#1.2--------------EXPERIMENT PARAMETERS-----------------------

nr_of_customer_statuses = 3
global_grid_size = 11
max_steps = 50

cust_no = int(prompt('Number of customers (select between 1 and 10):'))
customers_coord = ([1,2],[6,4], [1,3], [2,3], [9,8], [7,6], [3,7], [5,6], [3,5] , [4,7])#, [7,8], [1,6], [7,1]) 
if cust_no == 0:
    cust_no = 1
elif cust_no > 10:
    cust_no = 10

customers_coord = customers_coord[:cust_no]

epsilon = 0.1
alpha = 0.01
gamma = 0.99

starting_statuses_prob = 0.3

new_request_prob = float(prompt('Probability for new service requests (select between 0 and 0.02):'))
if new_request_prob > 1:
    new_request_prob = 1
elif new_request_prob < 0:
    new_request_prob = 0


render_on_global = False  
evaluation = False
evaluation_switch = True

plot_every = 1000
num_episodes = 100000
num_episodes = int(prompt('Number of episodes (select between 0 and 3000000):'))

# a prompt message asking what run mode to be used
run_mode = prompt('Select run mode (M for Manual, D for DP, R for RL, H for HRL):')

print("------------------Scenario specs------------------------")
print("Number of customers: ", cust_no, customers_coord)
print("Probability for new service requests: ", new_request_prob)
print("Number of episodes: ", num_episodes)
print("Run mode: ", run_mode)
print("--------------------------------------------------------")

#1.3--------------CONSTANTS---------------------------

Point = Tuple[int, int]
T = TypeVar("T")
TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
}

COLOR_NAMES = sorted(list(COLORS.keys()))


#1.4-------------------ACTIONS------------------

class Actions_(IntEnum):
    left = 0
    right = 1
    up = 2
    down = 3
    

#1.5------------------OBJECTS---------------------
class WorldObj:

    """
    Base class for grid world objects
    """

    def __init__(self, type: str, color: str):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos: Point | None = None

        # Current position of the object
        self.cur_pos: Point | None = None



    def encode(self) -> tuple[int, int, int]:
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)


class Goal(WorldObj):
    def __init__(self):
        super().__init__("goal", "green")

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class Ball(WorldObj):
    def __init__(self, color="blue"):
        super().__init__("ball", color)

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class Wall(WorldObj):
    def __init__(self, color: str = "grey"):
        super().__init__("wall", color)

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


#1.6--------------------GRID------------------------------

class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache: dict[tuple[Any, ...], Any] = {}

    def __init__(self, width: int, height: int):
        assert width >= 3
        assert height >= 3
        self.width: int = width
        self.height: int = height

        self.grid: list[WorldObj | None] = [None] * (width * height)


    def set(self, i: int, j: int, v: WorldObj | None):
        assert (
            0 <= i < self.width
        ), f"column index {i} outside of grid of width {self.width}"
        assert (
            0 <= j < self.height
        ), f"row index {j} outside of grid of height {self.height}"
        self.grid[j * self.width + i] = v

    def get(self, i: int, j: int) -> WorldObj | None:

        assert 0 <= i < self.width
        assert 0 <= j < self.height
        assert self.grid is not None
        return self.grid[j * self.width + i]

    def horz_wall(
        self,
        x: int,
        y: int,
        length: int | None = None,
        obj_type: Callable[[], WorldObj] = Wall,
    ):

        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(
        self,
        x: int,
        y: int,
        length: int | None = None,
        obj_type: Callable[[], WorldObj] = Wall,
    ):

        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x: int, y: int, w: int, h: int):

        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)


    @classmethod
    def render_tile(
        cls,
        obj: WorldObj | None,
        agent_dir: int | None = None,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 3,
    ) -> np.ndarray:
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_circle(0.5, 0.5, 0.15)
            
            # Rotate the agent based on its direction
            fill_coords(img, tri_fn, (255, 0, 0))


        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
        self,
        tile_size: int,
        agent_pos: tuple[int, int],
        agent_dir: int | None = None,
        highlight_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                assert highlight_mask is not None
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img
        

        return img


#2--------------------ENVIRONMENT--------------------------------------------------------------------------------------------------------------------------------------------------------

#2.1--------------------MINIGRID ENVIRONMENT--------------

class MiniGridEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }


    def __init__(
        self,
        grid_size: int = global_grid_size,
        width: int | None = None,
        height: int | None = None,
        max_steps: int = max_steps,
        render_mode: str | None = None,
        screen_size: int | None = 640,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):


        """
        The main variables of the environment are initiated. 
         - Width and height of the grid are set equal to the parameter grid_size which can vary.
         - Starting position of the agent is set to the middle of the grid
         - The class Grid is initialized
         - The rendering parameters are set
         - The coordinates of customers are read and transformed into numeric positions in the grid
         - The action space is determined
         - The space of all customers statuses is determined

        For dynamic programming solution
         - The transition probability matrix is calculated by iterating through the state-action pairs using the step function
  
        """

        #Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size
        # assert width is not None and height is not None

        self.screen_size = screen_size
        self.render_size = None
        self.window = None
        self.clock = None

        # Environment configuration
        self.width = grid_size
        self.height = grid_size
        self.max_steps = max_steps

        # Initialize step
        self.step_count = 0

        # Determine starting position and Initialize current position 
        self.agent_pos: np.ndarray | tuple[int, int] = None
        self.agent_start_pos = (int(float(global_grid_size) // 2), int(float(global_grid_size) // 2))
        self.agent_pos = self.agent_start_pos 
        self.agent_start_pos_nr = self._position(global_grid_size, self.agent_start_pos[0], self.agent_start_pos[1])

        # Current grid
        self.grid = Grid(width, height)

        # Rendering attributes
        self.render_mode = render_mode
        self.highlight = highlight
        self.tile_size = tile_size
        self.agent_pov = agent_pov

        # Action enumeration for this environment
        self.actions = Actions_

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Load customer coordinates to a data frame
        self.customers_coord = customers_coord 
        self.nr_of_customers = len(self.customers_coord)
        self.customers_coord = pd.DataFrame({"Coordinates": self.customers_coord})

        # Create a list with customer numberred positions
        self.customers_pos = list()
        for c in range(self.nr_of_customers):
            self.customers_pos.append(self._position(global_grid_size, self.customers_coord.iloc[c][0][0], self.customers_coord.iloc[c][0][1]))


        # Find space of all customers and statuses conbinations
        self.c_state = list(product(range(nr_of_customer_statuses), repeat = self.nr_of_customers))

        #Remove the first element of the list consisting of 0s (0,0,0,0,0,..)
        self.c_state.pop(0)

        # When the problem is solved with dynamic programming, the probability matrix is calculated
        if run_mode == 'D':
            self.reset(1)
            #determine observation space
            self.observation_space = spaces.Discrete(global_grid_size*global_grid_size*len(self.c_state))

            #prepare probability transition matrix
            self.P = defaultdict(lambda: defaultdict())
            position = global_grid_size**2 

            for p in range(1,position):
                
                check_p = self.rev_position(p)

                # when the vehicle moves out of the grip boundaries, the transition is skipped
                if check_p[0]< 1 or check_p[1] < 1 or check_p[0] >  global_grid_size -2 or check_p[1] > global_grid_size -2:

                    continue

                for a in range(len(self.actions)):

                    for cs in self.c_state:
                            
                        self.statuses_list = [*cs]

                        self.P[p, *cs][a] = self.step( a, (p, *cs)) 
                            

    # Reset of the enviroment to the initial state: 
    #   - a new grid is generated (customer coordinates are read, visual grid is generated, vehicle is placed at the depot, initial customer statuses are determined based on binomial probability simulation) 
    #   - step count is set to 0
    def reset(
        self, episode,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None, 
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        # Step count since episode start
        self.step_count = 0
        self.time_out = False
        self.render_on = False

        if episode % plot_every == 0 or run_mode == "M" or evaluation == True:
            self.render_on = True

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        if (self.render_on == True and render_on_global == True) or evaluation == True:
            self.render()
        

    #calculates Manhattan distance 
    def manhattan(self, a, b ):
        return sum(abs(val1-val2) for val1, val2 in zip(a,b))

    #determines the numberred position in the grid based on the coordinates (row and col) 
    def _position(self, ncols, row, col):
        return row * global_grid_size + col + 1

    #determines the coordinates (row and col) based on the numberred position in the grid 
    def rev_position(self, position):
        
        row = position // global_grid_size

        col = position % global_grid_size - 1

        return tuple((row , col))

    #determines new position coordinates based on current position and action
    def move_to_coord(self, row, col, action):
        
        if row < 1 or col < 1 or row > global_grid_size -2 or col > global_grid_size -2:
            pass
        else:
            # Move left
            if action == self.actions.left:
                row = row - 1
            # Move right
            elif action == self.actions.right:
                row = row + 1
            # Move up
            elif action == self.actions.up:
                col = col - 1
            # Move down
            elif action == self.actions.down:
                col = col + 1   

            else:
                raise ValueError(f"Unknown action: {action}")
        return tuple((row, col))


    # new grid is generated
    # - customer coordinates are read
    # - vehicle is placed at the depot
    # - initial customer statuses are determined based on binomial probability simulation
    # - visual grid is generated
    def _gen_grid(self, width, height):
        
        # read customer coordinates
        self.customers_coord = customers_coord

        # Place the agent
        self.agent_pos = self.agent_start_pos 
        self.agent_dir = 0 
        
        # Place the customers. Ensure there is at least one active customer in the first epoch
        active_cust = 0

        # the simulation is run until at least one customer gets an active service request
        while active_cust == 0:
            self.cust_status = binom.rvs(n = 1, p = starting_statuses_prob, size = self.nr_of_customers)
            active_cust = sum(self.cust_status)

        # generate customer table with statuses
        self.cust_table = pd.DataFrame({"Coordinates": self.customers_coord, "Status": self.cust_status})
        self.statuses_list = self.cust_table['Status'].to_list()
       
        # create visual grid
        if (self.render_on == True and render_on_global == True) or run_mode == "M" or evaluation == True:
            
            # Create an empty grid
            self.grid = Grid(width, height)
            
            # Generate the surrounding walls
            self.grid.wall_rect(0, 0, width, height)

            # Place depot
            self.put_obj(Goal(), int(float(global_grid_size) // 2), int(float(global_grid_size) // 2))

            # Plce objects in the grid
            for i in range(len(self.statuses_list)):

                if  self.statuses_list[i] == 1:
                    self.put_obj(Goal(), self.cust_table.iloc[i,0][0], self.cust_table.iloc[i,0][1])
                else:
                    self.grid.set(self.cust_table.iloc[i,0][0], self.cust_table.iloc[i,0][1],  Ball(COLOR_NAMES[2]) )

            # set step count to 0
            self.count_index = 0

    # Function that puts an object at a specific position in the grid
    def put_obj(self, obj: WorldObj, i: int, j: int):
       
        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)


    # Step function uses the current state and action and inputs and determines the new state
    # As part of the step function, the binomial probability simulation runs for potential service requests at every step
    # The step function operations can differ per run mode (dynamic programming, reinforcement learning, hierarchical reinforcement learning)
    def step(
        self, action: ActType, state, time_out = False
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1
    
        reward = -1
        score = -1
        terminated = False
        truncated = False
        n_state = None

        # store current position and its number and determine new position and its number
        current_position = state[0]
        current_position_coord = self.rev_position(current_position)
        new_position_coord = self.move_to_coord(current_position_coord[0], current_position_coord[1], action)
        new_position = self._position(global_grid_size, new_position_coord[0], new_position_coord[1])

        # update the global variable of the position to the new position
        self.agent_pos = new_position_coord

        # if new position is out of the grid, then penalty is received and new position is switched to previous position
        if new_position_coord[0] < 1 or new_position_coord[1] < 1 or new_position_coord[0] >  global_grid_size -2 or new_position_coord[1] > global_grid_size -2:

            reward -= 1
            new_position = current_position
            new_position_coord = current_position_coord
            self.agent_pos = new_position_coord
        
        # For run mode Hierarchical reinforcement learning
        if run_mode == "H"  :
            
            # if time has not run out
            if time_out == False:

                # run simulation for potential new service requests
                self.statuses_list = [1 if self.statuses_list[i] == 0 and binom.rvs(n=1, p=new_request_prob, size=1) == 1 else self.statuses_list[i] for i in range(self.nr_of_customers)]

            # target position is set as the sub-goal of the upper policy which is the second element of the state variable
            target_position_coord = self.rev_position(state[1]) 
            
            # update the visual grid
            if (self.render_on == True and render_on_global == True) or evaluation == True:
                
                for i in range(self.nr_of_customers):    
                
                    if self.statuses_list[i] == 1: 
                        self.put_obj(Goal(), self.cust_table.iloc[i,0][0], self.cust_table.iloc[i,0][1])

                self.grid.set( target_position_coord[0], target_position_coord[1],     Ball(COLOR_NAMES[0]) )

            # if new position equals the sub-goal position
            if state[1] == new_position:
                reward += 20
                score += 20
                terminated = True
                
                # if the new position is not the depot
                if new_position != self.agent_start_pos_nr: 

                    # find the index of the new position in the customer list
                    g = env.customers_pos.index(new_position) 

                    # update the status of the satisfied customer 
                    self.statuses_list[g] = 2

                    # update the visual grid
                    if (self.render_on == True and render_on_global == True) or evaluation == True:
                        self.grid.set( new_position_coord[0], new_position_coord[1],     Ball(COLOR_NAMES[5]) )

            else:
                
                # if new position does not equal the sub-goal, _
                # a penalty equal to negative manhattan distance between the new position and sub-goal position divided by the grid size squared is collected
                reward -= self.manhattan(new_position_coord, target_position_coord) / (global_grid_size * 2)

            # the new state is determined as the new position and the sub-goal (subgoal does not change)
            n_state = (new_position, state[1])
            
            # visual rendering if applicable
            if (self.render_on == True and render_on_global == True) or evaluation == True:
                self.render()
                time.gmtime(0)
        
        # For run mode Dynamic programming
        elif run_mode == "D":
            
            # check if new position matches any of the customers position
            try:
                # if yes, the status of the relevant customer is updated and reward is collected
                g = self.customers_pos.index(new_position)

                if self.statuses_list[g] == 1:

                    reward += 20
                    score += 20
                    self.statuses_list[g] = 2
             
            except ValueError:
                None

            # new state consists of new position and the list of customer current statuses
            n_state = (new_position, *self.statuses_list) 
            
            #if vehicle is on depot before time is up
            if self.agent_pos == self.agent_start_pos: 
                
                terminated = True
                reward += np.sum(np.array(self.statuses_list) == 1)*(-20) # penalty equal to 20 is collected for each unserved customer
                score += 20
    
        # For run modes Reinforcement learning and manual
        elif run_mode == "R" or run_mode == "M":
            
            # time out flag set when max steps are exceeded
            if self.step_count == max_steps:
                self.time_out = True

            # simulation for new service requests stops after the max steps are exceeded
            if self.time_out == False:
                self.statuses_list = [1 if self.statuses_list[i] == 0 and binom.rvs(n=1, p=new_request_prob, size=1) == 1 else self.statuses_list[i] for i in range(self.nr_of_customers)]

            # update visual grid
            if (self.render_on == True and render_on_global == True) or run_mode == "M":

                for i in range(self.nr_of_customers):    
                    
                    if self.statuses_list[i] == 1: 
                        self.put_obj(Goal(), self.cust_table.iloc[i,0][0], self.cust_table.iloc[i,0][1])

            # check if new position matches any of the customers position
            try:
                g = self.customers_pos.index(new_position)

                # if new position matches a customer with an active service request, the status of the customer is updated and reward is collected
                if self.statuses_list[g] == 1 and self.time_out == False:

                    reward += 20
                    score += 20
                    self.statuses_list[g] = 2

                    # update visual grid
                    if (self.render_on == True and render_on_global == True) or run_mode == "M":
                        self.grid.set( new_position_coord[0], new_position_coord[1],     Ball(COLOR_NAMES[5]) )
             
            except ValueError:
                None

            # update visual grid
            if (self.render_on == True and render_on_global == True) or evaluation == True:
                self.render()
                time.gmtime(0)

            # calculate min required time as the manhattan distance between new position and depot position
            min_required_time = self.manhattan(new_position_coord,self.agent_start_pos )
            
            # if min required time exceeds remaining available steps or there is no active customer service request
            if min_required_time >= max_steps - self.step_count or np.sum(np.array(self.statuses_list) == 1) == 0:  
            
                back_to_depot = True

            else:
                back_to_depot = False
            
            # new state consists of new position, return to depot flag and list of customer statuses
            n_state = (new_position, back_to_depot, *self.statuses_list) 

            # if new position matches depot position and return to depot flag is set, reward is collected and episode is terminated
            if self.agent_pos == self.agent_start_pos and back_to_depot == True: 
                terminated = True
                reward += 20
                score += 20

            # if position does not match the depot position and return to depot is set, then a penalty is collected
            elif state[1] == True:

                reward += (self.manhattan(current_position_coord, self.agent_start_pos)- self.manhattan(new_position_coord, self.agent_start_pos))/ (global_grid_size * 2)

        return n_state, reward, terminated, truncated, score

    # supports the visualization of the grid in different modes
    def render(self):
        img = self.grid.render(
            self.tile_size,
            self.agent_pos,
            self.agent_dir)

        self.count_index += 1


        img = np.transpose(img, axes=(1, 0, 2))
        if self.render_size is None:
            self.render_size = img.shape[:2]
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.screen_size, self.screen_size)
            )
            pygame.display.set_caption("minigrid")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        surf = pygame.surfarray.make_surface(img)

        # Create background with mission description
        offset = surf.get_size()[0] * 0.1
        # offset = 32 if self.agent_pov else 64
        bg = pygame.Surface(
            (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
        )
        bg.convert()
        bg.fill((255, 255, 255))
        bg.blit(surf, (offset / 2, 0))

        bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

        font_size = 22
        text = "Render mode: " + run_mode + " | Step count: " + str(self.count_index) + " | Time remaining: " + str(max_steps - self.count_index)
        font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
        text_rect = font.get_rect(text, size=font_size)
        text_rect.center = bg.get_rect().center
        text_rect.y = bg.get_height() - font_size * 1.5
        font.render_to(bg, text_rect, text, size=font_size)

        self.window.blit(bg, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()


    def close(self):

        if self.window:
            pygame.quit()

#2.2----------------MANUAL CONTROL----------------------------------

# enables the use of keyboard keys for manual navigation in the grid when run mode Manual is selected
class ManualControl:
    def __init__(
        self,
        env: Env,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False


    def start(self):
        """Start the window display with blocking event loop"""

        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)


    def step(self, action: Actions_):

        #print(self.env.statuses_list)

        state = (self.env._position(global_grid_size, self.env.agent_pos[0], self.env.agent_pos[1]), False, *list(self.env.cust_status))

        state, reward, terminated, truncated, score = self.env.step(action, state)

        #print(f"step={self.env.step_count}, reward={reward:.2f}")
        
        if terminated:
            self.reset(self.seed)
        elif truncated:
            self.reset(self.seed)
        else:
            self.env.render()


    def reset(self, seed=None):
        self.env.reset(1, seed=seed)
        self.env.render()


    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            
            #export a csv with all steps
            structure_list = list()
            dict = {'steps': structure_list}
            df = pd.DataFrame(dict)
            df.to_csv('GFG.csv')

            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions_.left,
            "right": Actions_.right,
            "up": Actions_.up,
            "down": Actions_.down,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)


#3.-----------MAIN-------------------------------------------------------------------------------------------------------------------------------------------------------------

#3.1----------MANUAL------------------------
# in manual run mode, the agent is completely controlled by the user who interacts with the enviorment through the keyboard
if run_mode == "M":
    def main():

        env = MiniGridEnv(render_mode="human") 

        manual_control = ManualControl(env, seed=42)
        manual_control.start()
        
    if __name__ == "__main__":
        main()

#3.2---------DYNAMIC PROGRAMMING--------------
# in DP run mode, the agent learns a near optimal policy by interacting with the enviorment using dynamic programming
elif run_mode == "D":

    timestamps = list()
    
    env = MiniGridEnv(render_mode="rgb_array")


    # Dynamic programming
    def dynamic_program(env, gamma=gamma, eps=1e-6):
        
        start_timestamp = time.time()

        # determine the action space
        env.nA = env.action_space.n

        # determine the state space
        observation_space = spaces.Discrete(global_grid_size*global_grid_size*len(env.c_state))
        env.nS = observation_space.n 

        # initialize the V table
        V = {s: 0 for s in env.P.keys()} 

        # learning loop
        while True:
            # initialize the Q table with -2
            Q     = np.ones(( env.nS , env.nA))*(-2)  
            delta = 0.0

            # for each state
            for j,s in enumerate(env.P.keys()): 
                
                # initialize value for state s
                Vs = 0.0

                # for each action
                for a in range(env.nA):
                    
                    # get next state, reward, termination flag from probability transition table P
                    next_state, reward, terminated, truncated, _ = env.P[s][a]

                    # Q table update
                    Q[j,a] += 1 * (reward + gamma * V[next_state])

                # value as the mean of Q values for state
                Vs    = np.mean(Q[j,])

                # max between previous value and incremental increase in the value of state s
                delta = max(delta, np.abs(V[s]-Vs))
            
                V[s]  = Vs

            # if delta is lower then eps, stop
            if delta < eps:
                break

        timestamps.append(time.time() - start_timestamp)

        return V, Q

    # call dynamic programming
    V, Q = dynamic_program(env)

    # initialize policy
    policy = {s: np.ones([env.nA]) / env.nA for s in env.P.keys()}

    # initialize Q
    Q = np.zeros((env.nS, env.nA))

    # for each state s
    for j,s in enumerate(env.P.keys()):
        # for each action a
        for a in range(env.nA):
            # get next state, reward, termination flag from probability transition table P
            next_state, reward, terminated, truncated, _ = env.P[s][a]

            # determine Q value given gamma discount factor
            Q[j,a] += 1 * (reward + gamma * V[next_state])

        # find optional actions in q table
        a_opt     = np.flatnonzero(Q[j,] == np.max(Q[j,]))

        # initialize policy for state s
        policy[s] = np.zeros([env.nA])

        # assign the probability for optimal actions or actions
        policy[s][a_opt] = 1.0 / len(a_opt)

    
    # policy data frame
    df1 = pd.DataFrame(policy)
    # export to csv
    df1.to_csv('policy.csv')


    #policy evaluation
    num_episodes = 10000   
    output_quality = []

    for episode in range(1,num_episodes+1): 
            
        # env reset
        env.reset(1)
        
        # initial state
        state = (env.agent_start_pos_nr , *env.statuses_list)

        score = 0  
        total_score = 0


        while True:
            
            # select action based on policy
            a = np.random.choice(env.actions, p = policy[state])

            # get step output based on action selected
            next_state, reward, terminated, truncated, score = env.step(a,state)

            total_score += score
            state = next_state

            if terminated:
                break
        
        # append total score to output quality 
        output_quality.append(total_score)

    print(np.mean(output_quality))

    dict4 = {'timestamps': timestamps, 'average quality':np.mean(output_quality)}
    df4 = pd.DataFrame(dict4)
    df4.to_csv('D_'+ str(len(customers_coord))+ '_' + str(new_request_prob)+'.csv')

#3.3-----------------REINFORCEMENT LEARNING-------------------------------
# in RL run mode, the agent learns with Q learning a policy by interacting with the environment using epsilon greedy method

elif run_mode == "R":

    # Update Q table with temporal different zero
    def update_Q(alpha, gamma, Q, state, action, reward, next_state=None):
        """Returns updated Q-value for the most recent experience."""
        current = Q[state][action]  # estimate in Q-table (for current state, action pair)
        Qsa_next = np.max(Q[next_state]) if next_state is not None else 0  # value of next state 
        target = reward + (gamma * Qsa_next)               # construct TD target
        new_value = current + (alpha * (target - current)) # get updated value 
        return new_value

    # Epsilon greedy for action selection
    def epsilon_greedy(Q, state, nA, eps):
        """Selects epsilon-greedy action for supplied state.

        Params
        ======
            Q (dictionary): action-value function
            state (int): current state
            nA (int): number actions in the environment
            eps (float): epsilon
        """
        # select greedy action with probability epsilon
        if random.random() > eps: 
            return np.argmax(Q[state])
        # otherwise, select an action randomly
        else:                     
            return random.choice(np.arange(nA)) 

    # Q learning
    def q_learning(env, num_episodes, alpha = alpha, gamma=gamma, plot_every=plot_every):
        """Q-Learning - TD Control

        Params
        ======
            num_episodes (int): number of episodes to run the algorithm
            alpha (float): learning rate
            gamma (float): discount factor
            plot_every (int): number of episodes to use when calculating average score
        """
        # number of actions
        nA = env.action_space.n             

        # initialize empty dictionary of arrays with -7 
        Q = defaultdict(lambda: np.ones(nA)*(-7))  
        start_timestamp =time.time() 

        # monitor performance
        tmp_scores = deque(maxlen=plot_every)     # deque for keeping track of scores
        avg_scores = deque(maxlen=num_episodes)   # average scores over every plot_every episodes
        timestamps = deque(maxlen=num_episodes) 

        for episode in range(1, num_episodes+1):

            # monitor progress
            if episode % 100 == 0:
                print("\rEpisode {}/{}".format(episode, num_episodes), end="")
                sys.stdout.flush()

            score = 0                                              
            back_to_depot = False
            
            # reset env
            env.reset(episode)

            # initilize state
            state = (env.agent_start_pos_nr, back_to_depot, *(list(env.statuses_list))) 
            
            while True:
                
                # select action with epsilon greedy
                action = epsilon_greedy(Q, state, nA, epsilon)        

                # step
                next_state, reward, done,  _ , inc_score =   env.step(action, state) 

                # acculumate score
                score += inc_score                                    

                # update Q values
                Q[state][action] = update_Q(alpha, gamma, Q, state, action, reward, next_state)    

                state = next_state                           

                if done:
                    tmp_scores.append(score)                 
                    break
                    

            if (episode % plot_every == 0):
                avg_scores.append(np.mean(tmp_scores))
                timestamps.append(time.time()-start_timestamp)


        # plot performance
        plt.plot(np.linspace(0,num_episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))
        plt.xlabel('Episode Number')
        plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
        plt.savefig('R_'+ str(len(customers_coord))+ '_' + str(new_request_prob)+'.png')
        plt.show()

        print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))

        dict4 = {'episodes': list(range(plot_every,num_episodes + 1 , plot_every)),'avg_scores': avg_scores,'timestamps': timestamps }
        df4 = pd.DataFrame(dict4)
        df4.to_csv('R_'+ str(len(customers_coord))+ '_' + str(new_request_prob)+'.csv')

        return Q

    # run environment
    env = MiniGridEnv(render_mode="rgb_array")

    # create Q table
    Q__ = q_learning(env, num_episodes)

    # number of actions
    nA = env.action_space.n                

    # initialize policy with zeros
    policy__ = {key: np.zeros(nA) for key in Q__.keys()}

    # set optimal action with probability 1 
    for s in Q__.keys():
        g = np.argmax(Q__[s])
        policy__[s][g] = 1

    # policy evaluation
    evaluation = evaluation_switch
    num_episodes = 10000
    output_quality = []
    success_count = 0

    for episode in range(1,num_episodes+1): 

        # reset env  
        env.reset(1)
        
        # initial state
        state = (env.agent_start_pos_nr ,False, *env.statuses_list)

        score = 0  
        total_score = 0
        global_step_count = 0

        while True:
            # if the action matches a customer position
            try:
                # select the optimal action based on policy
                a = np.random.choice(env.actions, p = policy__[state])

                global_step_count += 1
            
            # if the action matches the depot position
            except KeyError:
                break
            
            # step
            next_state, reward, terminated, truncated, score = env.step(a,state)
            
            # accumulate score
            total_score += score

            state = next_state

            # if termination condition was met
            if terminated:
                # if step count does not exceed max steps
                if global_step_count <= max_steps:
                    # increase the success count
                    success_count += 1

                break
            else:
                if global_step_count > max_steps:
                    break

        output_quality.append(total_score)

    print(np.mean(output_quality))
    print(np.mean(success_count/num_episodes))

#3.4-----------------HIERARCHICAL REINFORCEMENT LEARNING-----------------
# in HRL run mode, the agent learns with Q learning upper and lower policies by interacting with the environment using epsilon greedy method

elif run_mode == "H":

    env = MiniGridEnv(render_mode="rgb_array")

    # Hyperparameters
    
    # alpha
    manager_alpha = alpha
    worker_alpha = alpha

    # Epsilon-greedy parameters
    epsilon_manager = epsilon
    epsilon_worker = epsilon

    # initialize depot position
    depot_position = [env.agent_start_pos_nr]

    # extend list customer depot positions with the depot position 
    customer_depot_pos = env.customers_pos + depot_position

    # ensure sub goals do not contain same value in from and to positions
    spe = set(zip(customer_depot_pos, customer_depot_pos))
    sub_goals = [i for i in product(customer_depot_pos, customer_depot_pos) if i not in spe]
    
    # initialize worker big Q table 
    worker_big_Q = {}

    for table_index in (sub_goals):
        worker_small_Q = defaultdict(lambda: np.zeros(env.action_space.n))
        # Create dictionaries in each q table
        for loc_index in range(1,global_grid_size**2):
            for action_index in range(env.action_space.n):
                
                position_coord = env.rev_position(loc_index)

                new_position_coord = env.move_to_coord(position_coord[0],position_coord[1], action_index  )

                new_position = env._position(global_grid_size, new_position_coord[0], new_position_coord[1])

                if new_position_coord[0] < 1 or new_position_coord[1] < 1 or new_position_coord[0] >  global_grid_size -2 or new_position_coord[1] > global_grid_size -2:
                    worker_small_Q[loc_index][action_index] = -10*global_grid_size 
                    continue
                
                worker_small_Q[loc_index][action_index] = -2*global_grid_size 

        worker_big_Q[table_index] = worker_small_Q

    # All customer status combinations
    c_state = env.c_state

    # Define the manager Q-table 
    manager_q_table = {}
    manager_states = list()
    manager_states = [(index_pos, *index_c_state) for index_pos in customer_depot_pos for index_c_state in c_state]
    manager_q_table = {index_state: {index_action: -2*global_grid_size  for index_action in customer_depot_pos if index_state[0] != index_action} for index_state in manager_states}

    # Update Q table function
    def update_Q_x(alpha, gamma, Q, state, action, reward, next_state=None):
        """Returns updated Q-value for the most recent experience."""

        current = Q[state][action] 
        
        #WORKER
        if isinstance(Q[state], np.ndarray):
            Qsa_next = np.max(Q[next_state[0]]) if next_state is not None else 0  # value of next state 
            
        #MANAGER
        else:
            
            if next_state is not None:
                Qsa_next = np.max(np.array(list(Q[next_state].values())))
            else:
                Qsa_next = 0

        target = reward + (gamma * Qsa_next)               # construct TD target
        new_value = current + (alpha * (target - current)) # get updated value 

        return new_value

    # Epsilon-greedy policy for exploration
    def epsilon_greedy_policy(Q, state, epsilon, action_space):
        if np.random.rand() < epsilon:
            return np.random.choice(np.array(action_space))
            
        else:
            # WORKER
            if isinstance(Q[state], np.ndarray):
                item_position = np.argmax(Q[state])
                value_to_return = item_position
            # MANAGER
            else:
                item_position = np.argmax(list(Q[state].values()))
                value_to_return = list(Q[state].keys())[item_position]
            return value_to_return 


    # monitor performance
    tmp_scores = deque(maxlen=plot_every)     # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episodes)   # average scores over every plot_every episodes
    timestamps = deque(maxlen=num_episodes) 

    # Q learning loops
    def q_learning(env, num_episodes, gamma=gamma, plot_every=plot_every):

        start_timestamp = time.time()
        
        for episode in range(1,num_episodes+1): 
            
            if episode % 100 == 0:
                print("\rEpisode {}/{}".format(episode, num_episodes), end="")
                sys.stdout.flush()

            
            score = 0    
            env.reset(episode)
            manager_state = (env.agent_start_pos_nr  , *(list(env.statuses_list))) 
          
            done_m = False
            global_step_count = 0

            # Upper level loop (Manager)
            while not done_m:
                
                # initialize upper action (Manager)
                manager_action = manager_state[0]
                
                # select upper action (Manager)
                
                # multiple active customers
                if env.statuses_list.count(1) > 0:
                    active_status_customers = [index for index, status in enumerate(env.statuses_list) if status == 1]
                    active_customer_pos = [customer_depot_pos[pos] for pos in active_status_customers]

                    # to ensure it does not select the current position and it is part of the active customers
                    while manager_action == manager_state[0] or manager_action not in active_customer_pos: 
                        
                        # epsilon greedy for action selection
                        manager_action = epsilon_greedy_policy(manager_q_table, manager_state, epsilon_manager, active_customer_pos)

                # only 1 active customer
                elif env.statuses_list.count(1) == 1:
                    
                    # assign the only remaining action
                    manager_action = active_customer_pos[0]
                
                # no active customers
                else:
                    manager_action = depot_position[0]


                step_count = 0

                done_w = False

                # sub goal consist of current customer position and next customer position (or depot)
                sub_goal = (manager_state[0], manager_action)
                
                # initial worker state equals the sub-goal
                worker_state = sub_goal

                time_out = False

                # Lower level loop (worker)
                while not done_w:
                    
                    # calculate the min required time to complete the sub-goal
                    min_required_time = env.manhattan(env.rev_position(worker_state[0]),env.rev_position(depot_position[0]) )

                    # if min time required higher than the remaining available steps and current position is not depot
                    if min_required_time >= max_steps - global_step_count - step_count and manager_state[0] != depot_position[0] and manager_action != depot_position[0]: #
                        
                        # re-direct the vehicle to the depot by adjusting the sub-goal
                        sub_goal = (manager_state[0], depot_position[0])
                        worker_state = (worker_state[0], depot_position[0])
                        manager_action = depot_position[0]

                        # if step count exceeds steps
                        if global_step_count == max_steps:
                            time_out = True

                    # worker action is selected based on epsilon greedy policy
                    worker_action = epsilon_greedy_policy(worker_big_Q[sub_goal], worker_state[0], epsilon_worker, env.actions)

                    # step
                    next_state_w, reward_w, done_w , _, inc_score  = env.step(worker_action, worker_state, time_out) 

                    # update lower Q table (worker)
                    worker_big_Q[sub_goal][worker_state[0]][worker_action] = update_Q_x(worker_alpha, gamma, worker_big_Q[sub_goal], worker_state[0], worker_action, reward_w, next_state_w)   
                    
                    worker_state = next_state_w

                    step_count += 1

                    score += inc_score 

                # if manager action is the depot position
                if manager_action == depot_position[0]:
                    
                    # termination condition met
                    done_m = True
                    
                    # collect -1 penalty for every step and for every unserved active customer
                    reward_m = step_count * (-1) + np.sum(np.array(env.statuses_list) == 1)*(-20) 
                    
                    tmp_scores.append(score)  
                    
                    if (episode % plot_every == 0):
                        avg_scores.append(np.mean(tmp_scores))
                        timestamps.append(time.time() - start_timestamp)

                else:
                    
                    # if customer is reached, collect -1 penalty for each step and reward of 20
                    reward_m =  step_count * (-1) + 20


                global_step_count += step_count

                # new manager state consists of manager action (or new position) and list of all updated customer statuses
                new_manager_state = (manager_action, *(env.statuses_list))

                # update upper Q table (manager)
                manager_q_table[manager_state][manager_action] = update_Q_x(manager_alpha, gamma, manager_q_table, manager_state, manager_action, reward_m, new_manager_state)    

                manager_state = new_manager_state

        plt.plot(np.linspace(0,num_episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))
        plt.xlabel('Episode Number')
        plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
        plt.savefig('H_'+ str(len(customers_coord))+ '_' + str(new_request_prob)+'.png')
        plt.show()

        print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))

        dict4 = {'episodes': list(range(plot_every,num_episodes +1,plot_every)),'avg_scores': avg_scores,'timestamps': timestamps }
        df4 = pd.DataFrame(dict4)
        df4.to_csv('H_'+ str(len(customers_coord))+ '_' + str(new_request_prob)+'.csv')

        return manager_q_table

    Q_x = q_learning(env, num_episodes)


    #policy evaluation

    evaluation = evaluation_switch

    # number of actions for upper policy
    nA_manager = len(customer_depot_pos)

    # initialize upper policy with zeros
    policy_manager = {key: np.zeros(nA_manager) for key in manager_q_table.keys()}

    # construct the upper policy table
    # for all statses of upper Q table
    for s in manager_q_table.keys():

        manager_locations = list(manager_q_table[s].keys())

        s_ordered = [None] * len(env.customers_pos)

        #to sequence the current position if it is customer
        try:
            d = env.customers_pos.index(s[0])
            s_ordered[d] = 2

        except ValueError:
            None

        #to sequence the other customers positions
        for s_el in manager_locations:
            
            try:
                d = env.customers_pos.index(s_el)
                
                s_ordered[d] = s[d+1]

            except ValueError:
                None

        g = np.argmax(np.array(list(manager_q_table[s].values()))*(np.array((s_ordered))==1) - 1e6*(np.array((s_ordered))!=1))

        policy_manager[s][g] = 1
    
    # construct the lower policy table
    policy_worker = {}

    for table_index in (sub_goals):
        # initialize lower policy table
        worker_small_Q = defaultdict(lambda: np.zeros(env.action_space.n))
        # Create dictionaries in each q table
        for loc_index in range(1,global_grid_size**2):
            for action_index in range(env.action_space.n):
                
                position_coord = env.rev_position(loc_index)

                new_position_coord = env.move_to_coord(position_coord[0],position_coord[1], action_index  )

                new_position = env._position(global_grid_size, new_position_coord[0], new_position_coord[1])

                if new_position_coord[0] < 1 or new_position_coord[1] < 1 or new_position_coord[0] >  global_grid_size -2 or new_position_coord[1] > global_grid_size -2:
                  
                    continue

                worker_small_Q[loc_index][action_index] = 0

        policy_worker[table_index] = worker_small_Q

    # for each state in lower Q table
    for s in worker_big_Q.keys():
        
        # assign probability 1 to actions with highest Q value
        for pos in range(1,global_grid_size**2):
            g = int(np.argmax(worker_big_Q[s][pos]))
            policy_worker[s][pos][g] = 1
    

    num_episodes = 10000
    output_quality = []
    avg_scores_X = deque(maxlen=num_episodes)
    plot_every_x = 100

    success_count = 0

    for episode in range(1,num_episodes+1): 

        # reset env
        env.reset(1)

        # initialize upper agent state
        manager_state = (env.agent_start_pos_nr  , *(list(env.statuses_list))) 

        done_m = False
        global_step_count = 0
        score = 0 

        while not done_m:
            
            # initialize upper agent action
            manager_action = manager_state[0]
            
            # select upper agent action
            # multiple active customers
            if env.statuses_list.count(1) > 0:
                active_status_customers = [index for index, status in enumerate(env.statuses_list) if status == 1]
                active_customer_pos = [customer_depot_pos[pos] for pos in active_status_customers]
                
                policy_of_active_customers = [policy_manager[manager_state][pos] for pos in active_status_customers]

                # to ensure it does not select the current position and it is part of the active customers
                while manager_action == manager_state[0] or manager_action not in active_customer_pos: 
                    
                    # get action with highest probability
                    manager_action = np.random.choice(active_customer_pos, p = policy_of_active_customers)
            
            # only 1 active customer
            elif env.statuses_list.count(1) == 1:
                
                # assign the remaining active customer as action 
                manager_action = active_customer_pos[0]
            
            # no active customers
            else:
                manager_action = depot_position[0]

            step_count = 0
            done_w = False

            # set sub-goal as the current customer position and new customer position
            sub_goal = (manager_state[0], manager_action)

            worker_state = sub_goal

            time_out = False

            while not done_w:

                # calculate the min required time to complete the sub-goal 
                min_required_time = env.manhattan(env.rev_position(worker_state[0]),env.rev_position(depot_position[0]) )

                # if min time required higher than the remaining available steps and current position is not depot
                if min_required_time >= max_steps - global_step_count - step_count and manager_state[0] != depot_position[0] and manager_action != depot_position[0]: #
                    
                    # re-direct the vehicle to the depot by adjusting the sub-goal
                    sub_goal = (manager_state[0], depot_position[0])
                    worker_state = (worker_state[0], depot_position[0])
                    manager_action = depot_position[0]

                    # if step count exceeds steps
                    if global_step_count >= max_steps: 
                        time_out = True

                # worker action is selected based on optimal policy
                worker_action = np.random.choice(env.actions, p = policy_worker[sub_goal][worker_state[0]])

                # step
                next_state_w, reward_w, done_w , _, inc_score  = env.step(worker_action, worker_state, time_out) 

                worker_state = next_state_w

                step_count += 1

                score += inc_score 

            # if manager action is the depot position
            if manager_action == depot_position[0]:
                
                # termination condition met
                done_m = True

                if global_step_count + step_count <= max_steps: 
                    success_count += 1

                # collect -1 penalty for every step and for every unserved active customer
                reward_m = step_count * (-1) + np.sum(np.array(env.statuses_list) == 1)*(-20) 
            
                tmp_scores.append(score)  

                if (episode % plot_every_x == 0):
                    avg_scores_X.append(np.mean(tmp_scores))

            else:
                # if customer is reached, collect -1 penalty for each step and reward of 20
                reward_m =  step_count * (-1) + 20

            global_step_count += step_count

            # new manager state consists of manager action (or new position) and list of all updated customer statuses
            new_manager_state = (manager_action, *(env.statuses_list))
            
            manager_state = new_manager_state

        output_quality.append(score)

    plt.plot(np.linspace(0,num_episodes,len(avg_scores_X),endpoint=False), np.asarray(avg_scores_X))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every_x)
    plt.ylim(bottom=0)
    plt.show()
    print(('Best Average Reward over %d Episodes: ' % plot_every_x), np.max(avg_scores_X))

    print(np.mean(output_quality))
    print(np.mean(success_count/num_episodes))

