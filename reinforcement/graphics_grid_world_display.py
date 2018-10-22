"""Graphical display for grid worlds.

Champlain College CSI-480, Fall 2018
The following code was adapted by Joshua Auerbach (jauerbach@champlain.edu)
from the UC Berkeley Pacman Projects (see license and attribution below).

----------------------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""

import util
import graphics_utils as gu
from functools import reduce


class GraphicsGridWorldDisplay:
    """Graphical display for grid world (controlling class)."""

    def __init__(self, grid_world, size=120, speed=1.0):
        """Create graphics given a gird world and parameters."""
        self.grid_world = grid_world
        self.size = size
        self.speed = speed

    def start(self):
        """Start the display."""
        setup(self.grid_world, size=self.size)

    def pause(self):
        """Wait for key press."""
        gu.wait_for_keys()

    def display_values(self, agent, current_state=None,
                       message='Agent Values'):
        """Display state values."""
        values = util.Counter()
        policy = {}
        states = self.grid_world.get_states()
        for state in states:
            values[state] = agent.get_value(state)
            policy[state] = agent.get_policy(state)
        draw_values(self.grid_world, values, policy, current_state, message)
        gu.sleep(0.05 / self.speed)

    def display_null_values(self, current_state=None, message=''):
        """Display NULL values."""
        values = util.Counter()
        states = self.grid_world.get_states()
        for state in states:
            values[state] = 0.0
        draw_null_values(self.grid_world, current_state, '')
        gu.sleep(0.05 / self.speed)

    def display_q_values(self, agent, current_state=None,
                         message='Agent Q-Values'):
        """Display q values."""
        q_values = util.Counter()
        states = self.grid_world.get_states()
        for state in states:
            for action in self.grid_world.get_possible_actions(state):
                q_values[(state, action)] = agent.get_q_value(state, action)
        draw_q_values(self.grid_world, q_values, current_state, message)
        gu.sleep(0.05 / self.speed)


BACKGROUND_COLOR = gu.format_color(0, 0, 0)
EDGE_COLOR = gu.format_color(1, 1, 1)
OBSTACLE_COLOR = gu.format_color(0.5, 0.5, 0.5)
TEXT_COLOR = gu.format_color(1, 1, 1)
MUTED_TEXT_COLOR = gu.format_color(0.7, 0.7, 0.7)
LOCATION_COLOR = gu.format_color(0, 0, 1)

WINDOW_SIZE = -1
GRID_SIZE = -1
GRID_HEIGHT = -1
MARGIN = -1


def setup(grid_world, title="GridWorld Display", size=120):
    """Set up the display."""
    global GRID_SIZE, MARGIN, SCREEN_WIDTH, SCREEN_HEIGHT, GRID_HEIGHT,\
        WINDOW_SIZE
    grid = grid_world.grid
    WINDOW_SIZE = size
    GRID_SIZE = size
    GRID_HEIGHT = grid.height
    MARGIN = GRID_SIZE * 0.75
    screen_width = (grid.width - 1) * GRID_SIZE + MARGIN * 2
    screen_height = (grid.height - 0.5) * GRID_SIZE + MARGIN * 2

    gu.begin_graphics(screen_width,
                      screen_height,
                      BACKGROUND_COLOR, title=title)


def draw_null_values(grid_world, current_state=None, message=''):
    """Draw NULL values."""
    grid = grid_world.grid
    blank()
    for x in range(grid.width):
        for y in range(grid.height):
            state = (x, y)
            grid_type = grid[x][y]
            is_exit = (str(grid_type) != grid_type)
            is_current = (current_state == state)
            if grid_type == '#':
                draw_square(x, y, 0, 0, 0, None, None, True, False, is_current)
            else:
                draw_null_square(grid_world.grid, x, y, False, is_exit,
                                 is_current)
    pos = to_screen(((grid.width - 1.0) / 2.0, -0.8))
    gu.text(pos, TEXT_COLOR, message, "Courier", -32, "bold", "c")


def draw_values(grid_world, values, policy, current_state=None,
                message='State Values'):
    """Draw state values."""
    grid = grid_world.grid
    blank()
    value_list = [values[state] for state in grid_world.get_states()] + [0.0]
    min_value = min(value_list)
    max_value = max(value_list)
    for x in range(grid.width):
        for y in range(grid.height):
            state = (x, y)
            grid_type = grid[x][y]
            is_exit = (str(grid_type) != grid_type)
            is_current = (current_state == state)
            if grid_type == '#':
                draw_square(x, y, 0, 0, 0, None, None, True, False, is_current)
            else:
                value = values[state]
                action = None
                if policy is not None and state in policy:
                    action = policy[state]
                    actions = grid_world.get_possible_actions(state)
                if action not in actions and 'exit' in actions:
                    action = 'exit'
                val_string = '%.2f' % value
                draw_square(x, y, value, min_value, max_value, val_string,
                            action, False, is_exit, is_current)
    pos = to_screen(((grid.width - 1.0) / 2.0, -0.8))
    gu.text(pos, TEXT_COLOR, message, "Courier", -32, "bold", "c")


def draw_q_values(grid_world, q_values, current_state=None,
                  message='State-Action Q-Values'):
    """Draw q-values."""
    grid = grid_world.grid
    blank()
    state_cross_actions = [[(state, action) for action in
                            grid_world.get_possible_actions(state)]
                           for state in grid_world.get_states()]
    q_states = reduce(lambda x, y: x + y, state_cross_actions, [])
    q_value_list = [q_values[(state, action)]
                    for state, action in q_states] + [0.0]
    min_value = min(q_value_list)
    max_value = max(q_value_list)
    for x in range(grid.width):
        for y in range(grid.height):
            state = (x, y)
            grid_type = grid[x][y]
            is_exit = (str(grid_type) != grid_type)
            is_current = (current_state == state)
            actions = grid_world.get_possible_actions(state)
            if actions is None or len(actions) == 0:
                actions = [None]
            # best_q = max([q_values[(state, action)] for action in actions])
            # best_actions = [action for action in actions
            #                if q_values[(state, action)] == best_q]

            q = util.Counter()
            val_strings = {}
            for action in actions:
                v = q_values[(state, action)]
                q[action] += v
                val_strings[action] = '%.2f' % v
            if grid_type == '#':
                draw_square(x, y, 0, 0, 0, None, None, True, False, is_current)
            elif is_exit:
                action = 'exit'
                value = q[action]
                val_string = '%.2f' % value
                draw_square(x, y, value, min_value, max_value, val_string,
                            action, False, is_exit, is_current)
            else:
                draw_square_q(x, y, q, min_value, max_value, val_strings,
                              actions, is_current)
    pos = to_screen(((grid.width - 1.0) / 2.0, -0.8))
    gu.text(pos, TEXT_COLOR, message, "Courier", -32, "bold", "c")


def blank():
    """Clear the screen."""
    gu.clear_screen()


def draw_null_square(grid, x, y, is_obstacle, is_terminal, is_current):
    """Draw NULL square at given location."""
    square_color = get_color(0, -1, 1)

    if is_obstacle:
        square_color = OBSTACLE_COLOR

    (screen_x, screen_y) = to_screen((x, y))
    square((screen_x, screen_y),
           0.5 * GRID_SIZE,
           color=square_color,
           filled=1,
           width=1)

    square((screen_x, screen_y),
           0.5 * GRID_SIZE,
           color=EDGE_COLOR,
           filled=0,
           width=3)

    if is_terminal and not is_obstacle:
        square((screen_x, screen_y),
               0.4 * GRID_SIZE,
               color=EDGE_COLOR,
               filled=0,
               width=2)
        gu.text((screen_x, screen_y),
                TEXT_COLOR,
                str(grid[x][y]),
                "Courier", -24, "bold", "c")

    # text_color = TEXT_COLOR

    if not is_obstacle and is_current:
        gu.circle((screen_x, screen_y), 0.1 * GRID_SIZE, LOCATION_COLOR,
                  fill_color=LOCATION_COLOR)

    # if not is_obstacle:
    #   text( (screen_x, screen_y), text_color, val_str, "Courier", 24,
    #          "bold", "c")


def draw_square(x, y, val, min_, max_, val_str, action, is_obstacle,
                is_terminal, is_current):
    """Draw square at given location."""
    square_color = get_color(val, min_, max_)

    if is_obstacle:
        square_color = OBSTACLE_COLOR

    (screen_x, screen_y) = to_screen((x, y))
    square((screen_x, screen_y),
           0.5 * GRID_SIZE,
           color=square_color,
           filled=1,
           width=1)
    square((screen_x, screen_y),
           0.5 * GRID_SIZE,
           color=EDGE_COLOR,
           filled=0,
           width=3)
    if is_terminal and not is_obstacle:
        square((screen_x, screen_y),
               0.4 * GRID_SIZE,
               color=EDGE_COLOR,
               filled=0,
               width=2)

    if action == 'north':
        gu.polygon([(screen_x, screen_y - 0.45 * GRID_SIZE),
                    (screen_x + 0.05 * GRID_SIZE, screen_y - 0.40 * GRID_SIZE),
                    (screen_x - 0.05 * GRID_SIZE, screen_y - 0.40 * GRID_SIZE)
                    ], EDGE_COLOR, filled=1, smoothed=False)
    if action == 'south':
        gu.polygon([(screen_x, screen_y + 0.45 * GRID_SIZE),
                    (screen_x + 0.05 * GRID_SIZE, screen_y + 0.40 * GRID_SIZE),
                    (screen_x - 0.05 * GRID_SIZE, screen_y + 0.40 * GRID_SIZE)
                    ], EDGE_COLOR, filled=1, smoothed=False)
    if action == 'west':
        gu.polygon([(screen_x - 0.45 * GRID_SIZE, screen_y),
                    (screen_x - 0.4 * GRID_SIZE, screen_y + 0.05 * GRID_SIZE),
                    (screen_x - 0.4 * GRID_SIZE, screen_y - 0.05 * GRID_SIZE)
                    ], EDGE_COLOR, filled=1, smoothed=False)
    if action == 'east':
        gu.polygon([(screen_x + 0.45 * GRID_SIZE, screen_y),
                    (screen_x + 0.4 * GRID_SIZE, screen_y + 0.05 * GRID_SIZE),
                    (screen_x + 0.4 * GRID_SIZE, screen_y - 0.05 * GRID_SIZE)
                    ], EDGE_COLOR, filled=1, smoothed=False)

    text_color = TEXT_COLOR

    if not is_obstacle and is_current:
        gu.circle((screen_x, screen_y), 0.1 * GRID_SIZE,
                  outline_color=LOCATION_COLOR, fill_color=LOCATION_COLOR)

    if not is_obstacle:
        gu.text((screen_x, screen_y), text_color, val_str,
                "Courier", -30, "bold", "c")


def draw_square_q(x, y, q_vals, min_val, max_val, val_strs, best_actions,
                  is_current):
    """Draw square with q values."""
    (screen_x, screen_y) = to_screen((x, y))

    center = (screen_x, screen_y)
    nw = (screen_x - 0.5 * GRID_SIZE, screen_y - 0.5 * GRID_SIZE)
    ne = (screen_x + 0.5 * GRID_SIZE, screen_y - 0.5 * GRID_SIZE)
    se = (screen_x + 0.5 * GRID_SIZE, screen_y + 0.5 * GRID_SIZE)
    sw = (screen_x - 0.5 * GRID_SIZE, screen_y + 0.5 * GRID_SIZE)
    n = (screen_x, screen_y - 0.5 * GRID_SIZE + 5)
    s = (screen_x, screen_y + 0.5 * GRID_SIZE - 5)
    w = (screen_x - 0.5 * GRID_SIZE + 5, screen_y)
    e = (screen_x + 0.5 * GRID_SIZE - 5, screen_y)

    actions = list(q_vals.keys())
    for action in actions:

        wedge_color = get_color(q_vals[action], min_val, max_val)

        if action == 'north':
            gu.polygon((center, nw, ne), wedge_color, filled=1, smoothed=False)
            # text(n, text_color, val_str, "Courier", 8, "bold", "n")
        if action == 'south':
            gu.polygon((center, sw, se), wedge_color, filled=1, smoothed=False)
            # text(s, text_color, val_str, "Courier", 8, "bold", "s")
        if action == 'east':
            gu.polygon((center, ne, se), wedge_color, filled=1, smoothed=False)
            # text(e, text_color, val_str, "Courier", 8, "bold", "e")
        if action == 'west':
            gu.polygon((center, nw, sw), wedge_color, filled=1, smoothed=False)
            # text(w, text_color, val_str, "Courier", 8, "bold", "w")

    square((screen_x, screen_y),
           0.5 * GRID_SIZE,
           color=EDGE_COLOR,
           filled=0,
           width=3)
    gu.line(ne, sw, color=EDGE_COLOR)
    gu.line(nw, se, color=EDGE_COLOR)

    if is_current:
        gu.circle((screen_x, screen_y),
                  0.1 * GRID_SIZE, LOCATION_COLOR, fill_color=LOCATION_COLOR)

    for action in actions:
        text_color = TEXT_COLOR
        if q_vals[action] < max(q_vals.values()):
            text_color = MUTED_TEXT_COLOR
        val_str = ""
        if action in val_strs:
            val_str = val_strs[action]
        h = -20
        if action == 'north':
            # polygon( (center, nw, ne), wedge_color, filled = 1, smooth = 0)
            gu.text(n, text_color, val_str, "Courier", h, "bold", "n")
        if action == 'south':
            # polygon( (center, sw, se), wedge_color, filled = 1, smooth = 0)
            gu.text(s, text_color, val_str, "Courier", h, "bold", "s")
        if action == 'east':
            # polygon( (center, ne, se), wedge_color, filled = 1, smooth = 0)
            gu.text(e, text_color, val_str, "Courier", h, "bold", "e")
        if action == 'west':
            # polygon( (center, nw, sw), wedge_color, filled = 1, smooth = 0)
            gu.text(w, text_color, val_str, "Courier", h, "bold", "w")


def get_color(val, min_val, max_val):
    """Get color for a value based on min and max values."""
    r, g = 0.0, 0.0
    if val < 0 and min_val < 0:
        r = val * 0.65 / min_val
    if val > 0 and max_val > 0:
        g = val * 0.65 / max_val
    return gu.format_color(r, g, 0.0)


def square(pos, size, color, filled, width):
    """Create a square with given properties."""
    x, y = pos
    dx, dy = size, size
    return gu.polygon([(x - dx, y - dy), (x - dx, y + dy),
                       (x + dx, y + dy), (x + dx, y - dy)],
                      outline_color=color, fill_color=color, filled=filled,
                      width=width, smoothed=False)


def to_screen(point):
    """Convert a point in grid coordinates to screen coordinates."""
    (grid_x, grid_y) = point
    x = grid_x * GRID_SIZE + MARGIN
    y = (GRID_HEIGHT - grid_y - 1) * GRID_SIZE + MARGIN
    return (x, y)


def to_grid(point):
    """Convert from screen coordinate to grid coordinates."""
    (x, y) = point
    x = int((y - MARGIN + GRID_SIZE * 0.5) / GRID_SIZE)
    y = int((x - MARGIN + GRID_SIZE * 0.5) / GRID_SIZE)
    print(point, "-->", (x, y))
    return (x, y)
