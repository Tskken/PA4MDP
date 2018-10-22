"""Classes to support a GridWorld Markov Decision Process.

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

import random
import sys
import mdp
import environment
import util
import optparse
import re


class GridWorld(mdp.MarkovDecisionProcess):
    """GridWorld MDP comprised of a grid, a living reward, and noise."""

    def __init__(self, grid, living_reward=0.0, noise=0.2):
        """Create GridWorld from given grid, living_reward, and noise."""
        # layout
        if isinstance(grid, list):
            grid = make_grid(grid)
        self.grid = grid

        # parameters
        self.living_reward = living_reward
        self.noise = noise

    def set_living_reward(self, reward):
        """Set the (negative) reward for exiting "normal" states.

        Note that in the R+N text, this reward is on entering a state and
        therefore is not clearly part of the state's uture rewards.
        """
        self.living_reward = reward

    def set_noise(self, noise):
        """Set the probability of moving in an unintended direction."""
        self.noise = noise

    def get_possible_actions(self, state):
        """Return list of valid actions for 'state'.

        Note that you can request moves into walls and that "exit" states
        transition to the terminal state under the special action "done".
        """
        if state == self.grid.terminal_state:
            return ()
        x, y = state
        if type(self.grid[x][y]) == int:
            return ('exit', )
        return ('north', 'west', 'south', 'east')

    def get_states(self):
        """Return list of all states."""
        # The true terminal state.
        states = [self.grid.terminal_state]
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if self.grid[x][y] != '#':
                    state = (x, y)
                    states.append(state)
        return states

    def get_reward(self, state, action, next_state):
        """Get reward for state, action, next_state transition.

        Note that the reward depends only on the state being departed
        (as in the R+N book examples, which more or less use this convention).
        """
        if state == self.grid.terminal_state:
            return 0.0
        x, y = state
        cell = self.grid[x][y]
        if type(cell) == int or type(cell) == float:
            return cell
        return self.living_reward

    def get_start_state(self):
        """Get the starting state of this GridWorld."""
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if self.grid[x][y] == 'S':
                    return (x, y)
        raise Exception('Grid has no start state')

    def is_terminal(self, state):
        """Return if state is a terminal state.

        Note: Only the TERMINAL_STATE state is *actually* a terminal state.
        The other "exit" states are technically non-terminals with a single
        action "exit" which leads to the true terminal state.

        This convention is to make the grids line up with the examples
        in the R+N textbook.
        """
        return state == self.grid.terminal_state

    def get_transition_states_and_probs(self, state, action):
        """Return list of (next_state, prob) pairs.

        These represent the states reachable from 'state' by taking
        'action' along with their transition probabilities.
        """
        if action not in self.get_possible_actions(state):
            raise Exception("Illegal action!")

        if self.is_terminal(state):
            return []

        x, y = state

        if type(self.grid[x][y]) == int or type(self.grid[x][y]) == float:
            term_state = self.grid.terminal_state
            return [(term_state, 1.0)]

        successors = []

        north_state = (self.__is_allowed(y + 1, x) and (x, y + 1)) or state
        west_state = (self.__is_allowed(y, x - 1) and (x - 1, y)) or state
        south_state = (self.__is_allowed(y - 1, x) and (x, y - 1)) or state
        east_state = (self.__is_allowed(y, x + 1) and (x + 1, y)) or state

        if action == 'north' or action == 'south':
            if action == 'north':
                successors.append((north_state, 1 - self.noise))
            else:
                successors.append((south_state, 1 - self.noise))

            mass_left = self.noise
            successors.append((west_state, mass_left / 2.0))
            successors.append((east_state, mass_left / 2.0))

        if action == 'west' or action == 'east':
            if action == 'west':
                successors.append((west_state, 1 - self.noise))
            else:
                successors.append((east_state, 1 - self.noise))

            mass_left = self.noise
            successors.append((north_state, mass_left / 2.0))
            successors.append((south_state, mass_left / 2.0))

        successors = self.__aggregate(successors)

        return successors

    def __aggregate(self, states_and_probs):
        counter = util.Counter()
        for state, prob in states_and_probs:
            counter[state] += prob
        new_states_and_probs = []
        for state, prob in sorted(list(counter.items())):
            new_states_and_probs.append((state, prob))
        return new_states_and_probs

    def __is_allowed(self, y, x):
        if y < 0 or y >= self.grid.height:
            return False
        if x < 0 or x >= self.grid.width:
            return False
        return self.grid[x][y] != '#'


class GridWorldEnvironment(environment.Environment):
    """A GridWorldEnvironment."""

    def __init__(self, grid_world):
        """Create environment from grid_world and put agent in start state."""
        self.grid_world = grid_world
        self.reset()

    def get_current_state(self):
        """Return the current state of enviornment.

        Overrides environment.Environment.get_current_state
        """
        return self.state

    def get_possible_actions(self, state):
        """Return possible actions the agent can take in the given state.

        Can return the empty list if we are in a terminal state.

        Overrides environment.Environment.get_possible_actions
        """
        return self.grid_world.get_possible_actions(state)

    def do_action(self, action):
        """Perform the given action in the current environment state.

        Update the enviornment.

        Return a (reward, next_state) pair.

        Overrides environment.Environment.do_action
        """
        state = self.get_current_state()
        (next_state, reward) = self.get_random_next_state(state, action)
        self.state = next_state
        return (next_state, reward)

    def get_random_next_state(self, state, action, rand_obj=None):
        """Sample from transitions of taking action from state."""
        rand = -1.0
        if rand_obj is None:
            rand = random.random()
        else:
            rand = rand_obj.random()
        sum = 0.0
        successors = self.grid_world.get_transition_states_and_probs(state,
                                                                     action)
        for next_state, prob in successors:
            sum += prob
            if sum > 1.0:
                raise Exception('Total transition probability more than one; '
                                'sample failure.')
            if rand < sum:
                reward = self.grid_world.get_reward(state, action, next_state)
                return (next_state, reward)
        raise Exception('Total transition probability less than one; '
                        'sample failure.')

    def reset(self):
        """Reset the current state to the start state.

        Overrides environment.Environment.reset
        """
        self.state = self.grid_world.get_start_state()


class Grid:
    """A 2-dimensional array of immutables backed by a list of lists.

    Data is accessed via grid[x][y] where (x,y) are cartesian coordinates
    with x horizontal, y vertical and the origin (0,0) in the bottom left
    corner.

    The __str__ method constructs an output that is oriented appropriately.
    """

    def __init__(self, width, height, initial_value=' '):
        """Create Grid of given size.

        Initialize all cells with initial_value.
        """
        self.width = width
        self.height = height
        self.data = [[initial_value for y in range(height)]
                     for x in range(width)]
        self.terminal_state = 'TERMINAL_STATE'

    def __getitem__(self, i):
        """Return ith row of data, supports read access via grid[x][y]."""
        return self.data[i]

    def __setitem__(self, i, item):
        """Set ith row of data, supports write access via grid[x][y]."""
        self.data[i] = item

    def __eq__(self, other):
        """Return if two Grids are equal."""
        if other is None:
            return False
        return self.data == other.data

    def __hash__(self):
        """Use immutable form of data for hashing."""
        return hash(tuple(self.data))

    def copy(self):
        """Deep copy the grid."""
        g = Grid(self.width, self.height)
        g.data = [x[:] for x in self.data]
        return g

    def deep_copy(self):
        """Deep copy the grid (alternate method)."""
        return self.copy()

    def shallow_copy(self):
        """Shallow copy the grid (will share data object)."""
        g = Grid(self.width, self.height)
        g.data = self.data
        return g

    def _get_legacy_text(self):
        t = [[self.data[x][y] for x in range(self.width)]
             for y in range(self.height)]
        t.reverse()
        return t

    def __str__(self):
        """Return text representation of the grid oriented properly."""
        return str(self._get_legacy_text())


def make_grid(grid_string):
    """Make a Grid object from a string description."""
    width, height = len(grid_string[0]), len(grid_string)
    grid = Grid(width, height)
    for ybar, line in enumerate(grid_string):
        y = height - ybar - 1
        for x, el in enumerate(line):
            grid[x][y] = el
    return grid


# Example grids

def get_cliff_grid():
    """Make cliff grid."""
    grid = [[' ', ' ', ' ', ' ', ' '],
            ['S', ' ', ' ', ' ', 10],
            [-100, -100, -100, -100, -100]]
    return GridWorld(make_grid(grid))


def get_cliff_grid2():
    """Make cliff grid 2."""
    grid = [[' ', ' ', ' ', ' ', ' '],
            [8, 'S', ' ', ' ', 10],
            [-100, -100, -100, -100, -100]]
    return GridWorld(grid)


def get_discount_grid():
    """Make discount grid."""
    grid = [[' ', ' ', ' ', ' ', ' '],
            [' ', '#', ' ', ' ', ' '],
            [' ', '#', 1, '#', 10],
            ['S', ' ', ' ', ' ', ' '],
            [-10, -10, -10, -10, -10]]
    return GridWorld(grid)


def get_bridge_grid():
    """Make bridge grid."""
    grid = [['#', -100, -100, -100, -100, -100, '#'],
            [1, 'S', ' ', ' ', ' ', ' ', 10],
            ['#', -100, -100, -100, -100, -100, '#']]
    return GridWorld(grid)


def get_book_grid():
    """Make cliff grid."""
    grid = [[' ', ' ', ' ', +1],
            [' ', '#', ' ', -1],
            ['S', ' ', ' ', ' ']]
    return GridWorld(grid)


def get_maze_grid():
    """Make maze grid."""
    grid = [[' ', ' ', ' ', +1],
            ['#', '#', ' ', '#'],
            [' ', '#', ' ', ' '],
            [' ', '#', '#', ' '],
            ['S', ' ', ' ', ' ']]
    return GridWorld(grid)


def get_user_action(state, action_function):
    """Get an action from the user (rather than the agent).

    Used for debugging and lecture demos.
    """
    import graphics_utils
    action = None
    while True:
        keys = graphics_utils.wait_for_keys()
        if 'Up' in keys:
            action = 'north'
        if 'Down' in keys:
            action = 'south'
        if 'Left' in keys:
            action = 'west'
        if 'Right' in keys:
            action = 'east'
        if 'q' in keys:
            sys.exit(0)
        if action is None:
            continue
        break
    actions = action_function(state)
    if action not in actions:
        action = actions[0]
    return action


def run_episode(agent, environment, discount, decision, display,
                message, pause, episode):
    """Run an episode of agent in environment."""
    returns = 0
    total_discount = 1.0
    environment.reset()
    if 'start_episode' in dir(agent):
        agent.start_episode()
    message("BEGINNING EPISODE: " + str(episode) + "\n")
    while True:

        # DISPLAY CURRENT STATE
        state = environment.get_current_state()
        display(state)
        pause()

        # END IF IN A TERMINAL STATE
        actions = environment.get_possible_actions(state)
        if len(actions) == 0:
            message("EPISODE " + str(episode) + " COMPLETE: RETURN WAS "
                    + str(returns) + "\n")
            return returns

        # GET ACTION (USUALLY FROM AGENT)
        action = decision(state)
        if action is None:
            raise Exception('Error: Agent returned None action')

        # EXECUTE ACTION
        next_state, reward = environment.do_action(action)
        message("Started in state: " + str(state) +
                "\nTook action: " + str(action) +
                "\nEnded in state: " + str(next_state) +
                "\nGot reward: " + str(reward) + "\n")
        # UPDATE LEARNER
        if 'observe_transition' in dir(agent):
            agent.observe_transition(state, action, next_state, reward)

        returns += reward * total_discount
        total_discount *= discount

    if 'stop_episode' in dir(agent):
        agent.stop_episode()


def parse_options():
    """Parse options when calling from command line."""
    opt_parser = optparse.OptionParser()
    opt_parser.add_option('-d', '--discount', action='store',
                          type='float', dest='discount', default=0.9,
                          help='Discount on future (default %default)')
    opt_parser.add_option('-r', '--living_reward', action='store',
                          type='float', dest='living_reward', default=0.0,
                          metavar="R", help='Reward for living for a time '
                                            'step (default %default)')
    opt_parser.add_option('-n', '--noise', action='store',
                          type='float', dest='noise', default=0.2,
                          metavar="P", help='How often action results in ' +
                          'unintended direction (default %default)')
    opt_parser.add_option('-e', '--epsilon', action='store',
                          type='float', dest='epsilon', default=0.3,
                          metavar="E", help='Chance of taking a random action '
                                            'in q-learning (default %default)')
    opt_parser.add_option('-l', '--learning_rate', action='store',
                          type='float', dest='learning_rate', default=0.5,
                          metavar="P",
                          help='TD learning rate (default %default)')
    opt_parser.add_option('-i', '--iterations', action='store',
                          type='int', dest='iters', default=10,
                          metavar="K", help='Number of rounds of value '
                                            'iteration (default %default)')
    opt_parser.add_option('-k', '--episodes', action='store',
                          type='int', dest='episodes', default=1,
                          metavar="K", help='Number of epsiodes of the MDP '
                                            'to run (default %default)')
    opt_parser.add_option('-g', '--grid', action='store',
                          metavar="G", type='string', dest='grid',
                          default="BookGrid",
                          help='Grid to use (case sensitive; options are '
                               'BookGrid, BridgeGrid, CliffGrid, MazeGrid, '
                               'default %default)')
    opt_parser.add_option('-w', '--window_size', metavar="X", type='int',
                          dest='grid_size', default=150,
                          help='Request a window width of X pixels '
                               '*per grid cell* (default %default)')
    opt_parser.add_option('-a', '--agent', action='store', metavar="A",
                          type='string', dest='agent', default="random",
                          help="Agent type (options are 'random', 'value' and "
                               "'q', default %default)")
    opt_parser.add_option('-t', '--text', action='store_true',
                          dest='text_display', default=False,
                          help='Use text-only ASCII display')
    opt_parser.add_option('-p', '--pause', action='store_true',
                          dest='pause', default=False,
                          help='Pause GUI after each time step '
                               'when running the MDP')
    opt_parser.add_option('-q', '--quiet', action='store_true',
                          dest='quiet', default=False,
                          help='Skip display of any learning episodes')
    opt_parser.add_option('-s', '--speed', action='store', metavar="S",
                          type=float, dest='speed', default=1.0,
                          help='Speed of animation, S > 1.0 is faster, '
                               '0.0 < S < 1.0 is slower (default %default)')
    opt_parser.add_option('-m', '--manual', action='store_true',
                          dest='manual', default=False,
                          help='Manually control agent')
    opt_parser.add_option('-v', '--value_steps', action='store_true',
                          default=False,
                          help='Display each step of value iteration')

    opts, args = opt_parser.parse_args()

    if opts.manual and opts.agent != 'q':
        print('## Disabling Agents in Manual Mode (-m) ##')
        opts.agent = None

    # MANAGE CONFLICTS
    if opts.text_display or opts.quiet:
        opts.pause = False

    if opts.manual:
        opts.pause = True

    if opts.manual and opts.text_display:
        opt_parser.error("Cannot run in manual mode with text display.")

    return opts


FIRST_CAP_RE = re.compile('(.)([A-Z][a-z]+)')
ALL_CAP_RE = re.compile('([a-z0-9])([A-Z])')


def get_function(name):
    """Get function for creating given grid."""
    s1 = FIRST_CAP_RE.sub(r'\1_\2', name)
    return "get_" + ALL_CAP_RE.sub(r'\1_\2', s1).lower()


def main():
    """Run as program."""
    opts = parse_options()
    ###########################
    # GET THE GRIDWORLD
    ###########################

    import grid_world

    mdp_function = getattr(grid_world, get_function(opts.grid))
    mdp = mdp_function()
    mdp.set_living_reward(opts.living_reward)
    mdp.set_noise(opts.noise)
    env = grid_world.GridWorldEnvironment(mdp)
    ###########################
    # GET THE DISPLAY ADAPTER
    ###########################

    import text_grid_world_display
    display = text_grid_world_display.TextGridWorldDisplay(mdp)
    if not opts.text_display:
        import graphics_grid_world_display
        display = graphics_grid_world_display.GraphicsGridWorldDisplay(
                    mdp, opts.grid_size, opts.speed)
    try:
        display.start()
    except KeyboardInterrupt:
        sys.exit(0)
    ###########################
    # GET THE AGENT
    ###########################

    import value_iteration_agents
    import q_learning_agents
    a = None
    if opts.agent == 'value':
        a = value_iteration_agents.ValueIterationAgent(
                mdp, opts.discount, opts.iters)
    elif opts.agent == 'q':
        def action_fn(state):
            return mdp.get_possible_actions(state)
        q_learn_opts = {'gamma': opts.discount,
                        'alpha': opts.learning_rate,
                        'epsilon': opts.epsilon,
                        'action_fn': action_fn}
        a = q_learning_agents.QLearningAgent(**q_learn_opts)
    elif opts.agent == 'random':
        # No reason to use the random agent without episodes
        if opts.episodes == 0:
            opts.episodes = 10

        class RandomAgent:
            def get_action(self, state):
                return random.choice(mdp.get_possible_actions(state))

            def get_value(self, state):
                return 0.0

            def get_q_value(self, state, action):
                return 0.0

            def get_policy(self, state):
                """NOTE: 'random' is a special policy value.

                Don't use it in your code.
                """
                return 'random'

            def update(self, state, action, next_state, reward):
                pass

        a = RandomAgent()
    else:
        if not opts.manual:
            raise Exception('Unknown agent type: ' + opts.agent)

    ###########################
    # RUN EPISODES
    ###########################
    # DISPLAY Q/V VALUES BEFORE SIMULATION OF EPISODES
    try:
        if not opts.manual and opts.agent == 'value':
            if opts.value_steps:
                for i in range(opts.iters):
                    temp_agent = value_iteration_agents.ValueIterationAgent(
                                    mdp, opts.discount, i)
                    display.display_values(temp_agent,
                                           message="VALUES AFTER " + str(i) +
                                           " ITERATIONS")
                    display.pause()

            display.display_values(a, message="VALUES AFTER " +
                                              str(opts.iters) + " ITERATIONS")
            display.pause()
            display.display_q_values(a, message="Q-VALUES AFTER " +
                                                str(opts.iters) + " ITERATIONS"
                                     )
            display.pause()
    except KeyboardInterrupt:
        sys.exit(0)

    # FIGURE OUT WHAT TO DISPLAY EACH TIME STEP (IF ANYTHING)
    if not opts.quiet:
        if opts.manual and opts.agent is None:
            def display_callback(state):
                display.display_null_values(state)
        else:
            if opts.agent == 'random':
                def display_callback(state):
                    display.display_values(a, state, "CURRENT VALUES")
            elif opts.agent == 'value':
                def display_callback(state):
                    display.display_values(a, state, "CURRENT VALUES")
            elif opts.agent == 'q':
                def display_callback(state):
                    display.display_q_values(a, state, "CURRENT Q-VALUES")
    else:
        def display_callback(x):
            pass

    def message_callback(x):
        if not opts.quiet:
            print(x)

    # FIGURE OUT WHETHER TO WAIT FOR A KEY PRESS AFTER EACH TIME STEP
    def pause_callback():
        if opts.pause:
            display.pause()

    # FIGURE OUT WHETHER THE USER WANTS MANUAL CONTROL
    # (FOR DEBUGGING AND DEMOS)
    if opts.manual:
        def decision_callback(state):
            return get_user_action(state, mdp.get_possible_actions)
    else:
        decision_callback = a.get_action

    # RUN EPISODES
    if opts.episodes > 0:
        print()
        print("RUNNING", opts.episodes, "EPISODES")
        print()
    returns = 0
    for episode in range(1, opts.episodes + 1):
        returns += run_episode(a, env, opts.discount, decision_callback,
                               display_callback, message_callback,
                               pause_callback, episode)
    if opts.episodes > 0:
        print()
        print("AVERAGE RETURNS FROM START STATE: " +
              str((returns + 0.0) / opts.episodes))
        print()
        print()

    # DISPLAY POST-LEARNING VALUES / Q-VALUES
    if opts.agent == 'q' and not opts.manual:
        try:
            display.display_q_values(a, message="Q-VALUES AFTER " +
                                     str(opts.episodes) + " EPISODES")
            display.pause()
            display.display_values(a, message="VALUES AFTER " +
                                              str(opts.episodes) + " EPISODES")
            display.pause()
        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == '__main__':
    main()
