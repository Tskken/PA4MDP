"""Environment abstract base class definition.

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


from util import raise_not_defined


class Environment:
    """Interface for an abstract environment."""

    def get_current_state(self):
        """Return the current state of enviornment."""
        raise_not_defined()

    def get_possible_actions(self, state):
        """Return possible actions the agent can take in the given state.

        Can return the empty list if we are in a terminal state.
        """
        raise_not_defined()

    def do_action(self, action):
        """Perform the given action in the current environment state.

        Update the enviornment.

        Return a (reward, next_state) pair.
        """
        raise_not_defined()

    def reset(self):
        """Reset the current state to the start state."""
        raise_not_defined()

    def is_terminal(self):
        """Return if the enviornment has entered a terminal state.

        This means there are no successors.
        """
        state = self.get_current_state()
        actions = self.get_possible_actions(state)
        return len(actions) == 0
