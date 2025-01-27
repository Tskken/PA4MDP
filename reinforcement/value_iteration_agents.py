"""Defines the ValueIterationAgent class for solving known MDPs.

Champlain College CSI-480, Fall 2018
The following code was adapted by Joshua Auerbach (jauerbach@champlain.edu)
from the UC Berkeley Pacman Projects (see license and attribution below).

Author: Dylan Blanchard, Sloan Anderson, and Stephen Johnson
Class: CSI-480-01
Assignment: PA 4 -- MDP's
Due Date: November 5, 2018 11:59 PM

Certification of Authenticity:
I certify that this is entirely my own work, except where I have given
fully-documented references to the work of others. I understand the definition
and consequences of plagiarism and acknowledge that the assessor of this
assignment may, for the purpose of assessing this assignment:
- Reproduce this assignment and provide a copy to another member of academic
- staff; and/or Communicate a copy of this assignment to a plagiarism checking
- service (which may then retain a copy of this assignment on its database for
- the purpose of future plagiarism checking)

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

from learning_agents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """A value iteration agent for solving known MDPs.

    * Please read learning_agents.py before reading this.*

    A ValueIterationAgent takes a Markov decision process (see mdp.py)
    on initialization and runs value iteration for a given number of iterations
    using the supplied discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """Initialize agent with given mdp and run value iteration.

        Your value iteration agent should take an mdp on construction,
        run the indicated number of iterations and then act according to
        the resulting policy.

        Some useful mdp methods you will use:
            mdp.get_states()
            mdp.get_possible_actions(state)
            mdp.get_transition_states_and_probs(state, action)
            mdp.get_reward(state, action, next_state)
            mdp.is_terminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here
        # *** YOUR CODE HERE ***
        for i in range(self.iterations):
            new_values = util.Counter()
            for state in self.mdp.get_states():
                q_val_list = [self.get_q_value(state, action)
                              for action
                              in self.mdp.get_possible_actions(state)]
                if len(q_val_list) != 0:
                    new_values[state] = max(q_val_list)

            self.values = new_values

    def get_value(self, state):
        """Return the value of the state (computed in __init__).

        Overrides learning_agents.ValueEstimationAgent.get_value
        """
        return self.values[state]

    def compute_q_value_from_values(self, state, action):
        """Compute the Q-value of action in state from self.values."""
        # *** YOUR CODE HERE ***"
        # Using list comprehension.
        # This segment is equal to this sudo code:
        #   q_val = 0
        #   for action in transition(state, action):
        #       q_val += (prob * (reword(state, action, next_state) + discount
        #               * Value(next_state)))
        #   return q_val
        return sum(
            [(prob *
                (self.mdp.get_reward(state, action, next_state) +
                 (self.discount * self.get_value(next_state))))
                for next_state, prob
                in self.mdp.get_transition_states_and_probs(state, action)])

    def compute_action_from_values(self, state):
        """Return the optimal action from state using computed values.

        The policy is the best action in the given state according to the
        values currently stored in self.values.

        You may break ties any way you see fit.  Note that if there are no
        legal actions, which is the case at the terminal state, you should
        return None.
        """
        # *** YOUR CODE HERE ***"
        best_action = None
        best_cost = float('-inf')

        for action in self.mdp.get_possible_actions(state):
            q_val = self.get_q_value(state, action)
            if q_val > best_cost:
                best_action = action
                best_cost = q_val

        return best_action

    def get_policy(self, state):
        """Return the best action to take in the state.

        Overrides learning_agents.ValueEstimationAgent.get_policy
        """
        return self.compute_action_from_values(state)

    def get_action(self, state):
        """Return the policy at the state (no exploration here).

        Overrides learning_agents.ValueEstimationAgent.get_action
        """
        return self.compute_action_from_values(state)

    def get_q_value(self, state, action):
        """Return Q-value of (state,action) Q-state.

        Overrides learning_agents.ValueEstimationAgent.get_q_value
        """
        return self.compute_q_value_from_values(state, action)
