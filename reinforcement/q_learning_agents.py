"""Q-learning agents for Gridworld, Crawler and Pacman.

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

from learning_agents import ReinforcementAgent
# noqa on the following since not directly used, but needed for
# ApproximateQAgent
from feature_extractors import IdentityExtractor, CoordinateExtractor  # noqa
from feature_extractors import SimpleExtractor  # noqa
import util


class QLearningAgent(ReinforcementAgent):
    """Q-Learning Agent: ReinforcementAgent implementing q-learning.

    Functions you should fill in:
        - compute_value_from_q_values
        - compute_action_from_q_values
        - get_q_value
        - get_action
        - update

    Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

    Functions you should use
        - self.get_legal_actions(state),
            which returns legal actions for a state
    """

    def __init__(self, **kwargs):
        """Initialize Q-values here.

        Extends learning_agents.ReinforcementAgent
        """
        ReinforcementAgent.__init__(self, **kwargs)
        self.counter = util.Counter()

    def get_q_value(self, state, action):
        """Return Q(state,action).

        Should return 0.0 if we have never seen a state or the Q node value
        otherwise
        """
        return self.counter[(state, action)]

    def compute_value_from_q_values(self, state):
        """Return max_action Q(state,action).

        Where the max is over legal actions.

        Note that if there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        actions = self.get_legal_actions(state)
        if len(actions) == 0:
            return 0.0
        max_value = float("-inf")
        for action in actions:
            if max_value <= self.get_q_value(state, action) or \
                    max_value == float("-inf"):
                max_value = self.get_q_value(state, action)

        return max_value

    def compute_action_from_q_values(self, state):
        """Compute the best action to take in a state.

        Note that if there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        actions = self.get_legal_actions(state)
        if len(actions) == 0:
            return None
        max_value = float("-inf")
        state_action = ""
        for action in actions:
            if max_value <= self.get_q_value(state, action) or \
                    max_value == float("-inf"):
                max_value = self.get_q_value(state, action)
                state_action = action

        return state_action

    def get_action(self, state):
        """Compute the action to take in the current state.

        With probability self.epsilon, we should take a random action and
        take the best policy action otherwise.

        Note that if there are no legal actions, which is the case at the
        terminal state, you should choose None as the action.

        HINT 1: You might want to use util.flip_coin(prob)
        HINT 2: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_actions = self.get_legal_actions(state)
        if len(legal_actions) == 0:
            return None

        import random
        if util.flip_coin(self.epsilon):
            action = random.choice(legal_actions)
        else:
            action = self.compute_action_from_q_values(state)

        return action

    def update(self, state, action, next_state, reward):
        """Perform Q-value update.

        The parent class calls this to observe a state = action => next_state
        and reward transition.

        NOTE: You should never call this function, it will be called on your
        behalf
        """
        self.counter[(state, action)] = \
            ((1 - self.alpha) *
                self.get_q_value(state, action)) + \
            (self.alpha * (reward + self.discount *
                           self.compute_value_from_q_values(next_state)))

    def get_policy(self, state):
        """Return the best action to take in the state.

        Overrides: learning_agents.ValueEstimationAgent.get_policy
        """
        return self.compute_action_from_q_values(state)

    def get_value(self, state):
        """Return value of this state under tbe best action.

        Overrides: learning_agents.ValueEstimationAgent.get_value
        """
        return self.compute_value_from_q_values(state)


class PacmanQAgent(QLearningAgent):
    """A QLearningAgent with different default parameters for Pacman."""

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, num_training=0,
                 **kwargs):
        """Initialize with parameters.

        The default parameters can be changed from the pacman.py command line.

        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha        - learning rate
        epsilon      - exploration rate
        gamma        - discount factor
        num_training - number of training episodes,
                       i.e. no learning after these many episodes
        """
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['num_training'] = num_training
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **kwargs)

    def get_action(self, state):
        """Call get_action of QLearningAgent and inform parent of action.

        Do not change or remove this method.
        """
        action = QLearningAgent.get_action(self, state)
        self.do_action(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """Agent for approximate Q-learning.

    You should only have to overwrite get_q_value and update.
    All other QLearningAgent functions should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **kwargs):
        """Initialize with default feature extractor."""
        self.feature_extractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **kwargs)
        self.weights = util.Counter()

    def get_weights(self):
        """Return feature weights."""
        return self.weights

    def get_q_value(self, state, action):
        """Return approximate Q-value: Q(state,action) = w * feature_vector.

        Note: * here is the dot_product operator
        """
        # *** YOUR CODE HERE ***
        q_value = 0.0
        feature_list = self.feature_extractor.get_features(state, action)
        for feat_key in feature_list.keys():
            q_value = q_value + self.weights[feat_key] * feature_list[feat_key]
        return q_value

    def update(self, state, action, next_state, reward):
        """Update weights based on transition."""
        # *** YOUR CODE HERE ***
        feature_list = self.feature_extractor.get_features(state, action)
        difference = reward + self.discount * self.get_value(next_state) - \
            self.get_q_value(state, action)
        for feat_key in feature_list.keys():
            self.weights[feat_key] = self.weights[feat_key] + self.alpha * \
                                     difference * feature_list[feat_key]

    def final(self, state):
        """Finalize at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodes_so_far == self.num_training:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
