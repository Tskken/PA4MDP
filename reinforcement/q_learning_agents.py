"""Q-learning agents for Gridworld, Crawler and Pacman.

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

        # *** YOUR CODE HERE ***

    def get_q_value(self, state, action):
        """Return Q(state,action).

        Should return 0.0 if we have never seen a state or the Q node value
        otherwise
        """
        # *** YOUR CODE HERE ***
        util.raise_not_defined()

    def compute_value_from_q_values(self, state):
        """Return max_action Q(state,action).

        Where the max is over legal actions.

        Note that if there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        # *** YOUR CODE HERE ***
        util.raise_not_defined()

    def compute_action_from_q_values(self, state):
        """Compute the best action to take in a state.

        Note that if there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        # *** YOUR CODE HERE ***
        util.raise_not_defined()

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
        # legal_actions = self.get_legal_actions(state)
        # action = None
        # *** YOUR CODE HERE ***
        util.raise_not_defined()

        # return action

    def update(self, state, action, next_state, reward):
        """Perform Q-value update.

        The parent class calls this to observe a state = action => next_state
        and reward transition.

        NOTE: You should never call this function, it will be called on your
        behalf
        """
        # *** YOUR CODE HERE ***
        util.raise_not_defined()

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
        util.raise_not_defined()

    def update(self, state, action, next_state, reward):
        """Update weights based on transition."""
        # *** YOUR CODE HERE ***
        util.raise_not_defined()

    def final(self, state):
        """Finalize at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodes_so_far == self.num_training:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
