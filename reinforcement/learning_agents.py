"""Base classes for learning agents.

Contains the base classes ValueEstimationAgent and QLearningAgent,
which your agents will extend.

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

from game import Agent
import util
import time


class ValueEstimationAgent(Agent):
    """Abstract agent for estimating values for an environment.

    Assigns values to (state,action) Q-Values for an environment. As well as
    a value to a state and a policy given respectively by,

    V(s) = max_{a in actions} Q(s,a)
    policy(s) = arg_max_{a in actions} Q(s,a)

    Both ValueIterationAgent and QLearningAgent inherit from this agent.
    While a ValueIterationAgent has a model of the environment via a
    MarkovDecisionProcess (see mdp.py) that is used to estimate Q-Values before
    ever actually acting, the QLearningAgent estimates Q-Values while acting in
    the environment.
    """

    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, num_training=10):
        """Initialize agent with options, which can be passed via command line.

        Options can be passed in via the Pacman command line using
        -a alpha=0.5,...

        alpha        - learning rate
        epsilon      - exploration rate
        gamma        - discount factor
        num_training - number of training episodes,
                       i.e. no learning after these many episodes
        """
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.num_training = int(num_training)

    ####################################
    #    Override These Functions      #
    ####################################
    def get_q_value(self, state, action):
        """Return Q-value of (state,action) Q-state."""
        util.raise_not_defined()

    def get_value(self, state):
        """Return value of this state under tbe best action.

        Concretely, this is given by

            V(s) = max_{a in actions} Q(s,a)
        """
        util.raise_not_defined()

    def get_policy(self, state):
        """Return the best action to take in the state.

        Note that because we might want to explore, this might not always
        coincide with get_action.

        Concretely, this is given by

            policy(s) = arg_max_{a in actions} Q(s,a)

        If many actions achieve the maximal Q-value, it doesn't matter
        which is selected.
        """
        util.raise_not_defined()

    def get_action(self, state):
        """Choose an action to take from state and return it.

        Note: because we might want to explore, this might not always
        coincide with get_policy.

        Note: legal action available from state.get_legal_actions()
        """
        util.raise_not_defined()


class ReinforcementAgent(ValueEstimationAgent):
    """Abstract ReinforcementAgent that learns from experience.

    A ValueEstimationAgent that estimates Q-Values (as well as policies) from
    experience rather than a model

    What you need to know:
        - The environment will call
            observe_transition(state,action,next_state,delta_reward),
          which will call
            update(state, action, next_state, delta_reward)
          which you should override.

        - Use self.get_legal_actions(state) to know which actions are
        available in a state
    """

    NUM_EPS_UPDATE = 100

    ####################################
    #    Override These Functions      #
    ####################################

    def update(self, state, action, next_state, reward):
        """Will be called after observing a transition and reward.

        You need to override this.
        """
        util.raise_not_defined()

    ####################################
    #    Read These Functions          #
    ####################################

    def get_legal_actions(self, state):
        """Get the actions available for a given state.

        This is what you should use to obtain legal actions for a state.
        """
        return self.action_fn(state)

    def observe_transition(self, state, action, next_state, delta_reward):
        """Observe a (s, a, s') transition.

        This will be called by the environment to inform the agent of
        observed transitions.

        This will result in a call to self.update on the same arguments.

        NOTE: Do *NOT* override or call this function directly
        """
        self.episode_rewards += delta_reward
        self.update(state, action, next_state, delta_reward)

    def start_episode(self):
        """Start a new episode.

        This will be called by the environment when a new episode is starting.
        """
        self.last_state = None
        self.last_action = None
        self.episode_rewards = 0.0

    def stop_episode(self):
        """Stop the episode.

        This will be called by the environment when an episode is done.
        """
        if self.episodes_so_far < self.num_training:
            self.accum_train_rewards += self.episode_rewards
        else:
            self.accum_test_rewards += self.episode_rewards
        self.episodes_so_far += 1
        if self.episodes_so_far >= self.num_training:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning

    def is_in_training(self):
        """Return whether in (first num_training episodes)."""
        return self.episodes_so_far < self.num_training

    def is_in_testing(self):
        """Return whether in testing iteration (i.e. not in training)."""
        return not self.is_in_training()

    def __init__(self, action_fn=None, num_training=100, epsilon=0.5,
                 alpha=0.5, gamma=1):
        """Initialize agent with provided values.

        Args:
            action_fn    : Function which takes a state and returns the list of
                           legal actions
            alpha        : learning rate
            epsilon      : exploration rate
            gamma        : discount factor
            num_training : number of training episodes, i.e. no learning after
                           this many episodes
        """
        if action_fn is None:
            def action_fn(state):
                return state.get_legal_actions()
        self.action_fn = action_fn
        self.episodes_so_far = 0
        self.accum_train_rewards = 0.0
        self.accum_test_rewards = 0.0
        self.num_training = int(num_training)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)

    ################################
    # Controls needed for Crawler  #
    ################################

    def set_epsilon(self, epsilon):
        """Set epsilon (exploration rate)."""
        self.epsilon = epsilon

    def set_learning_rate(self, alpha):
        """Set alpha (learning rate)."""
        self.alpha = alpha

    def set_discount(self, discount):
        """Set gamma (discount factor)."""
        self.discount = discount

    def do_action(self, state, action):
        """Update previous state and action.

        Will be called by inherited class when an action is taken in a state.
        """
        self.last_state = state
        self.last_action = action

    ###################
    # Pacman Specific #
    ###################

    def observation_function(self, state):
        """State where we ended up after our last action.

        The simulation should somehow ensure this is called
        """
        if self.last_state is not None:
            reward = state.get_score() - self.last_state.get_score()
            self.observe_transition(self.last_state, self.last_action, state,
                                    reward)
        return state

    def register_initial_state(self, state):
        """Register the initial state."""
        self.start_episode()
        if self.episodes_so_far == 0:
            print('Beginning %d episodes of Training' % (self.num_training))

    def final(self, state):
        """Finalize episode.

        Called by Pacman game at the terminal state.
        """
        delta_reward = state.get_score() - self.last_state.get_score()
        self.observe_transition(self.last_state, self.last_action, state,
                                delta_reward)
        self.stop_episode()

        # Make sure we have this var
        if 'episode_start_time' not in self.__dict__:
            self.episode_start_time = time.time()
        if 'last_window_accum_rewards' not in self.__dict__:
            self.last_window_accum_rewards = 0.0
        self.last_window_accum_rewards += state.get_score()

        if self.episodes_so_far % ReinforcementAgent.NUM_EPS_UPDATE == 0:
            print('Reinforcement Learning Status:')
            window_avg = (self.last_window_accum_rewards /
                          float(ReinforcementAgent.NUM_EPS_UPDATE))
            if self.episodes_so_far <= self.num_training:
                train_avg = (self.accum_train_rewards /
                             float(self.episodes_so_far))
                print('\tCompleted %d out of %d training episodes' %
                      (self.episodes_so_far, self.num_training))
                print('\tAverage Rewards over all training: %.2f' %
                      (train_avg))
            else:
                test_avg = (float(self.accum_test_rewards) /
                            (self.episodes_so_far - self.num_training))
                print('\tCompleted %d test episodes' %
                      (self.episodes_so_far - self.num_training))
                print('\tAverage Rewards over testing: %.2f' % test_avg)
            print('\tAverage Rewards for last %d episodes: %.2f' %
                  (ReinforcementAgent.NUM_EPS_UPDATE, window_avg))
            print('\tEpisode took %.2f seconds' %
                  (time.time() - self.episode_start_time))
            self.last_window_accum_rewards = 0.0
            self.episode_start_time = time.time()

        if self.episodes_so_far == self.num_training:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
