"""Feature extractors for Pacman game states.

Used for the approximate Q-learning agent (in q_learning_agents.py).

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

from game import Actions
import util


class FeatureExtractor:
    """Base FeatureExtractor class, define get_features interface."""

    def get_features(self, state, action):
        """Return a dictionary from features to counts.

        Usually, the count will just be 1.0 for indicator functions.
        """
        util.raise_not_defined()


class IdentityExtractor(FeatureExtractor):
    """FeatureExtractor that just uses the (s,a) pair as feature."""

    def get_features(self, state, action):
        """Return a dictionary from features to counts.

        Overrides FeatureExtractor.get_features to just return the (s,a)
        identity feature with count 1.0
        """
        feats = util.Counter()
        feats[(state, action)] = 1.0
        return feats


class CoordinateExtractor(FeatureExtractor):
    """FeatureExtractor that has features for state, action, and coords."""

    def get_features(self, state, action):
        """Return a dictionary from features to counts.

        Overrides FeatureExtractor.get_features to return count of 1.0 for
        state, action, x-coordinate, and y-coordinate.
        """
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats


def closest_food(pos, food, walls):
    """Return the distance to the closest food.

    This is similar to the function that we have worked on in the search
    project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.get_legal_neighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None


class SimpleExtractor(FeatureExtractor):
    """FeatureExtractor that returns features for a basic reflex Pacman.

    The feature are as follows:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def get_features(self, state, action):
        """Return a dictionary from features to counts.

        Overrides FeatureExtractor.get_features to return the features
        described in the class docstring.
        """
        # extract the grid of food and wall locations
        # and get the ghost locations
        food = state.get_food()
        walls = state.get_walls()
        ghosts = state.get_ghost_positions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.get_pacman_position()
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] =\
            sum((next_x, next_y) in Actions.get_legal_neighbors(g, walls)
                for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closest_food((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = (float(dist) /
                                        (walls.width * walls.height))
        features.divide_all(10.0)
        return features
