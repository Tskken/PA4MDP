"""Analysis questions.

Set the given parameters to obtain the specified policies through
value iteration.

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


def question2():
    """Answer question 2."""
    answer_discount = 0.9
    answer_noise = 0.0
    return answer_discount, answer_noise


def question3a():
    """Answer question 3a.

    Optimal policy should prefer the close exit (+1), risking the cliff (-10)

    If not possible, return 'NOT POSSIBLE'
    """
    answer_discount = 0.1
    answer_noise = 0.0
    answer_living_reward = 0.8
    return answer_discount, answer_noise, answer_living_reward


def question3b():
    """Answer question 3b.

    Optimal policy should prefer the close exit (+1),
    but avoiding the cliff (-10)

    If not possible, return 'NOT POSSIBLE'
    """
    answer_discount = 0.2
    answer_noise = 0.2
    answer_living_reward = 0.5
    return answer_discount, answer_noise, answer_living_reward


def question3c():
    """Answer question 3c.

    Optimal policy should prefer the distant exit (+10),
    risking the cliff (-10)

    If not possible, return 'NOT POSSIBLE'
    """
    answer_discount = 0.8
    answer_noise = 0.0
    answer_living_reward = 0.2
    return answer_discount, answer_noise, answer_living_reward


def question3d():
    """Answer question 3d.

    Optimal policy should prefer the distant exit (+10),
    avoiding the cliff (-10)

    If not possible, return 'NOT POSSIBLE'
    """
    answer_discount = 0.8
    answer_noise = 0.2
    answer_living_reward = 0.5
    return answer_discount, answer_noise, answer_living_reward


def question3e():
    """Answer question 3e.

    Optimal policy should avoid both exits and the cliff
    (so an episode should never terminate)

    If not possible, return 'NOT POSSIBLE'
    """
    answer_discount = 0.0
    answer_noise = 0.0
    answer_living_reward = 0.8
    return answer_discount, answer_noise, answer_living_reward


def question6():
    """Answer question 6.

    If not possible, return 'NOT POSSIBLE'
    """
    string = 'NOT POSSIBLE'
    return string


def main():
    """If run as a script just print out answers."""
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))


if __name__ == '__main__':
    main()
