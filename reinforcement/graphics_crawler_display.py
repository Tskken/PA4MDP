"""Graphical display for crawling robot.

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

import tkinter
import q_learning_agents
import time
import threading
import sys
import crawler

robot_type = 'crawler'


class Application:
    """Main application class."""

    def sigmoid(self, x):
        """Return sigmoid of x."""
        return 1.0 / (1.0 + 2.0 ** (-x))

    def increment_speed(self, inc):
        """Increment speed by inc."""
        self.tick_time *= inc
        #        self.epsilon = min(1.0, self.epsilon)
        #        self.epsilon = max(0.0,self.epsilon)
        #        self.learner.set_speed(self.epsilon)
        self.speed_label['text'] = 'Step Delay: %.5f' % (self.tick_time)

    def increment_epsilon(self, inc):
        """Increment epsilon by inc."""
        self.ep += inc
        self.epsilon = self.sigmoid(self.ep)
        self.learner.set_epsilon(self.epsilon)
        self.epsilon_label['text'] = 'Epsilon: %.3f' % (self.epsilon)

    def increment_gamma(self, inc):
        """Increment gamma by inc."""
        self.ga += inc
        self.gamma = self.sigmoid(self.ga)
        self.learner.set_discount(self.gamma)
        self.gamma_label['text'] = 'Discount: %.3f' % (self.gamma)

    def increment_alpha(self, inc):
        """Increment alpa by inc."""
        self.al += inc
        self.alpha = self.sigmoid(self.al)
        self.learner.set_learning_rate(self.alpha)
        self.alpha_label['text'] = 'Learning Rate: %.3f' % (self.alpha)

    def __init_gui(self, win):
        # Window #
        self.win = win

        # Initialize Frame #
        win.grid()
        self.dec = -.5
        self.inc = .5
        self.tick_time = 0.1

        # Epsilon Button + Label #
        self.setup_speed_button_and_label(win)

        self.setup_epsilon_button_and_label(win)

        # Gamma Button + Label #
        self.setup_gamma_button_and_label(win)

        # Alpha Button + Label #
        self.setup_alpha_button_and_label(win)

        # Exit Button #
        # self.exit_button = Tkinter.Button(win,text='Quit', command=self.exit)
        # self.exit_button.grid(row=0, column=9)

        # Simulation Buttons #
        #        self.setup_simulation_buttons(win)

        # Canvas #
        self.canvas = tkinter.Canvas(root, height=200, width=1000)
        self.canvas.grid(row=2, columnspan=10)

    def setup_alpha_button_and_label(self, win):
        """Set up the alpha button and its label."""
        self.alpha_minus = tkinter.Button(
            win, text="-", command=(lambda: self.increment_alpha(self.dec)))
        self.alpha_minus.grid(row=1, column=3, padx=10)

        self.alpha = self.sigmoid(self.al)
        self.alpha_label = tkinter.Label(
            win, text='Learning Rate: %.3f' % (self.alpha))
        self.alpha_label.grid(row=1, column=4)

        self.alpha_plus = tkinter.Button(
            win, text="+", command=(lambda: self.increment_alpha(self.inc)))
        self.alpha_plus.grid(row=1, column=5, padx=10)

    def setup_gamma_button_and_label(self, win):
        """Set up the gamme button and its label."""
        self.gamma_minus = tkinter.Button(
            win, text="-", command=(lambda: self.increment_gamma(self.dec)))
        self.gamma_minus.grid(row=1, column=0, padx=10)

        self.gamma = self.sigmoid(self.ga)
        self.gamma_label = tkinter.Label(
            win, text='Discount: %.3f' % (self.gamma))
        self.gamma_label.grid(row=1, column=1)

        self.gamma_plus = tkinter.Button(
            win, text="+", command=(lambda: self.increment_gamma(self.inc)))
        self.gamma_plus.grid(row=1, column=2, padx=10)

    def setup_epsilon_button_and_label(self, win):
        """Set up the epsilon button and its label."""
        self.epsilon_minus = tkinter.Button(
            win, text="-", command=(lambda: self.increment_epsilon(self.dec)))
        self.epsilon_minus.grid(row=0, column=3)

        self.epsilon = self.sigmoid(self.ep)
        self.epsilon_label = tkinter.Label(
            win, text='Epsilon: %.3f' % (self.epsilon))
        self.epsilon_label.grid(row=0, column=4)

        self.epsilon_plus = tkinter.Button(
            win, text="+", command=(lambda: self.increment_epsilon(self.inc)))
        self.epsilon_plus.grid(row=0, column=5)

    def setup_speed_button_and_label(self, win):
        """Set up the speed button and its label."""
        self.speed_minus = tkinter.Button(
            win, text="-", command=(lambda: self.increment_speed(.5)))
        self.speed_minus.grid(row=0, column=0)

        self.speed_label = tkinter.Label(
            win, text='Step Delay: %.5f' % (self.tick_time))
        self.speed_label.grid(row=0, column=1)

        self.speed_plus = tkinter.Button(
            win, text="+", command=(lambda: self.increment_speed(2)))
        self.speed_plus.grid(row=0, column=2)

    def skip_5k_steps(self):
        """Skip 5000 steps."""
        self.steps_to_skip = 5000

    def __init__(self, win):
        """Initialize the application."""
        self.ep = 0
        self.ga = 2
        self.al = 2
        self.step_count = 0
        # Init Gui

        self.__init_gui(win)

        # Init environment
        if robot_type == 'crawler':
            self.robot = crawler.CrawlingRobot(self.canvas)
            self.robot_environment =\
                crawler.CrawlingRobotEnvironment(self.robot)
        else:
            raise Exception("Unknown RobotType")

        # Init Agent
        def action_fn(state):
            return self.robot_environment.get_possible_actions(state)
        self.learner = q_learning_agents.QLearningAgent(action_fn=action_fn)

        self.learner.set_epsilon(self.epsilon)
        self.learner.set_learning_rate(self.alpha)
        self.learner.set_discount(self.gamma)

        # Start GUI
        self.running = True
        self.stopped = False
        self.steps_to_skip = 0
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def exit(self):
        """Exit the application."""
        self.running = False
        for i in range(5):
            if not self.stopped:
                time.sleep(0.1)
        try:
            self.win.destroy()
        except Exception:
            pass
        sys.exit(0)

    def step(self):
        """Step the application."""
        self.step_count += 1

        state = self.robot_environment.get_current_state()
        actions = self.robot_environment.get_possible_actions(state)
        if len(actions) == 0.0:
            self.robot_environment.reset()
            state = self.robot_environment.get_current_state()
            actions = self.robot_environment.get_possible_actions(state)
            print('Reset!')
        action = self.learner.get_action(state)
        if action is None:
            raise Exception('None action returned: Code Not Complete')
        next_state, reward = self.robot_environment.do_action(action)
        self.learner.observe_transition(state, action, next_state, reward)

    def animate_policy(self):
        """Animate the policy (currently unused)."""
        if robot_type != 'pendulum':
            raise Exception('Only pendulum can animate_policy')

        tot_width = self.canvas.winfo_reqwidth()
        tot_height = self.canvas.winfo_reqheight()

        length = 0.48 * min(tot_width, tot_height)
        x, y = tot_width - length - 30, length + 10

        angle_min, angle_max = self.robot.get_min_and_max_angle()
        vel_min, vel_max = self.robot.get_min_and_max_angle_velocity()

        if 'animate_policy_box' not in dir(self):
            self.canvas.create_line(x, y, x + length, y)
            self.canvas.create_line(x + length, y, x + length, y - length)
            self.canvas.create_line(x + length, y - length, x, y - length)
            self.canvas.create_line(x, y - length, x, y)
            self.animate_policy_box = 1
            self.canvas.create_text(x + length / 2, y + 10,
                                    text='angle')
            self.canvas.create_text(x - 30, y - length / 2,
                                    text='velocity')
            self.canvas.create_text(x - 60, y - length / 4,
                                    text='Blue = kick_left')
            self.canvas.create_text(x - 60, y - length / 4 + 20,
                                    text='Red = kick_right')
            self.canvas.create_text(x - 60, y - length / 4 + 40,
                                    text='White = do_nothing')

        angle_delta = (angle_max - angle_min) / 100
        vel_delta = (vel_max - vel_min) / 100
        for i in range(100):
            angle = angle_min + i * angle_delta

            for j in range(100):
                vel = vel_min + j * vel_delta
                state = self.robot_environment.get_state(angle, vel)
                max, arg_max = None, None
                if not self.learner.seen_state(state):
                    arg_max = 'unseen'
                else:
                    for action in ('kick_left', 'kick_right', 'do_nothing'):
                        q_val = self.learner.get_q_value(state, action)
                        if max is None or q_val > max:
                            max, arg_max = q_val, action
                if arg_max != 'unseen':
                    if arg_max == 'kick_left':
                        color = 'blue'
                    elif arg_max == 'kick_right':
                        color = 'red'
                    elif arg_max == 'do_nothing':
                        color = 'white'
                    dx = length / 100.0
                    dy = length / 100.0
                    x0, y0 = x + i * dx, y - j * dy
                    self.canvas.create_rectangle(x0, y0, x0 + dx, y0 + dy,
                                                 fill=color)

    def run(self):
        """Run the application."""
        self.step_count = 0
        self.learner.start_episode()
        while True:
            min_sleep = .01
            tm = max(min_sleep, self.tick_time)
            time.sleep(tm)
            self.steps_to_skip = int(tm / self.tick_time) - 1

            if not self.running:
                self.stopped = True
                return
            for i in range(self.steps_to_skip):
                self.step()
            self.steps_to_skip = 0
            self.step()
#          self.robot.draw()
        self.learner.stop_episode()

    def start(self):
        """Start the main loop."""
        self.win.mainloop()


def run():
    """Run the application (wrapper)."""
    global root
    root = tkinter.Tk()
    root.title('Crawler GUI')
    root.resizable(0, 0)
    #  root.mainloop()

    app = Application(root)

    def update_gui():
        app.robot.draw(app.step_count, app.tick_time)
        root.after(10, update_gui)
    update_gui()

    root.protocol('WM_DELETE_WINDOW', app.exit)
    try:
        app.start()
    except Exception:
        app.exit()
