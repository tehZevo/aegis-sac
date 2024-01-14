import gymnasium as gym
from gymnasium import spaces
import numpy as np
import threading
import time

from protopost import ProtoPost
from protopost import protopost_client as ppcl
from nd_to_json import nd_to_json, json_to_nd

#TODO: move to separate repo
class AegisEnv(gym.Env):
    def __init__(self, obs_shape, action_shape, port=80, nsteps=None, action_low=-1, action_high=1):

        if type(action_shape) is int:
            self.action_space = spaces.Discrete(action_shape)
        else:
            #self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=action_shape)
            self.action_space = spaces.Box(low=action_low, high=action_high, shape=action_shape)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape)

        self.step_reward = 0 #received rewards

        self.obs_shape = obs_shape
        self.observation = np.zeros(self.obs_shape)
        self.action = None
        #TODO: allow done to be set
        #self.done = False

        self.nsteps = nsteps
        self.step_counter = 0

        self.update_event = threading.Event()
        self.flask_wait = threading.Event()

        self.start_server(port)

    def start_server(self, port):

        def pp_reward(data):
            self.step_reward += data

        def pp_step(data):
            #get observation from data and store for .step later
            self.observation = json_to_nd(data)
            #set update event
            self.update_event.set()
            #wait for step to be ready
            self.flask_wait.wait()
            self.flask_wait.clear()

            return nd_to_json(self.action)

        routes = {
            "reward": pp_reward,
            "step": pp_step
        }

        def start_app():
            ProtoPost(routes).start(port)

        #run flask in separate thread
        thread = threading.Thread(target=start_app)
        thread.daemon = True
        thread.start()

    def step(self, action):
        #set action
        self.action = action

        #wait for step call
        self.flask_wait.set()
        self.update_event.wait()
        self.update_event.clear()

        #flip reward
        r = self.step_reward
        self.step_reward = 0

        done = False
        if self.nsteps is not None and self.step_counter >= self.nsteps:
            done = True

        return self.observation, r, done, False, {}

    def reset(self, **kwargs):
        self.step_counter = 0
        #TODO: figure this out (where should first obs come from?)
        #we could wait for a /step call, but then the action returned would be null
        return self.observation, {}
