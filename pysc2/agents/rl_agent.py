from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
from tensorflow import keras

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS


class RL_Agent(base_agent.BaseAgent):
    """ RL agent"""
    def step(self, obs):
        super(RL_Agent, self).step(obs)
        function_id, args = model.sample_action(obs)
        return actions.FunctionCall(function_id, args)