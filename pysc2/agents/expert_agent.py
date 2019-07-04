from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.lib import actions
from pysc2.agents import base_agent
from pysc2.lib import features
import numpy as np

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS

class ExpertAgent(base_agent.BaseAgent):

    def step(self, obs):
        super().step(obs)
        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            player_relative = obs.observation.feature_screen.player_relative
            y,x = (player_relative == _PLAYER_NEUTRAL).nonzero()

            if not x.all():
                return FUNCTIONS.no_op()
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            return FUNCTIONS.Move_screen("now", [x_mean, y_mean])
        else:
            return FUNCTIONS.select_army("select")

class Defeat(base_agent.BaseAgent):
    def step(self, obs):
        super().step(obs)

        if FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
            player_relative = obs.observation.feature_screen.player_relative
            y,x = (player_relative==_PLAYER_ENEMY).nonzero()

            enemy = list(zip(x,y))
            
            
        if FUNCTIONS.select_army.id in obs.observation.available_actions:
            return FUNCTIONS.select_army("select")
        
        return FUNCTIONS.no_op()


        
